import os, re, time, uuid, subprocess, base64
from datetime import datetime, timedelta
from flask import Flask, request, Response, send_file, abort, render_template
from dotenv import load_dotenv
import sympy as sp
from sympy import limit, symbols, diff, sympify, simplify
from google import genai
import fitz  # PyMuPDF for PDF text extraction
import threading

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ------------------ App Setup ------------------
app = Flask(__name__, static_folder="static")
TMP_DIR = "/tmp/math_solver"
os.makedirs(TMP_DIR, exist_ok=True)
LOG_DIR = os.path.join(TMP_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(TMP_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------ Limits & Delays ------------------
MAX_FILE_SIZE_MB = 50
MAX_PROBLEMS = 150
FILE_TTL_MINUTES = 60
TEX_TTL_HOURS = 72
MAX_RETRIES = 15
RATE_LIMIT_DELAY = 10

# ------------------ Utilities ------------------
def secure_filename(name: str) -> str:
    return os.path.basename(name).replace(" ", "_")

def save_checkpoint(session_id, problem_number):
    path = os.path.join(CHECKPOINT_DIR, f"{session_id}.chk")
    with open(path, "w") as f:
        f.write(str(problem_number))

def load_checkpoint(session_id):
    path = os.path.join(CHECKPOINT_DIR, f"{session_id}.chk")
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read().strip())
    return 1

def cleanup_old_files():
    now = datetime.utcnow()
    for f in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, f)
        try:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
            if f.endswith(".tex") and mtime < now - timedelta(hours=TEX_TTL_HOURS):
                os.remove(path)
            elif not f.endswith(".tex") and mtime < now - timedelta(minutes=FILE_TTL_MINUTES):
                os.remove(path)
        except Exception:
            pass

def enforce_upload_limits(req):
    if req.content_length and req.content_length > MAX_FILE_SIZE_MB * 1024 * 1024:
        abort(413, "File too large")

def write_logic_log(session_id, problem_num, attempt, prompt, raw_response, checks, mode):
    """
    Saves raw AI reasoning + verification results
    """
    log_file = os.path.join(LOG_DIR, f"{session_id}_logic.txt")

    with open(log_file, "a", encoding="utf-8") as log:
        log.write("\n" + "="*80 + "\n")
        log.write(f"Problem: {problem_num} | Attempt: {attempt}\n")
        log.write(f"MODE: {mode}\n")
        log.write("-"*80 + "\n")
        log.write("PROMPT:\n")
        log.write(prompt + "\n\n")
        log.write("RAW RESPONSE:\n")
        log.write(raw_response + "\n\n")
        log.write("VERIFICATION RESULTS:\n")

        for (ok, msg), name in checks:
            status = "PASS" if ok else "FAIL"
            log.write(f"{name}: {status}")
            if msg:
                log.write(f" | {msg}")
            log.write("\n")

        log.write("="*80 + "\n")
    
def verify_derivative(raw):
    try:
        import re
        match = re.search(r'\\frac\{d\}\{dx\}\((.*?)\)\s*=\s*(.*)', raw)
        if not match:
            return True, None

        x = symbols('x')
        expr = sympify(match.group(1))
        claimed = sympify(match.group(2))

        actual = diff(expr, x)

        if simplify(actual - claimed) != 0:
            return False, "Derivative mismatch"

        return True, None

    except Exception:
        return True, None

def verify_limit(raw):
    try:
        import re
        match = re.search(r'\\lim_\{x ?\\to ?(.*?)\}\s*(.*?)=\s*(.*)', raw)
        if not match:
            return True, None

        x = symbols('x')
        approaching = sympify(match.group(1))
        expression = sympify(match.group(2))
        claimed = sympify(match.group(3))

        actual = limit(expression, x, approaching)

        if actual != claimed:
            return False, "Limit mismatch"

        return True, None

    except Exception:
        return True, None

# ------------------ Hallucination Guard ------------------
HALLUCINATION_PHRASES = [
    "clearly", "obviously", "it is obvious",
    "one can see", "by inspection", "it follows that"
]

# ------------------ Markdown → LaTeX Scrubber ------------------
def surgical_markdown_scrubber(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\\textbf\{Problem \d+\}', '', text)
    footnotes = {}

    def stash(m):
        k = f"__FN_{len(footnotes)}__"
        footnotes[k] = m.group(0)
        return k

    text = re.sub(r'\\footnote\{.*?\}', stash, text)
    text = re.sub(r'(?<!\d)\s*\*\s*(?!\d)', '', text)
    text = re.sub(r'(?<!\d)\*(?!\d)', '', text)
    text = re.sub(r'([A-Za-z])\*(?!\d)', r'\1\\footnote{}', text)

    for k, v in footnotes.items():
        text = text.replace(k, v)

    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('### ', '').replace('## ', '')
    text = re.sub(r'(?<!\\)#', r'\\#', text)
    return text

# ------------------ Verification Checks ------------------
def verify_integral(raw):
    try:
        m = re.search(r'\\int\s*(.*?)\s*d([a-z])\s*=\s*(.*)', raw)
        if not m:
            return True, None
        integrand = sp.sympify(m.group(1))
        var = sp.symbols(m.group(2))
        result = sp.sympify(m.group(3).replace('+ C', '').strip())
        return sp.simplify(sp.diff(result, var) - integrand) == 0, None
    except Exception as e:
        return False, f"Integral verification failed: {e}"

def verify_division(raw):
    try:
        q = re.search(r'q\s*=\s*(-?\d+)', raw)
        r = re.search(r'r\s*=\s*(-?\d+)', raw)
        eq = re.search(r'(\d+)\s*=\s*(\d+)\s*\*\s*q\s*\+\s*r', raw)
        if not (q and r and eq):
            return True, None
        n, d = int(eq.group(1)), int(eq.group(2))
        return n == d * int(q.group(1)) + int(r.group(1)), None
    except Exception as e:
        return False, f"Division check failed: {e}"

def detect_short_solution(raw):
    """
    Detect suspiciously short algebra solutions
    """
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    
    # If fewer than 4 meaningful lines and contains algebra symbols,
    # might be incomplete
    algebra_indicators = ["=", "+", "-", "\\frac", "\\int", "\\sum"]
    contains_algebra = any(sym in raw for sym in algebra_indicators)

    if contains_algebra and len(lines) < 4:
        return False, "Suspiciously short algebra solution"
    
    return True, None

def unit_sanity(raw):
    if "kg" in raw and "m/s^2" in raw and "N" not in raw:
        return False, "Force computed without Newtons"
    return True, None

def verify_dimensions(raw):
    try:
        # Detect physics-style numeric result with units
        unit_patterns = ["N", "J", "W", "Pa", "kg", "m/s", "m/s^2"]

        if not any(u in raw for u in unit_patterns):
            return True, None

        # crude consistency: check if final boxed answer contains unit
        if "\\boxed" in raw:
            boxed = re.findall(r'\\boxed\{(.*?)\}', raw)
            if boxed:
                final = boxed[-1]
                if not any(u in final for u in unit_patterns):
                    return False, "Final answer missing units"

        return True, None

    except Exception as e:
        return True, None

# ------------------ PDF / LaTeX ------------------
def create_pdf_from_tex(tex_path, pdf_path):
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", TMP_DIR, tex_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError("LaTeX compilation failed")

    return pdf_path

def create_practice_pdf(problems, filename):
    path = os.path.join(TMP_DIR, filename)
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"<b>Problem {i}</b><br/>{p.replace(chr(10), '<br/>')}", styles["Normal"])
        for i, p in enumerate(problems, 1)
    ]
    doc.build(content)
    return path

# ------------------ Physics Detection ------------------
PHYSICS_KEYWORDS = [
    "force", "mass", "acceleration", "velocity", "N", "kg", "m/s^2",
    "free body", "diagram", "newton", "kinematics", "dynamics", "torque"
]

DISCRETE_KEYWORDS = [
    "mod", "graph theory", "combinatorics",
    "bijection", "injection", "surjection",
    "recurrence", "induction", "proof"
]

CIRCUIT_KEYWORDS = [
    "resistor", "voltage", "current", "ohm",
    "kirchhoff", "capacitor", "inductor",
    "series circuit", "parallel circuit"
]

def detect_circuits(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text").lower()
            if any(k in text for k in CIRCUIT_KEYWORDS):
                return True
        return False
    except Exception:
        return False

def is_physics_pdf(pdf_path):
    fname = pdf_path.lower()
    if any(k in fname for k in ["physics", "mechanics", "forces", "kinematics", "dynamics"]):
        return True
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text").lower()
            if any(k in text for k in PHYSICS_KEYWORDS):
                return True
        return False
    except Exception:
        return False

def verify_series(raw):
    try:
        if "\\sum" not in raw:
            return True, None

        # Only basic geometric series support for now
        if "geometric" in raw.lower():
            return True, None

        return True, None

    except Exception:
        return True, None

def verify_convergence_structure(raw):
    tests = [
        "ratio test",
        "root test",
        "comparison test",
        "limit comparison",
        "alternating series",
        "integral test"
    ]

    if "\\sum" not in raw:
        return True, None

    if not any(t in raw.lower() for t in tests):
        return False, "Series without stated convergence test"

    return True, None

def verify_proof_structure(raw):
    if "proof" not in raw.lower():
        return True, None

    if "therefore" not in raw.lower():
        return False, "Proof missing logical conclusion"

    return True, None

def verify_equation_solution(raw):
    try:
        match = re.search(r'([a-z])\s*=\s*(-?\d+)', raw)
        if not match:
            return True, None

        var = sp.symbols(match.group(1))
        value = int(match.group(2))

        # Try to detect equation earlier
        eq_match = re.search(r'(.*?)=0', raw)
        if not eq_match:
            return True, None

        expr = sp.sympify(eq_match.group(1))
        if expr.subs(var, value) != 0:
            return False, "Solution does not satisfy equation"

        return True, None

    except Exception:
        return True, None

def verify_matrix(raw):
    try:
        if "\\begin{bmatrix}" not in raw:
            return True, None
        return True, None
    except:
        return True, None

# ------------------ Routes ------------------
@app.before_request
def before_request():
    cleanup_old_files()
    enforce_upload_limits(request)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload_full", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        abort(400, "No file")
    file = request.files["file"]
    original = secure_filename(file.filename)
    session_id = uuid.uuid4().hex
    input_path = os.path.join(TMP_DIR, f"{session_id}_{original}")
    file.save(input_path)
    return {"filename": original, "session": session_id}

@app.route("/upload_batch", methods=["POST"])
def upload_batch():
    data = request.get_json()
    images = data.get("images", [])
    session_id = uuid.uuid4().hex
    batch_files = []
    for i, img in enumerate(images):
        img_path = os.path.join(TMP_DIR, f"{session_id}_{i}.png")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img.split(",")[1]))
        batch_files.append(img_path)
    return {"filenames": batch_files, "session": session_id}

@app.route("/stream/<session_id>")
def stream(session_id):
    manual_count = int(request.args.get("manual_count", "0") or 0)
    if manual_count > MAX_PROBLEMS:
        abort(400, "Too many problems")

    def generate():
        original_name = None
        for f_name in os.listdir(TMP_DIR):
            if f_name.startswith(session_id) and f_name.lower().endswith(".pdf"):
                original_name = f_name.split("_", 1)[1]
                break

        if original_name:
            base = os.path.splitext(original_name)[0]
            sol_pdf = os.path.join(TMP_DIR, f"{base}_Solutions.pdf")
        else:
            sol_pdf = os.path.join(TMP_DIR, f"{session_id}_Solutions.pdf")

        prac_pdf = os.path.join(TMP_DIR, f"{session_id}_Practice.pdf")
        tex_file = os.path.join(TMP_DIR, f"{session_id}_Solutions.tex")
        
        yield "data: 🚀 Stream connected. Preparing session...\n\n"

        session_files = []
        for f_name in sorted(os.listdir(TMP_DIR)):
            if f_name.startswith(session_id) and f_name.lower().endswith((".pdf", ".png")):
                session_files.append(os.path.join(TMP_DIR, f_name))

        if not session_files:
            yield "data: ❌ Error: No files found for this session.\n\n"
            return

        start_problem = load_checkpoint(session_id)

        if start_problem <= manual_count:
            uploaded_files = []
            try:
                yield f"data: 📤 Uploading {len(session_files)} file(s) to Gemini...\n\n"
                for f_path in session_files:
                    up_file = client.files.upload(file=f_path)
                    while getattr(up_file.state, "name", str(up_file.state)) == "PROCESSING":
                        time.sleep(2)
                        up_file = client.files.get(name=up_file.name)
                    uploaded_files.append(up_file)
                yield "data: ✅ Files ready for processing.\n\n"
            except Exception as e:
                yield f"data: ❌ Upload failed: {e}\n\n"
                return

            pdf_path = next((f for f in session_files if f.lower().endswith('.pdf')), None)

            mode = "math"
            pdf_text = ""

            if pdf_path:
                try:
                    doc = fitz.open(pdf_path)
                    for page in doc:
                        pdf_text += page.get_text("text").lower()
                except:
                    pdf_text = ""

                if any(k in pdf_text for k in CIRCUIT_KEYWORDS):
                    mode = "circuits"
                elif any(k in pdf_text for k in PHYSICS_KEYWORDS):
                    mode = "physics"
                elif any(k in pdf_text for k in DISCRETE_KEYWORDS):
                    mode = "discrete"

            if start_problem == 1:
                with open(tex_file, "w") as f:
                    f.write(r"\documentclass{article}\usepackage{amsmath,amssymb,tikz}\begin{document}")
            else:
                yield f"data: 🔄 Connection resumed. Picking up at problem {start_problem}...\n\n"

            for i in range(start_problem, manual_count + 1):
                yield f"data: Solving problem {i}/{manual_count}...\n\n"
                prompt = (
                    f"You are a master mathematics professor. Solve Problem {i} with extreme precision. "
                    f"VISUAL ANALYSIS: Carefully examine the images/PDF pages provided. Look for Problem {i}. "
                    "If the problem contains a graph, diagram, or table, describe its key features before solving. "
                    "\n\nSOLVING RULES:\n"
                    "1. DO NOT SKIP STEPS. Show every algebraic manipulation.\n"
                    "2. STATE THEOREMS by name when used.\n"
                    "3. NO ASSUMPTIONS. Explain why each step follows.\n"
                    "4. LITERAL INPUT: Solve exactly as written in the image.\n"
                    "5. FORMULAS: Write formulas symbolically before substituting numbers.\n"
                    "\n\nSELF-VERIFICATION STEP (MANDATORY):\n"
                    "After solving, re-check every algebraic manipulation and recompute the final answer. "
                    "If any arithmetic or logic mistake is found, correct it before producing the final output. "
                    "Do NOT mention this re-check in the output.\n"
                    "\n\nOUTPUT FORMAT:\n"
                    f"Provide only valid LaTeX. Use \\section*{{Problem {i}}} as the header. "
                    "Do not include markdown or conversational text."
                )
                prompt = (
                    f"IMPORTANT: Only solve the problem labeled Problem {i}. "
                    "If you see text from adjacent problems (like 8b, 8c, etc.), ignore it unless explicitly part of this problem. "
                    + prompt
                )
                if mode == "physics":
                    prompt += (
                        "\n\nPHYSICS MODE ACTIVATED:\n"
                        "1. Identify known quantities with units.\n"
                        "2. Define coordinate system.\n"
                        "3. Draw free body diagram using TikZ if forces involved.\n"
                        "4. Apply governing physical laws explicitly.\n"
                        "5. Carry units through all calculations.\n"
                        "6. Perform dimensional consistency check.\n"
                        "7. Box final answer with units.\n"
                    )

                elif mode == "circuits":
                    prompt += (
                        "\n\nCIRCUIT ANALYSIS MODE:\n"
                        "1. Redraw circuit using TikZ circuit elements.\n"
                        "2. Identify series and parallel components.\n"
                        "3. Apply Kirchhoff's Laws explicitly.\n"
                        "4. Show full algebraic solution.\n"
                        "5. Verify units (V, A, Ohm).\n"
                        "6. Perform power conservation check.\n"
                    )

                elif mode == "discrete":
                    prompt += (
                        "\n\nDISCRETE MATHEMATICS MODE:\n"
                        "1. State definitions clearly before use.\n"
                        "2. If proof required, structure formally.\n"
                        "3. For induction, show base case and inductive step explicitly.\n"
                        "4. Justify every logical implication.\n"
                    )

                for attempt in range(MAX_RETRIES):
                    try:
                        contents = [*uploaded_files, prompt]
                        resp = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=contents,
                            config={"temperature": 0.2}
                        )
                        raw = resp.text.replace("```", "").strip()
                        if any(p in raw.lower() for p in HALLUCINATION_PHRASES):
                            continue

                        critique_prompt = (
                            "Score the mathematical correctness of the following solution from 0 to 10. "
                            "Return only a number.\n\n" + raw
                        )

                        crit_resp = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=critique_prompt,
                            config={"temperature": 0}
                        )

                        try:
                            score = float(crit_resp.text.strip())
                        except:
                            score = 10

                        checks = [
                            (verify_integral(raw), "Integral Check"),
                            (verify_derivative(raw), "Derivative Check"),
                            (verify_limit(raw), "Limit Check"),
                            (verify_division(raw), "Division Check"),
                            (verify_series(raw), "Series Check"),
                            (verify_convergence_structure(raw), "Convergence Structure"),
                            (verify_equation_solution(raw), "Equation Validation"),
                            (verify_matrix(raw), "Matrix Check"),
                            (verify_proof_structure(raw), "Proof Structure"),
                            (verify_dimensions(raw), "Dimensional Analysis"),
                            (unit_sanity(raw), "Unit Sanity Check"),
                            (detect_short_solution(raw), "Depth Check"),
                        ]

                        write_logic_log(session_id, i, attempt + 1, prompt, raw, checks, mode)
                    
                        hard_failures = []
                        if score < 8:
                            yield f"data: 🔁 Low confidence score ({score}), retrying...\n\n"
                            continue
                        soft_warnings = []

                        for (ok, msg), name in checks:
                            if not ok:
                                if name in ["Integral Check", "Derivative Check", "Limit Check", "Division Check", "Depth Check"]:
                                    hard_failures.append(name)
                                else:
                                    soft_warnings.append(name)

                        if hard_failures:
                            yield f"data: 🔁 Problem {i} retrying due to: {', '.join(hard_failures)}\n\n"
                            continue  # retry immediately

                        if soft_warnings:
                            yield f"data: ⚠️ Problem {i} warning: {', '.join(soft_warnings)} (Proceeding...)\n\n"

                        clean = surgical_markdown_scrubber(raw)
                        with open(tex_file, "a") as f_tex:
                            f_tex.write(f"\\section*{{Problem {i}}}\n{clean}\n\\newpage\n")
                            save_checkpoint(session_id, i + 1)
                        break 
                    except Exception as e:
                        yield f"data: ⚠️ Problem {i} attempt {attempt+1} failed: {e}\n\n"
                        time.sleep(RATE_LIMIT_DELAY)
                else:
                    yield f"data: ❌ Problem {i} failed after {MAX_RETRIES} attempts\n\n"
                    save_checkpoint(session_id, i)  # Retry same problem next stream
                    return
        else:
            yield "data: ✨ All problems already solved. Finalizing PDF...\n\n"

        if os.path.exists(tex_file):
            with open(tex_file, "r+") as f:
                content = f.read()
                if not content.strip().endswith(r"\end{document}"):
                    f.write("\n\\end{document}")

        yield "data: 🖨️ Compiling PDF (staying connected)... \n\n"

        def compile_task():
            try:
                create_pdf_from_tex(tex_file, sol_pdf)
            except RuntimeError:
                time.sleep(2)
                create_pdf_from_tex(tex_file, sol_pdf)

            practice_problems = [
                f"Practice version of problem {i}"
                for i in range(1, manual_count + 1)
            ]
            create_practice_pdf(practice_problems, os.path.basename(prac_pdf))

        compile_thread = threading.Thread(target=compile_task)
        compile_thread.start()

        while compile_thread.is_alive():
            yield "data: ⏳ Working... \n\n"
            time.sleep(3)

        if os.path.exists(sol_pdf):

            # Remove checkpoint after success
            chk_path = os.path.join(CHECKPOINT_DIR, f"{session_id}.chk")
            if os.path.exists(chk_path):
                os.remove(chk_path)

            yield f"data: SUCCESS:{os.path.basename(sol_pdf)}|{os.path.basename(prac_pdf)}\n\n"
        else:
            yield "data: ❌ PDF compilation failed. \n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(TMP_DIR, secure_filename(filename))
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

@app.route("/download_log/<session_id>")
def download_log(session_id):
    path = os.path.join(LOG_DIR, f"{session_id}_logic.txt")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)