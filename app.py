import os, re, time, uuid, subprocess
from datetime import datetime, timedelta
from flask import Flask, request, Response, send_file, abort, send_from_directory
from dotenv import load_dotenv
import sympy as sp
from google import genai
import fitz  # PyMuPDF for PDF text extraction

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ------------------ App Setup ------------------
app = Flask(__name__, static_folder="static")
TMP_DIR = "/tmp/math_solver"
os.makedirs(TMP_DIR, exist_ok=True)

# ------------------ Limits & Delays ------------------
MAX_FILE_SIZE_MB = 50
MAX_PROBLEMS = 150
FILE_TTL_MINUTES = 60
TEX_TTL_HOURS = 72
MAX_RETRIES = 15
RATE_LIMIT_DELAY = 10  # seconds between Gemini calls

# ------------------ Utilities ------------------
def secure_filename(name: str) -> str:
    return os.path.basename(name).replace(" ", "_")

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
        if not m: return True, None
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
        if not (q and r and eq): return True, None
        n, d = int(eq.group(1)), int(eq.group(2))
        return n == d * int(q.group(1)) + int(r.group(1)), None
    except Exception as e:
        return False, f"Division check failed: {e}"

def step_consistency(raw):
    answers = re.findall(r'(q|r|x|y)\s*=\s*[-\w]+', raw)
    for a in answers:
        if raw.count(a) < 2:
            return False, f"Answer {a} not derived earlier"
    return True, None

def unit_sanity(raw):
    if "kg" in raw and "m/s^2" in raw and "N" not in raw:
        return False, "Force computed without Newtons"
    return True, None

# ------------------ PDF / LaTeX ------------------
def create_pdf_from_tex(tex_path, pdf_path):
    subprocess.run(["pdflatex", "-output-directory", TMP_DIR, tex_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pdf_path

def create_practice_pdf(problems, filename):
    path = os.path.join(TMP_DIR, filename)
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [Paragraph(f"<b>Problem {i}</b><br/>{p.replace(chr(10), '<br/>')}", styles["Normal"])
               for i, p in enumerate(problems, 1)]
    doc.build(content)
    return path

# ------------------ Physics Detection ------------------
PHYSICS_KEYWORDS = [
    "force", "mass", "acceleration", "velocity", "N", "kg", "m/s^2",
    "free body", "diagram", "newton", "kinematics", "dynamics", "torque"
]

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

# ------------------ Routes ------------------
@app.before_request
def before_request():
    cleanup_old_files()
    enforce_upload_limits(request)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/upload_full", methods=["POST"])
def upload_file():
    if "file" not in request.files: abort(400, "No file")
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
    original_name = secure_filename(data.get("original_name", "batch.pdf"))
    session_id = uuid.uuid4().hex
    batch_files = []
    for i, img in enumerate(images):
        img_path = os.path.join(TMP_DIR, f"{session_id}_{i}.png")
        with open(img_path, "wb") as f:
            f.write(bytes(img.split(",")[1], "utf-8"))
        batch_files.append(img_path)
    return {"filenames": batch_files, "session": session_id}

@app.route("/stream/<session_id>")
def stream(session_id):
    manual_count = int(request.args.get("manual_count", "0") or 0)
    if manual_count > MAX_PROBLEMS: abort(400, "Too many problems")
    tex_file = os.path.join(TMP_DIR, f"{session_id}_Solutions.tex")
    sol_pdf = os.path.join(TMP_DIR, f"{session_id}_Solutions.pdf")
    prac_pdf = os.path.join(TMP_DIR, f"{session_id}_Practice.pdf")

    def generate():
        if os.path.exists(tex_file):
            yield f"data: Found existing .tex, regenerating PDFs...\n\n"
            create_pdf_from_tex(tex_file, sol_pdf)
            practice = [f"Practice version of problem {i}" for i in range(1, manual_count+1)]
            create_practice_pdf(practice, os.path.basename(prac_pdf))
            yield f"data: SUCCESS:{os.path.basename(sol_pdf)}|{os.path.basename(prac_pdf)}\n\n"
            return
        with open(tex_file, "w") as f:
            f.write(r"\documentclass{article}\usepackage{amsmath,amssymb,tikz}\begin{document}")
        pdf_path = None
        for f_name in os.listdir(TMP_DIR):
            if f_name.startswith(session_id) and f_name.lower().endswith(".pdf"):
                pdf_path = os.path.join(TMP_DIR, f_name)
                break
        physics_flag = pdf_path and is_physics_pdf(pdf_path)
        for i in range(1, manual_count+1):
            yield f"data: Solving problem {i}/{manual_count}...\n\n"
            prompt = f"Solve problem {i}."
            if physics_flag:
                prompt += " Include a free body diagram using TikZ in LaTeX. No markdown."
            for attempt in range(MAX_RETRIES):
                try:
                    contents = [prompt]
                    if physics_flag: contents.insert(0, pdf_path)
                    resp = client.models.generate_content(model="gemini-2.5-flash-lite",
                                                          contents=contents)
                    raw = resp.text.replace("```", "").strip()
                    if any(p in raw.lower() for p in HALLUCINATION_PHRASES): continue
                    checks = [verify_integral(raw), verify_division(raw),
                              step_consistency(raw), unit_sanity(raw)]
                    errors = [e for ok, e in checks if not ok]
                    if errors: yield f"data: ❌ Problem {i}: {errors[0]}\n\n"; continue
                    clean = surgical_markdown_scrubber(raw)
                    with open(tex_file, "a") as f_tex:
                        f_tex.write(f"\\section*{{Problem {i}}}\n{clean}\n\\newpage\n")
                    break
                except Exception as e:
                    yield f"data: ⚠️ Problem {i} attempt {attempt+1} failed: {e}\n\n"
                    time.sleep(RATE_LIMIT_DELAY)
            else:
                yield f"data: ❌ Problem {i} failed after {MAX_RETRIES} attempts\n\n"
        with open(tex_file, "a") as f:
            f.write(r"\end{document}")
        create_pdf_from_tex(tex_file, sol_pdf)
        practice = [f"Practice version of problem {i}" for i in range(1, manual_count+1)]
        create_practice_pdf(practice, os.path.basename(prac_pdf))
        yield f"data: SUCCESS:{os.path.basename(sol_pdf)}|{os.path.basename(prac_pdf)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(TMP_DIR, secure_filename(filename))
    if not os.path.exists(path): abort(404)
    return send_file(path, as_attachment=True)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)