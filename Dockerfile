# Use an official Python image
FROM python:3.11-slim

# Install dependencies for LaTeX and system utilities
RUN apt-get update && \
    apt-get install -y \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port Render uses
EXPOSE $PORT

# Run the app with Gunicorn (shell form so $PORT expands correctly)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1