# ============================================================
# Dockerfile for Deepfake AI Detection Platform
# Optimized for Hugging Face Spaces (2 vCPU, 16 GB RAM) & Cloud
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

# 1. Install OS packages required by OpenCV, PIL, and FFmpeg for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Create non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 3. Pre-install CPU-only PyTorch & Torchvision
# Keeps the Docker build fast and avoids CUDA overhead on CPU instances
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application codebase and model checkpoint with user permissions
COPY --chown=user:user . .

# 6. Ensure cache and temporary upload directories are fully accessible
RUN mkdir -p /tmp /app/data /app/models && \
    chmod 777 /tmp && \
    chown -R user:user /app /tmp

# 7. Switch to non-root user
USER user

# 8. Expose default Hugging Face Spaces port (7860)
EXPOSE 7860

# 9. Launch FastAPI server listening on dynamic $PORT (defaults to 7860 for Hugging Face Spaces)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
