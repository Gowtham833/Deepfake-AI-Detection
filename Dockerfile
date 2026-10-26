# ============================================================
# Dockerfile for Deepfake Detection API (Railway Deployment)
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app

# 1. Install OS packages required by OpenCV and FFmpeg for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Pre-install CPU-only PyTorch & Torchvision
# Keeps Docker image lean (~1GB instead of ~4.5GB with CUDA) and speeds up build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Copy application codebase and models
COPY . .

# 5. Expose default port
EXPOSE 8000

# 6. Launch FastAPI server listening on dynamic Railway $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
