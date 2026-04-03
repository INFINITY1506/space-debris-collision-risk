# ============================================================
# Space Debris Collision Risk Predictor — Railway Deployment
# ============================================================
# Multi-stage build: frontend (static) + backend (FastAPI)
# ============================================================

# --- Stage 1: Build frontend ---
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Backend + serve frontend ---
FROM python:3.12-slim
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch to save memory)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r backend/requirements.txt && \
    pip install --no-cache-dir huggingface_hub

# Copy application code
COPY backend/ ./backend/
COPY training/ ./training/
COPY data/ ./data/
COPY download_models.py ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Download model checkpoints from HuggingFace
ARG HF_REPO_ID=infinity1506/space-debris-models
ENV HF_REPO_ID=${HF_REPO_ID}
RUN python download_models.py

# Expose port (Railway sets PORT env var)
ENV PORT=8000
EXPOSE ${PORT}

# Start backend (serves API + static frontend)
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
