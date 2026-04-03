#!/bin/bash
# ============================================================
# Space Debris Collision Risk Predictor — Setup Script
# ============================================================
# Run this after cloning the repo to install all dependencies
# and download the pre-trained model checkpoints.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

echo "========================================"
echo " Space Debris Collision Risk Predictor"
echo " Setup Script"
echo "========================================"
echo ""

# --- 1. Python dependencies ---
echo "[1/4] Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install -r backend/requirements.txt
    echo "  Done."
else
    echo "  ERROR: pip not found. Install Python 3.10+ first."
    exit 1
fi

# --- 2. PyTorch (CUDA optional) ---
echo ""
echo "[2/4] Checking PyTorch..."
python -c "import torch; print(f'  PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "  PyTorch not found. Installing CPU version..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo "  NOTE: For GPU support, install the CUDA version instead:"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
}

# --- 3. Frontend dependencies ---
echo ""
echo "[3/4] Installing frontend dependencies..."
if command -v npm &> /dev/null; then
    cd frontend && npm install && cd ..
    echo "  Done."
else
    echo "  WARNING: npm not found. Install Node.js 18+ to run the frontend."
fi

# --- 4. Model checkpoints ---
echo ""
echo "[4/4] Checking model checkpoints..."
MODEL_DIR="data/models"
MODEL_FILE="$MODEL_DIR/best_model.pth"

if [ -f "$MODEL_FILE" ]; then
    echo "  Model checkpoints already present. Skipping download."
else
    echo "  Model checkpoints not found!"
    echo ""
    echo "  The pre-trained model files (~290MB total) are too large for GitHub."
    echo "  Download them from the link below and place them in data/models/:"
    echo ""
    echo "  >>> https://drive.google.com/drive/folders/YOUR_FOLDER_ID <<<"
    echo ""
    echo "  Required files:"
    echo "    - best_model.pth          (~60MB)"
    echo "    - ckpt_ep039_auc0.9999.pth (~60MB)"
    echo "    - ckpt_ep041_auc0.9999.pth (~60MB)"
    echo "    - ckpt_ep048_auc0.9999.pth (~60MB)"
    echo "    - last.pth                (~60MB)"
    echo ""
    echo "  After downloading, place all .pth files in: $MODEL_DIR/"
    echo ""
    echo "  NOTE: The backend requires at least best_model.pth to run."
fi

# --- Done ---
echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " To start the app:"
echo "   1. Backend:  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "   2. Frontend: cd frontend && npm run dev"
echo "   3. Open:     http://localhost:5173"
echo "========================================"
