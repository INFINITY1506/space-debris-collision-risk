@echo off
REM ============================================================
REM Space Debris Collision Risk Predictor — Setup Script (Windows)
REM ============================================================
REM Run this after cloning the repo to install all dependencies.
REM
REM Usage: double-click setup.bat or run from terminal
REM ============================================================

echo ========================================
echo  Space Debris Collision Risk Predictor
echo  Setup Script (Windows)
echo ========================================
echo.

REM --- 1. Python dependencies ---
echo [1/4] Installing Python dependencies...
pip install -r backend\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: pip failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)
echo   Done.

REM --- 2. PyTorch check ---
echo.
echo [2/4] Checking PyTorch...
python -c "import torch; print(f'  PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   PyTorch not found. Installing CPU version...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo   NOTE: For GPU support, install CUDA version instead:
    echo     pip install torch --index-url https://download.pytorch.org/whl/cu121
)

REM --- 3. Frontend dependencies ---
echo.
echo [3/4] Installing frontend dependencies...
where npm >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd frontend
    call npm install
    cd ..
    echo   Done.
) else (
    echo   WARNING: npm not found. Install Node.js 18+ to run the frontend.
)

REM --- 4. Model checkpoints ---
echo.
echo [4/4] Checking model checkpoints...
if exist "data\models\best_model.pth" (
    echo   Model checkpoints already present. Skipping.
) else (
    echo   Model checkpoints not found!
    echo.
    echo   The pre-trained model files (~290MB total) are too large for GitHub.
    echo   Download them and place in data\models\:
    echo.
    echo   Required files:
    echo     - best_model.pth            (~60MB)
    echo     - ckpt_ep039_auc0.9999.pth  (~60MB)
    echo     - ckpt_ep041_auc0.9999.pth  (~60MB)
    echo     - ckpt_ep048_auc0.9999.pth  (~60MB)
    echo     - last.pth                  (~60MB)
    echo.
    echo   NOTE: The backend requires at least best_model.pth to run.
)

REM --- Done ---
echo.
echo ========================================
echo  Setup complete!
echo.
echo  To start the app:
echo    1. Backend:  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo    2. Frontend: cd frontend ^&^& npm run dev
echo    3. Open:     http://localhost:5173
echo ========================================
echo.
pause
