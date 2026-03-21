@echo off
REM =========================================================================
REM  Photo Colorizer — Windows setup script
REM  Target GPU: NVIDIA RTX 3070 (Ampere, 8 GB VRAM, CUDA 12.1)
REM =========================================================================

echo.
echo === Photo Colorizer Setup ===
echo.

REM ---- Create virtual environment ----------------------------------------
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM ---- Activate venv ------------------------------------------------------
call venv\Scripts\activate.bat

REM ---- Upgrade pip --------------------------------------------------------
echo Upgrading pip...
python -m pip install --upgrade pip

REM ---- Install PyTorch with CUDA 12.1 ------------------------------------
echo.
echo Installing PyTorch with CUDA 12.1 support (for RTX 3070)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM ---- Install remaining dependencies ------------------------------------
echo.
echo Installing remaining dependencies...
pip install -r requirements.txt

REM ---- Optional: attempt DDColor -----------------------------------------
echo.
echo Attempting to install DDColor (optional — falls back to colorizers)...
pip install ddcolor 2>nul || echo DDColor not available; colorizers will be used.

REM ---- Verify CUDA -------------------------------------------------------
echo.
echo === CUDA verification ===
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo Setup complete!  Run the app with:
echo   venv\Scripts\activate.bat
echo   python app.py
echo.
pause
