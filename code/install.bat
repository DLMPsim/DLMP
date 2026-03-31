@echo off
cd /d "%~dp0"

echo Checking Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python and try again.
    pause
    exit
)

echo.
echo Installing DLMP dependencies...
echo.

python -m pip install --upgrade pip

echo Installing core packages...
python -m pip install -r requirements.txt

echo Installing CPU-only PyTorch...
python -m pip install torch==2.2.2+cpu torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

echo.
echo DLMP installation finished.
pause
