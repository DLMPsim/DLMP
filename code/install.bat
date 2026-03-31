@echo off
cd /d "%~dp0"

echo Checking Python version...

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v

echo Detected Python version: %PYVER%

echo %PYVER% | findstr /r "^3\.10" >nul
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.10 is required.
    echo Please install Python 3.10 and ensure it is on PATH.
    echo.
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
