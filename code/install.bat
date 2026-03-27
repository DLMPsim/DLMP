@echo off
cd /d "%~dp0"

echo Installing DLMP dependencies...

python -m pip install --upgrade pip

REM Install core dependencies
python -m pip install -r requirements.txt

REM Install PyTorch CPU separately
python -m pip install torch==2.2.2+cpu torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installation finished.
pause