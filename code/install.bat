@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ==========================================
echo DLMP installation (Python 3.10 + NumPy 1.x)
echo ==========================================
echo.

cd /d "%~dp0"

set "PY_EXE="

for /f "delims=" %%i in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do set "PY_EXE=%%i"

if not defined PY_EXE (
    echo [ERROR] Python 3.10 was not found.
    echo Install Python 3.10 and rerun.
    goto :fail
)

echo Using Python 3.10:
echo %PY_EXE%
echo.

"%PY_EXE%" -m pip install --upgrade pip setuptools wheel

echo Installing NumPy 1.26.4...
"%PY_EXE%" -m pip uninstall -y numpy >nul 2>nul
"%PY_EXE%" -m pip install numpy==1.26.4

if exist requirements.txt (
    "%PY_EXE%" -m pip install -r requirements.txt
) else (
    "%PY_EXE%" -m pip install mesa==2.3.2 matplotlib PyQt5 scikit-learn pandas pillow
)

echo Installing PyTorch CPU...
"%PY_EXE%" -m pip install torch==2.2.2+cpu torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

echo Re-pin NumPy...
"%PY_EXE%" -m pip install --force-reinstall numpy==1.26.4

echo Verifying...
"%PY_EXE%" -c "import numpy, torch; print('NumPy:', numpy.__version__); print('Torch:', torch.__version__)"

echo Creating launcher...
> dlmp.bat (
    echo @echo off
    echo cd /d "%%~dp0"
    echo py -3.10 GUIDLMP.py
    echo pause
)

echo ==========================================
echo INSTALL COMPLETE
echo ==========================================

pause
exit /b 0

:fail
echo Installation failed.
pause
exit /b 1
