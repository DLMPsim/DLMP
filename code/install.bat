@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo DLMP installation
echo ==========================================
echo.

REM Check that Python exists
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found in PATH.
    echo Please install Python 3.10 or 3.11, then run this script again.
    echo.
    pause
    exit /b 1
)

REM Capture Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Detected Python version: %PYVER%
echo.

REM Enforce tested Python versions for pinned PyTorch stack
echo %PYVER% | findstr /r "^3\.10\." >nul
if %errorlevel%==0 goto version_ok

echo %PYVER% | findstr /r "^3\.11\." >nul
if %errorlevel%==0 goto version_ok

echo ERROR: This installer supports Python 3.10 or 3.11 only.
echo Your current Python version is %PYVER%.
echo.
echo Reason:
echo The pinned CPU-only PyTorch version used by DLMP
echo (torch 2.2.2+cpu, torchvision 0.17.2) is not available
echo for newer Python versions such as 3.14.
echo.
echo Please install Python 3.10 or 3.11 and make sure that
echo "python" in your terminal points to that installation.
echo.
pause
exit /b 1

:version_ok
echo Python version is compatible.
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto pip_error
echo.

REM Install core requirements
echo Installing packages from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 goto req_error
echo.

REM Install pinned CPU-only PyTorch
echo Installing CPU-only PyTorch...
python -m pip install torch==2.2.2+cpu torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto torch_error
echo.

echo ==========================================
echo DLMP installation completed successfully.
echo ==========================================
echo.
pause
exit /b 0

:pip_error
echo.
echo ERROR: pip upgrade failed.
echo Please check your Python installation and internet connection.
echo.
pause
exit /b 1

:req_error
echo.
echo ERROR: Installation from requirements.txt failed.
echo Please review requirements.txt and your internet connection.
echo.
pause
exit /b 1

:torch_error
echo.
echo ERROR: CPU-only PyTorch installation failed.
echo This usually means:
echo - your Python version is not supported by torch 2.2.2+cpu, or
echo - your internet connection blocked the download.
echo.
echo Recommended fix:
echo Use Python 3.10 or 3.11, then run this installer again.
echo.
pause
exit /b 1
