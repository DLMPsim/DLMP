@echo off
cd /d "%~dp0"
python -c "import sys; print(sys.executable)"
python GUIDLMP_v2.py
pause
