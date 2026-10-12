@echo off
cd /d "%~dp0"
call venv\Scripts\activate
start "" "%~dp0index.html"
python -m uvicorn main:app --reload
pause
