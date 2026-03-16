@echo off
set BASEDIR=%~dp0
cd /d %BASEDIR%
cd ComfyUI

call venv\Scripts\activate.bat
start "" http://127.0.0.1:8188
python main.py
start "" http://127.0.0.1:8188
pause
