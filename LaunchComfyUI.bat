@echo off
set BASEDIR=%~dp0
cd /d %BASEDIR%
cd ComfyUI

call venv\Scripts\activate.bat
REM Install ComfyUI dependencies on first run (or when requirements change)
if not exist "%BASEDIR%venv\._comfyui_deps_installed" (
    echo Installing ComfyUI dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r "%BASEDIR%ComfyUI\requirements.txt"
    python -m pip install -r "%BASEDIR%ComfyUI\manager_requirements.txt"
    echo ok>"%BASEDIR%venv\._comfyui_deps_installed"
)
start "" http://127.0.0.1:8188
python main.py
start "" http://127.0.0.1:8188
pause
