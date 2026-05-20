@echo off

set "VENV_DIR=%~dp0venv"

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

set "PYTHON_EXECUTABLE=%VENV_DIR%\Scripts\python.exe"

set "PATH=%VENV_DIR%\Scripts;%PATH%"

REM Install dependencies on first run (or when requirements change)
if not exist "%VENV_DIR%\._draft3d_deps_installed" (
    echo Installing Draft3D dependencies...
    "%PYTHON_EXECUTABLE%" -m pip install --upgrade pip
    "%PYTHON_EXECUTABLE%" -m pip install -r "%~dp0requirements.txt"
    echo ok>"%VENV_DIR%\._draft3d_deps_installed"
)

echo Launching GUI...
"%PYTHON_EXECUTABLE%" GUI.py

deactivate

pause

