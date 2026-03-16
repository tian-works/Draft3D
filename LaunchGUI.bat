@echo off

set "VENV_DIR=%~dp0venv"

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

set "PYTHON_EXECUTABLE=%VENV_DIR%\Scripts\python.exe"

set "PATH=%VENV_DIR%\Scripts;%PATH%"

echo Launching GUI...
"%PYTHON_EXECUTABLE%" GUI.py

deactivate

pause

