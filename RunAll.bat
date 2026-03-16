@echo off
chcp 65001 >nul
setlocal

set "BASEDIR=%~dp0"

echo Launching ComfyUI backend...
start "ComfyUI" cmd /k call "%BASEDIR%LaunchComfyUI.bat"

echo Waiting 8 seconds for backend to initialize...
timeout /t 8 /nobreak >nul

echo Launching GUI frontend...
start "GUI" cmd /k call "%BASEDIR%LaunchGUI.bat"

exit

