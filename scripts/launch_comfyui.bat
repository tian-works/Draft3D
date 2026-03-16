@echo off

rem Wrapper script that reuses the existing top-level LaunchComfyUI.bat
rem so the new project layout remains backwards compatible.

set "BASEDIR=%~dp0"
cd /d "%BASEDIR%.."

call LaunchComfyUI.bat

