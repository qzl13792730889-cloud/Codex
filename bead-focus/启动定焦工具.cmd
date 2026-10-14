@echo off
setlocal
cd /d "%~dp0"
set "BEAD_FOCUS_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BEAD_FOCUS_PY%" (
  "%BEAD_FOCUS_PY%" focus_gui.py
  if errorlevel 1 pause
  exit /b
)
py -3 focus_gui.py
if errorlevel 1 pause
