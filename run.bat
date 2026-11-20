@echo off
rem Starts NeuroAssist (after install.ps1). Opens http://127.0.0.1:7860 in the browser.
rem   run.bat          this computer only
rem   run.bat --lan    also reachable from a phone on the same Wi-Fi (inference stays on this computer)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Not installed yet. Run:  powershell -ExecutionPolicy Bypass -File install.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m neuroassist.installer --check
".venv\Scripts\python.exe" -m neuroassist.ui %*
