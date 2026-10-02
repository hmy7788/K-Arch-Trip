@echo off
rem First-time setup: creates .venv, installs packages, downloads cloudflared, checks the model file.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1" %*
pause
