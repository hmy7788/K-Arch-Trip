@echo off
rem Starts the server and a public https link (trycloudflare) with a QR code.
rem Keep this window open while people use the link. Run setup.bat once before this.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\share.ps1" %*
pause
