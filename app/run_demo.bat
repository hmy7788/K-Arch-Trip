@echo off
cd /d "%~dp0.."
echo Looking for a python with torch + streamlit ...
set PY=
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File tools\pyfind.ps1`) do set PY=%%P
if "%PY%"=="" (
  echo.
  echo [!] No python with torch + streamlit was found.
  echo     Open "Miniforge Prompt" ^(or Anaconda Prompt^), activate your env, then:
  echo         pip install -r requirements.txt
  echo     and run this file again ^(or run it from that prompt^).
  pause
  exit /b 1
)
for %%I in ("%PY%") do set PYDIR=%%~dpI
set PATH=%PYDIR%;%PYDIR%Scripts;%PYDIR%Library\bin;%PATH%
echo Using: %PY%
echo.
echo Starting the demo (checkpoints\model).
echo If the browser does not open by itself, go to http://localhost:8501
echo Press Ctrl+C in this window to stop it.
echo.
"%PY%" -m streamlit run app\demo.py -- --checkpoint checkpoints\model
pause
