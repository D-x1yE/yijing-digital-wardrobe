@echo off
cd /d "%~dp0"
start "" "http://127.0.0.1:8765/real.html"
"E:\555\envs\sd_torch\python.exe" "%~dp0serve_real.py"
pause
