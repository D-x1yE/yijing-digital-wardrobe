@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8770/studio.html
E:\555\envs\sd_torch\python.exe serve_studio.py
pause
