@echo off
cd /d "%~dp0"
E:\555\envs\sd_torch\python.exe start_avatar_demo.py
if errorlevel 1 pause
