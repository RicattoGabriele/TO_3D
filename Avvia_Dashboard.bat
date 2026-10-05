@echo off
title CE-3D Launcher
cd /d "%~dp0"
echo ========================================================
echo Avvio della Dashboard CE-3D...
echo ========================================================

rem Termina eventuali istanze residue che occupano la porta 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

python -m streamlit run dashboard/app.py --server.port=8501
pause
