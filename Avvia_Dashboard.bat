@echo off
title CE-3D Launcher
cd /d "%~dp0"
echo ========================================================
echo Avvio della Dashboard CE-3D...
echo ========================================================
python -m streamlit run dashboard/app.py --server.port=8501
pause
