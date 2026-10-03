@echo off
title CE-3D Dashboard Launcher
echo ========================================================
echo Avvio della Dashboard CE-3D (Ottimizzazione Topologica)
echo ========================================================
start "" http://localhost:8501
python -m streamlit run dashboard/app.py --server.port=8501
pause
