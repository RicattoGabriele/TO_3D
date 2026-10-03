@echo off
title Setup Ambiente CE-3D
echo ===================================================================
echo   Installazione Dipendenze CE-3D per Nuove Macchine
echo ===================================================================
echo.
echo 1. Verifica versione Python...
python --version
if errorlevel 1 (
    echo [ERRORE] Python non trovato! Assicurati di installare Python (versione 3.10 o successiva)
    echo e spunta la casella "Add Python to PATH" durante l'installazione.
    pause
    exit /b 1
)

echo.
echo 2. Aggiornamento pip e installazione delle librerie necessarie...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ===================================================================
echo   Installazione completata con successo!
echo   Ora puoi fare doppio clic su "Avvia_Dashboard.bat" per eseguire l'app.
echo ===================================================================
pause
