@echo off
:: BetWinninGames — Predicciones del fin de semana
:: Configura en Task Scheduler para ejecutar Vie/Sab a las 10:00

cd /d C:\Users\MSI\Desktop\REPOS\BetWinninGames

set LOGFILE=logs\main_%date:~-4,4%-%date:~-7,2%-%date:~0,2%.log

echo [%date% %time%] Iniciando predicciones... >> %LOGFILE%
python main.py >> %LOGFILE% 2>&1
echo [%date% %time%] Finalizado. >> %LOGFILE%
