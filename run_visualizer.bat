@echo off
echo.
echo  BetWinninGames - Iniciando servidor local...
echo  Abre: http://localhost:8080
echo  Pulsa Ctrl+C para parar el servidor.
echo.

:: Abre el navegador tras 1 segundo
start "" timeout /t 1 /nobreak >nul
start "" "http://localhost:8080"

:: Arranca el servidor con Cache-Control: no-cache para que
:: predictions.js / tracker_data.js se recarguen siempre frescos
cd /d "%~dp0"
python visualizador\server.py 8080
