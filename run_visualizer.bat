@echo off
echo.
echo  BetWinninGames - Iniciando servidor local...
echo  Abre: http://localhost:8080
echo  Pulsa Ctrl+C para parar el servidor.
echo.

:: Abre el navegador tras 1 segundo
start "" timeout /t 1 /nobreak >nul
start "" "http://localhost:8080"

:: Arranca el servidor desde la carpeta visualizador
cd /d "%~dp0visualizador"
python -m http.server 8080
