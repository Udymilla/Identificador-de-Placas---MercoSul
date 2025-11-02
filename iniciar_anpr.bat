@echo off
title ANPR Demo - Iniciar Servidores
echo ===============================================
echo     Sistema ANPR - Reconhecimento de Placas
echo ===============================================
echo.

 Caminho base do projeto
cd d CUsersTIDocumentsanpr_demo_template

 --- Inicia o BACKEND (porta 8000) ---
echo Iniciando backend...
start BACKEND - FastAPI (porta 8000) cmd k cd backend && uvicorn appapp --reload --port 8000

timeout t 3 nul

 --- Inicia o FRONTEND (porta 8080) ---
echo Iniciando frontend...
start FRONTEND - HTTP Server (porta 8080) cmd k cd frontend && python -m http.server 8080

echo.
echo Servidores iniciados com sucesso!
echo.
echo   Acesse o sistema em httplocalhost8080
echo.
pause
