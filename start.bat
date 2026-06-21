@echo off
:: Re-launch ourselves inside cmd /k so the window NEVER closes on its own
if not defined STARTED_VIA_WRAPPER (
    set STARTED_VIA_WRAPPER=1
    cmd /k "%~f0" %*
    exit /b
)

cd /d "%~dp0"
title PDF Papers AI Agent
echo.
echo  ====================================
echo   PDF Papers AI Agent - Quick Start
echo  ====================================
echo.

:: Check Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    goto :done
)

:: Start database services
echo [1/5] Starting database services (MongoDB, Qdrant, Neo4j)...
docker compose up -d mongodb qdrant neo4j
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Docker services.
    goto :done
)

:: Create venv if it doesn't exist
if not exist .venv\Scripts\python.exe (
    echo [2/5] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Is Python 3.11+ installed?
        goto :done
    )
) else (
    echo [2/5] Virtual environment already exists.
)

:: Install dependencies
echo [3/5] Installing dependencies (this may take a minute on first run)...
.venv\Scripts\pip.exe install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    goto :done
)

:: Create .env if it doesn't exist
if not exist ".env" (
    echo [3/5] Creating .env from template...
    copy ".env.example" ".env" >nul
    echo.
    echo  ============================================
    echo   NOTE: Edit .env and set your LLM_API_KEY
    echo   for GraphRAG to work. The app will still
    echo   start without it ^(Hybrid Search works fine^).
    echo  ============================================
    echo.
)

:: Wait for databases to be ready
echo [4/5] Waiting for databases to be ready...
ping -n 11 127.0.0.1 >nul

:: Seed data if papers folder is empty or doesn't exist
dir /b papers\*.pdf >nul 2>&1
if %errorlevel% neq 0 (
    echo [4/5] Seeding sample data ^(downloading papers + building graph^)...
    .venv\Scripts\python.exe seed_data.py
) else (
    echo [4/5] Sample data already exists.
)

:: Start the server and open the browser
echo [5/5] Starting server on http://localhost:8001 ...
echo.
echo  ==========================================
echo   Opening browser to http://localhost:8001
echo   Press Ctrl+C in this window to stop.
echo  ==========================================
echo.
start http://localhost:8001
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001

:done
echo.
echo  Server stopped. This window will stay open.
echo.
