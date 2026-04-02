@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo      Resound Studio - One-Click Docker Setup
echo ============================================================
echo.

:: Check for Docker
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker not found. Please install Docker Desktop (https://www.docker.com/products/docker-desktop/)
    pause
    exit /b
)

:: Check for Docker Compose
where docker-compose >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] 'docker-compose' not found. Checking for 'docker compose'...
    docker compose version >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Docker Compose not found.
        pause
        exit /b
    )
    set DOCKER_COMPOSE=docker compose
) else (
    set DOCKER_COMPOSE=docker-compose
)

:: Model download pre-flight
echo [1/3] Pre-downloading the Qwen3-TTS 1.7B Base model to the host...
echo This is required to avoid long build times and ensure offline availability.
call download_models.bat

echo [2/3] Building and starting containers (This may take a few minutes)...
%DOCKER_COMPOSE% up -d --build

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker build failed.
    pause
    exit /b
)

echo.
echo [3/3] Finalizing setup...
echo.
echo ============================================================
echo DOCKER SETUP COMPLETE!
echo ============================================================
echo.
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000
echo - Logs:     %DOCKER_COMPOSE% logs -f
echo.
echo ============================================================
pause
