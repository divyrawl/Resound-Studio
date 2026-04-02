@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo      Resound Studio - Pro System Launcher
echo ============================================================

:: 1. Core Dependency Check
echo [1/5] Checking core dependencies...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in system PATH.
    pause
    exit /b
)

where pnpm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pnpm not found. Please install pnpm: 'npm install -g pnpm'
    pause
    exit /b
)

:: 2. Environment Check
echo [2/5] Verifying backend virtual environment...
if not exist "apps\api\venv" (
    echo [ERROR] 'apps\api\venv' missing. Run 'fix_gpu.bat' once to setup your environment.
    pause
    exit /b
)

echo Testing for CUDA...
.\apps\api\venv\Scripts\python -c "import torch; print('CUDA Available: ' + str(torch.cuda.is_available()))" > temp_cuda.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python environment is broken. Please run 'fix_gpu.bat'.
    pause
    exit /b
)
set /p CUDA_STATUS=<temp_cuda.txt
del temp_cuda.txt
echo %CUDA_STATUS%

if "%CUDA_STATUS%"=="CUDA Available: False" (
    echo.
    echo [WARNING] Running on CPU. Performance will be very slow.
    echo If you have an NVIDIA GPU, please run 'fix_gpu.bat' now.
    echo.
)

:: 3. Model Check
echo [3/5] Verifying and downloading required models (Qwen3-TTS 1.7B Base)...
call download_models.bat

:: 4. Launch Backend
echo [4/5] Starting Backend API (Port 8000)...
start "Resound Backend" cmd /k "cd apps\api && venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"

:: 5. Launch Frontend
echo [5/5] Starting Frontend Web App (Port 3000)...
start "Resound Frontend" cmd /k "cd apps\web && pnpm dev"

echo.
echo ------------------------------------------------------------
echo ALL SYSTEMS BOOTING
echo ------------------------------------------------------------
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000
echo ------------------------------------------------------------
echo.
echo Opening browser in 5 seconds...
timeout /t 5 >nul
start http://localhost:3000
echo.
echo Keep those windows open while using the app!
pause
