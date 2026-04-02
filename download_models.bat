@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo      Resound Studio - Model Downloader Helper
echo ============================================================
echo.

:: Check for backend venv
if not exist "apps\api\venv" (
    echo [ERROR] Backend virtual environment not found in 'apps\api\venv'.
    echo Please run setup.bat first.
    pause
    exit /b
)

echo [1/3] Activating backend virtual environment...
call apps\api\venv\Scripts\activate

echo [2/3] Ensuring huggingface-hub is installed...
python -m pip install huggingface-hub -q

echo [3/3] Running model downloader script...
python apps\api\download_models.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Model download failed.
    echo Please check your internet connection and ensure no Firewall is blocking 'huggingface.co'.
) else (
    echo.
    echo [SUCCESS] All models are ready!
)

echo.
echo ============================================================
pause
