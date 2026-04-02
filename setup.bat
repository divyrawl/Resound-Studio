@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo      Resound Studio - One-Click Setup Utility
echo ============================================================
echo.

:: 1. Core Dependency Check
echo [1/6] Checking core dependencies...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in system PATH. Please install Python 3.10 or 3.11.
    pause
    exit /b
)

where pnpm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pnpm not found. Please install pnpm globally: 'npm install -g pnpm'
    pause
    exit /b
)

:: 2. System Binaries Check (FFmpeg)
echo [2/6] Checking for FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] FFmpeg not found. It is required for audio processing.
    set /p install_ffmpeg="Would you like to try installing FFmpeg via winget? (y/n): "
    if /i "!install_ffmpeg!"=="y" (
        winget install ffmpeg
        if %ERRORLEVEL% neq 0 echo [ERROR] Winget installation failed. Please install FFmpeg manually.
    ) else (
        echo [INFO] Please install FFmpeg manually and add it to your PATH.
    )
) else (
    echo [OK] FFmpeg found.
)

:: 3. Frontend Setup
echo [3/6] Installing Frontend dependencies (pnpm)...
call pnpm install
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pnpm install failed.
    pause
    exit /b
)

:: 4. Backend Setup (Virtual Environment)
echo [4/6] Setting up Backend Python environment...
cd apps\api
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate

echo [4a] Upgrading core build tools...
python -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Failed to upgrade pip/setuptools. This usually happens due to Firewall/Antivirus (WinError 10013).
    echo Please ensure python.exe is allowed to access the internet.
)

echo [4b] Installing CUDA-optimized PyTorch (cu121)...
:: We use --no-cache-dir to prevent issues with corrupted downloads (Hash mismatch)
python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
if %ERRORLEVEL% neq 0 (
    echo [ERROR] PyTorch installation failed. 
    echo Check your internet connection and ensure no Firewall is blocking 'download.pytorch.org'.
    pause
    exit /b
)

echo [4c] Installing requirements.txt...
pip install -r requirements.txt --no-cache-dir

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Backend dependency installation failed.
    echo.
    echo Troubleshooting Tips:
    echo 1. Disable Antivirus/Firewall temporarily.
    echo 2. run 'python -m pip install --upgrade pip setuptools' manually.
    echo 3. Delete 'apps\api\venv' and try again.
    echo.
    pause
    exit /b
)
cd ..\..

:: 5. Environment Configuration
echo [5/6] Configuring environment variables...
if not exist ".env" (
    echo Creating root .env from .env.example...
    copy .env.example .env
)

if not exist "apps\web\.env.local" (
    echo Creating apps\web\.env.local...
    echo NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 > apps\web\.env.local
)

:: 6. Model Pre-download (Optional)
echo [6/7] Pre-downloading AI models (Qwen3-TTS 1.7B Base)...
echo This is highly recommended to avoid download delays during first use.
set /p download_models="Would you like to download the required 1.7B Base model now? (y/n): "
if /i "!download_models!"=="y" (
    call download_models.bat
) else (
    echo [INFO] Skipping pre-download. Models will be downloaded on first use.
)

:: 7. Final Check
echo [7/7] Finalizing setup...
echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo You can now start the application using:
echo    run.bat
echo.
echo If you have issues with GPU acceleration later, run:
echo    fix_gpu.bat
echo.
echo ============================================================
pause
