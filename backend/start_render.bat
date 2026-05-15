@echo off
REM ============================================
REM RENDER DEPLOYMENT STARTUP SCRIPT (Windows)
REM ============================================
REM This script starts the SmartAgri-AI backend
REM with optimized memory settings for deployment

setlocal enabledelayedexpansion

echo [STARTUP] SmartAgri-AI Backend - Optimized Mode
echo ==================================================

REM Environment configuration
set ENVIRONMENT=production
set RENDER=true
set LOW_MEMORY_MODE=true
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1

REM Get PORT from environment or default to 8000
if defined PORT (
    set "PORT=%PORT%"
) else (
    set "PORT=8000"
)

echo [CONFIG] PORT: %PORT%
echo [CONFIG] ENVIRONMENT: production
echo [CONFIG] LOW_MEMORY_MODE: enabled
echo [CONFIG] TensorFlow logging: suppressed

REM Verify Python and dependencies
echo [CHECK] Python version:
python --version

echo [CHECK] Verifying key packages:
pip list | findstr "fastapi uvicorn tensorflow pymongo motor"

REM Start the backend with optimal settings
echo [START] Starting FastAPI backend...
echo ==================================================

REM Use uvicorn with optimized settings for low memory:
REM - 1 worker (multi-worker would increase memory usage)
REM - Limited request timeout
python -m uvicorn main_fastapi:app ^
    --host 0.0.0.0 ^
    --port %PORT% ^
    --workers 1 ^
    --timeout-keep-alive 5 ^
    --timeout-notify 30

pause
