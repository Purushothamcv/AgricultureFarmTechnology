@echo off
chcp 65001 >nul
echo ========================================
echo   SmartAgri Backend Server
echo ========================================
echo.

cd /d "%~dp0backend"

echo [STEP 1] Checking Python...
python --version
if ERRORLEVEL 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [STEP 2] Starting server on http://localhost:8000...
echo.
echo Backend is running. Press Ctrl+C to stop.
echo ========================================
echo.

python -m uvicorn ultra_minimal_auth:app --host 0.0.0.0 --port 8000 --reload

pause
