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
echo [STEP 2] Checking if backend is already running...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://localhost:8000/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { Write-Output 'RUNNING'; exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Backend is already running on http://localhost:8000
    echo Open the existing backend terminal or stop it before starting a new instance.
    pause
    exit /b 0
)

echo.
echo [STEP 3] Freeing port 8000 if occupied...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

timeout /t 1 >nul

echo.
echo [STEP 4] Starting server on http://localhost:8000...
echo.
echo Backend is running. Press Ctrl+C to stop.
echo ========================================
echo.

python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000

pause
