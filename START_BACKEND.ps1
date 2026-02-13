# SmartAgri Backend Server Launcher
# This script starts the backend server in a visible window

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   SmartAgri Backend Server Launcher" -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Cyan

$projectRoot = "C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI"
$backendPath = Join-Path $projectRoot "backend"

# Change to backend directory
Set-Location $backendPath

Write-Host "[1/3] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n[2/3] Checking if port 8000 is available..." -ForegroundColor Yellow
$portInUse = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "      Port 8000 is in use. Killing processes..." -ForegroundColor Yellow
    $portInUse | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
    Write-Host "      Port 8000 is now free!" -ForegroundColor Green
} else {
    Write-Host "      Port 8000 is available!" -ForegroundColor Green
}

Write-Host "`n[3/3] Starting backend server on http://localhost:8000..." -ForegroundColor Yellow
Write-Host "      Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "========================================`n" -ForegroundColor Cyan

# Start the server (this will block and show all output)
python -m uvicorn ultra_minimal_auth:app --host 0.0.0.0 --port 8000 --reload

Write-Host "`nServer stopped." -ForegroundColor Yellow
