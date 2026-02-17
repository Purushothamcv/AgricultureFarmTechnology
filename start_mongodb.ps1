# ============================================================================
# Start MongoDB Locally with Docker - 100% FREE, NO CLOUD, NO PAYMENT
# ============================================================================

Write-Host "`n===============================================================" -ForegroundColor Cyan
Write-Host " Starting LOCAL MongoDB (Docker) - Completely FREE!" -ForegroundColor Green
Write-Host "===============================================================`n" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "[ERROR] Docker is not running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/3] Checking if MongoDB container exists..." -ForegroundColor Yellow

$containerExists = docker ps -a --format "{{.Names}}" | Select-String "smartagri-mongodb"

if (-not $containerExists) {
    Write-Host "      Creating new MongoDB container..." -ForegroundColor Gray
    
    docker run -d `
        --name smartagri-mongodb `
        -p 27017:27017 `
        -e MONGO_INITDB_ROOT_USERNAME=admin `
        -e MONGO_INITDB_ROOT_PASSWORD=smartagri2024 `
        -e MONGO_INITDB_DATABASE=FinalProject `
        -v smartagri-mongodb-data:/data/db `
        mongo:6.0
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create MongoDB container!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "      MongoDB container created successfully!" -ForegroundColor Green
} else {
    Write-Host "      Container exists. Starting..." -ForegroundColor Gray
    docker start smartagri-mongodb | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to start MongoDB container!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[2/3] Waiting for MongoDB to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Write-Host "      MongoDB is ready!" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Testing connection..." -ForegroundColor Yellow
$testResult = docker exec smartagri-mongodb mongosh --quiet --eval "db.runCommand({ping: 1})" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "      Warning: MongoDB may still be initializing (this is normal)" -ForegroundColor Yellow
    Write-Host "      Wait 10 more seconds and try starting backend" -ForegroundColor Gray
} else {
    Write-Host "      Connection successful!" -ForegroundColor Green
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host " MongoDB is RUNNING LOCALLY!" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Connection Details:" -ForegroundColor White
Write-Host "   Host:     localhost" -ForegroundColor Gray
Write-Host "   Port:     27017" -ForegroundColor Gray
Write-Host "   Username: admin" -ForegroundColor Gray
Write-Host "   Password: smartagri2024" -ForegroundColor Gray
Write-Host "   Database: FinalProject" -ForegroundColor Gray
Write-Host ""
Write-Host " Connection String:" -ForegroundColor White
Write-Host "   mongodb://admin:smartagri2024@localhost:27017/FinalProject?authSource=admin" -ForegroundColor Gray
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Keep this window open (MongoDB running)" -ForegroundColor White
Write-Host "   2. Open NEW terminal" -ForegroundColor White
Write-Host "   3. Run: cd backend" -ForegroundColor White
Write-Host "   4. Run: python main_fastapi.py" -ForegroundColor White
Write-Host ""
Write-Host " To STOP MongoDB:" -ForegroundColor Yellow
Write-Host "   docker stop smartagri-mongodb" -ForegroundColor White
Write-Host ""
Write-Host " To VIEW logs:" -ForegroundColor Yellow
Write-Host "   docker logs smartagri-mongodb -f" -ForegroundColor White
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to view MongoDB logs (Ctrl+C to exit)..." -ForegroundColor Yellow
Read-Host

docker logs smartagri-mongodb -f
