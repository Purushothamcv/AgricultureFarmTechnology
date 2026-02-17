@echo off
REM ============================================================================
REM Start MongoDB Locally with Docker - 100% FREE, NO CLOUD, NO PAYMENT
REM ============================================================================

echo.
echo ===============================================================
echo  Starting LOCAL MongoDB (Docker) - Completely FREE!
echo ===============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking if MongoDB container exists...
docker ps -a | findstr "smartagri-mongodb" >nul
if errorlevel 1 (
    echo      Creating new MongoDB container...
    docker run -d ^
        --name smartagri-mongodb ^
        -p 27017:27017 ^
        -e MONGO_INITDB_ROOT_USERNAME=admin ^
        -e MONGO_INITDB_ROOT_PASSWORD=smartagri2024 ^
        -e MONGO_INITDB_DATABASE=FinalProject ^
        -v smartagri-mongodb-data:/data/db ^
        mongo:6.0
    
    if errorlevel 1 (
        echo [ERROR] Failed to create MongoDB container!
        pause
        exit /b 1
    )
    
    echo      MongoDB container created successfully!
) else (
    echo      Container exists. Starting...
    docker start smartagri-mongodb
    
    if errorlevel 1 (
        echo [ERROR] Failed to start MongoDB container!
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Waiting for MongoDB to be ready...
timeout /t 5 /nobreak >nul
echo      MongoDB is ready!

echo.
echo [3/3] Testing connection...
docker exec smartagri-mongodb mongosh --quiet --eval "db.runCommand({ping: 1})" >nul 2>&1
if errorlevel 1 (
    echo      Warning: MongoDB may still be initializing (this is normal)
    echo      Wait 10 more seconds and try starting backend
) else (
    echo      Connection successful!
)

echo.
echo ===============================================================
echo  MongoDB is RUNNING LOCALLY!
echo ===============================================================
echo.
echo  Connection Details:
echo    Host:     localhost
echo    Port:     27017
echo    Username: admin
echo    Password: smartagri2024
echo    Database: FinalProject
echo.
echo  Connection String:
echo    mongodb://admin:smartagri2024@localhost:27017/FinalProject?authSource=admin
echo.
echo ===============================================================
echo.
echo  Next Steps:
echo    1. Keep this window open (MongoDB running)
echo    2. Open NEW terminal
echo    3. Run: cd backend
echo    4. Run: python main_fastapi.py
echo.
echo  To STOP MongoDB:
echo    docker stop smartagri-mongodb
echo.
echo  To VIEW logs:
echo    docker logs smartagri-mongodb -f
echo.
echo ===============================================================
echo.
echo Press any key to view MongoDB logs (Ctrl+C to exit)...
pause >nul

docker logs smartagri-mongodb -f
