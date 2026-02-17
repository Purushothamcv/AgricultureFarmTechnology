# 🚀 Docker Quick Start Guide - SmartAgri-AI

Get your SmartAgri-AI application running with Docker in under 5 minutes!

## Prerequisites

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- Git installed
- At least 2GB RAM available

## Quick Setup (3 Steps)

### Step 1: Clone & Configure

```bash
# Clone the repository
git clone https://github.com/Purushothamcv/AgricultureFarmTechnology.git
cd SmartAgri-AI

# Create backend environment file
cp backend/.env.example backend/.env

# Create frontend environment file  
cp frontend/.env.example frontend/.env
```

### Step 2: Edit Environment Variables

**Edit `backend/.env`:**
```bash
# Required: Add your API keys and secrets
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GROQ_API_KEY=your-groq-api-key
SECRET_KEY=run-this-command-openssl-rand-hex-32
```

**Edit `frontend/.env`:**
```bash
# Keep as-is for local development
VITE_API_BASE_URL=http://localhost:8001
VITE_GOOGLE_CLIENT_ID=your-google-client-id
```

### Step 3: Run with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs (optional)
docker-compose logs -f
```

## Access Your Application

- 🌐 **Frontend:** http://localhost:3000
- ⚙️ **Backend API:** http://localhost:8001
- 📚 **API Docs:** http://localhost:8001/docs
- 🔍 **Health Check:** http://localhost:8001/health

## Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a specific service
docker-compose restart backend

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Rebuild after code changes
docker-compose build
docker-compose up -d

# Clean everything (including database)
docker-compose down -v
```

## Verify Everything Works

1. **Check all containers are running:**
   ```bash
   docker-compose ps
   ```
   You should see 3 services: mongodb, backend, frontend (all "Up")

2. **Test backend health:**
   ```bash
   curl http://localhost:8001/health
   ```
   Should return: `{"status": "healthy"}`

3. **Open frontend in browser:**
   - Navigate to http://localhost:3000
   - You should see the SmartAgri-AI login page

## Troubleshooting

### Port Already in Use

**Error:** `Bind for 0.0.0.0:8001 failed: port is already allocated`

**Solution:**
```bash
# Windows - find and kill process
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# Or change the port in docker-compose.yml
```

### MongoDB Connection Failed

**Error:** `MongoServerError: Authentication failed`

**Solution:**
```bash
# Reset MongoDB
docker-compose down -v
docker-compose up -d
```

### Frontend Shows "Cannot connect to backend"

**Solution:**
1. Check backend is running: `docker-compose ps`
2. Check backend health: `curl http://localhost:8001/health`
3. Verify `frontend/.env` has correct `VITE_API_BASE_URL`

### ML Models Not Loading

**Error:** `Model file not found`

**Solution:**
```bash
# Ensure model files exist
ls backend/model/

# Rebuild backend
docker-compose build backend
docker-compose up -d backend
```

## Next Steps

- 📖 Read the full [Docker Deployment Guide](DOCKER_DEPLOYMENT.md) for production deployment
- 🔧 Configure additional API keys for weather and news features
- 🚀 Deploy to cloud platforms (Render, Railway, AWS, etc.)

## Need Help?

- Check the [Full Deployment Guide](DOCKER_DEPLOYMENT.md)
- Review [Backend Documentation](backend/README.md)
- Review [Frontend Documentation](frontend/README.md)
- Open an issue on [GitHub](https://github.com/Purushothamcv/AgricultureFarmTechnology/issues)

---

**Happy Farming! 🌱🚜**
