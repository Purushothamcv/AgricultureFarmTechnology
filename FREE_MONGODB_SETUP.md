# 🆓 FREE MongoDB Setup - No Payment Required!

## ❌ Don't Use MongoDB Atlas (Cloud)

You mentioned MongoDB Atlas is asking for payment. **You don't need it!** Use local MongoDB instead - **100% free, no credit card, no signup**.

---

## ✅ Solution: Local MongoDB with Docker (FREE)

### What You Need:
- Docker Desktop (FREE) - Already installed on your system

### What You Get:
- MongoDB running on YOUR computer (localhost)
- No cloud, no internet connection required
- No payment, no credit card
- All data stored locally

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start MongoDB

**Option A - Windows (Double-click)**:
```
start_mongodb.bat
```

**Option B - PowerShell**:
```powershell
.\start_mongodb.ps1
```

This will:
- ✅ Create MongoDB container (if not exists)
- ✅ Start MongoDB on port 27017
- ✅ Set username: `admin`, password: `smartagri2024`
- ✅ Create database: `FinalProject`

### Step 2: Update .env for LOCAL mode

Your `backend/.env` is already configured! But if running backend directly (not in Docker), change this line:

```env
# Use this for direct Python execution (python main_fastapi.py)
MONGODB_URL=mongodb://admin:smartagri2024@localhost:27017/FinalProject?authSource=admin
```

**Current .env already has**:
```env
MONGODB_URL=mongodb://admin:smartagri2024@mongodb:27017/FinalProject?authSource=admin
```

Change `mongodb:27017` → `localhost:27017` if running backend without Docker.

### Step 3: Start Backend

**New terminal window**:
```bash
cd backend
python main_fastapi.py
```

**Expected Output**:
```
✅ MongoDB connected successfully
✅ GROQ client initialized
🚀 Backend running on http://localhost:8001
```

---

## 🔍 Verify It's Working

### Test MongoDB Connection:
```bash
docker exec -it smartagri-mongodb mongosh -u admin -p smartagri2024
```

Inside MongoDB shell:
```javascript
use FinalProject
db.stats()  // Should show database stats
exit
```

### Test Backend Health:
```bash
curl http://localhost:8001/health
```

Should return:
```json
{
  "status": "healthy",
  "mongodb": "connected"
}
```

---

## 📊 MongoDB Management Commands

### Check if MongoDB is running:
```bash
docker ps | findstr mongodb
```

### View MongoDB logs:
```bash
docker logs smartagri-mongodb -f
```

### Stop MongoDB:
```bash
docker stop smartagri-mongodb
```

### Restart MongoDB:
```bash
docker start smartagri-mongodb
```

### Remove MongoDB (deletes data):
```bash
docker stop smartagri-mongodb
docker rm smartagri-mongodb
docker volume rm smartagri-mongodb-data
```

---

## ⚙️ Configuration Details

### Connection Information:
- **Host**: `localhost` (when backend runs directly) or `mongodb` (when backend runs in Docker)
- **Port**: `27017`
- **Username**: `admin`
- **Password**: `smartagri2024`
- **Database**: `FinalProject`
- **Auth Source**: `admin`

### Full Connection String:
```
mongodb://admin:smartagri2024@localhost:27017/FinalProject?authSource=admin
```

---

## 🐛 Troubleshooting

### Error: "Docker is not running"
**Solution**: Start Docker Desktop
- Windows: Search → Docker Desktop → Open
- Check system tray for Docker icon

### Error: "Port 27017 already in use"
**Solution**: Another MongoDB is running
```bash
# Find process using port 27017
netstat -ano | findstr :27017

# Kill the process (replace <PID> with actual process ID)
Stop-Process -Id <PID> -Force

# Or stop existing MongoDB container
docker stop smartagri-mongodb
```

### Error: "Authentication failed"
**Solution**: Recreate container with correct credentials
```bash
docker stop smartagri-mongodb
docker rm smartagri-mongodb
.\start_mongodb.ps1
```

### Error: "Connection timeout"
**Solution**: Wait for initialization (first start takes ~30 seconds)
```bash
# Check if MongoDB is still starting
docker logs smartagri-mongodb

# Wait for: "Waiting for connections on port 27017"
```

---

## 💡 Why Local MongoDB is Better for Development

| Feature | MongoDB Atlas (Cloud) | Local Docker MongoDB |
|---------|-------------------|---------------------|
| **Cost** | Free tier limited, paid for more | 100% FREE unlimited |
| **Internet** | Required | NOT required |
| **Speed** | Network latency | Instant (localhost) |
| **Data Privacy** | Stored on cloud | Stored on YOUR PC |
| **Setup** | Account, credit card verification | Just Docker |
| **Development** | Slow (network calls) | Fast (local) |

---

## 🌐 For Production (Render Deployment)

When deploying to Render, you can use:

**Option 1: MongoDB Atlas FREE Tier (M0)**
- Go to: https://www.mongodb.com/cloud/atlas/register
- Select **M0 (FREE)** tier - NO CREDIT CARD REQUIRED
- Choose **AWS** provider
- Select **Shared** cluster type
- DO NOT select M2/M5 (those are paid)

**Option 2: Render.com PostgreSQL (Alternative)**
- Use Render's free PostgreSQL instead
- Update code to use PostgreSQL instead of MongoDB

For now, **use local MongoDB** for development!

---

## 📝 Summary

✅ **What's configured**:
- ✅ GROQ_API_KEY (AI chatbot works)
- ✅ MongoDB local setup ready
- ✅ Google OAuth credentials
- ✅ All environment variables set

❌ **What you DON'T need**:
- ❌ MongoDB Atlas cloud account
- ❌ Credit card
- ❌ Any payment

🎯 **Next action**:
1. Run: `start_mongodb.ps1` (starts free local MongoDB)
2. Run: `cd backend && python main_fastapi.py` (starts backend)
3. Run: `cd frontend && npm run dev` (starts frontend)
4. Test authentication and chatbot features!

---

**No payment required. Everything is FREE!** 🎉
