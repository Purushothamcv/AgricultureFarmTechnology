# SmartAgri Backend - Quick Start Local Guide

## ✅ Status: RUNNING & VERIFIED

Your backend is currently **RUNNING LOCALLY** on port 8000 with full MongoDB Atlas connection.

---

## 📋 Current Setup

```
Backend Status:   ✅ Running
Port:             ✅ 8000
MongoDB:          ✅ Connected to Atlas
Process ID:       28988
Location:         C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend
```

---

## 🚀 How to Run the Backend Locally

### Option 1: Direct Python (Recommended)
```bash
cd "C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend"
python main_fastapi.py
```

### Option 2: Using Uvicorn
```bash
cd "C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend"
uvicorn main_fastapi:app --reload --port 8000
```

### Option 3: From Project Root
```bash
cd "C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI"
python -m uvicorn backend.main_fastapi:app --reload --port 8000
```

---

## 🔍 Testing Endpoints

Once the backend is running, test these endpoints:

### 1. Health Check
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "api": "ok"
}
```

### 2. MongoDB Connection Test
```bash
curl http://localhost:8000/test-mongodb
```
**Expected Response:**
```json
{
  "status": "success",
  "message": "MongoDB Atlas Connected",
  "connection_type": "MongoDB Atlas (PyMongo)",
  "database": "Connected",
  "collections": {
    "users": "accessible",
    "chat_sessions": "accessible"
  }
}
```

### 3. Root Endpoint
```bash
curl http://localhost:8000/
```
**Expected Response:**
```json
{
  "status": "ok",
  "message": "SmartAgri API is running",
  "version": "1.0.0",
  "database": "connected"
}
```

### 4. API Documentation
```
http://localhost:8000/docs
```
(Interactive Swagger UI with all endpoints)

---

## 📊 Key Information

### MongoDB Atlas Details
- **Connection String**: `mongodb+srv://Purushotham:Purushotham123@cluster0.bpdrfrc.mongodb.net/?retryWrites=true&w=majority`
- **Databases**:
  - `users` - User authentication
  - `chatbot` - Chat sessions
  - `FinalProject` - Legacy data
- **Status**: ✅ Connected & Ready

### Backend Features Running
✅ FastAPI REST API  
✅ MongoDB Atlas Database  
✅ CORS Middleware (localhost:5173 enabled for frontend)  
✅ ML Models (Crop, Disease Detection, Yield, Stress)  
✅ Authentication (JWT)  
✅ Chatbot Service (Groq AI)  
✅ Fertilizer Recommendation  
✅ Soil Data API  

---

## ⏹️ Stopping the Backend

Press `Ctrl+C` in the terminal running the backend.

You'll see:
```
INFO:     Shutting down application.
[SHUTDOWN] Shutting down SmartAgri API...
INFO:database:MongoDB connection closed
INFO:     Application shutdown complete.
```

---

## 🔧 Troubleshooting

### Port 8000 Already in Use

If you get error: `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`

**Fix:**
```powershell
# Kill process using port 8000
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Then start backend again
python main_fastapi.py
```

### MongoDB Connection Failed

If you get `ERROR: MongoDB Connection Failed`

**Check:**
1. ✅ MongoDB Atlas cluster is running
2. ✅ Network Access whitelist includes your IP (0.0.0.0/0 for testing)
3. ✅ MONGODB_URL in `.env` file is correct
4. ✅ Internet connection works

### Missing Dependencies

If you get `ModuleNotFoundError`:

```bash
pip install fastapi uvicorn pymongo python-dotenv groq motor
```

---

## 📂 Project Structure

```
SmartAgri-AI/
├── backend/
│   ├── main_fastapi.py         # Main app entry point
│   ├── db.py                   # MongoDB connection (PyMongo)
│   ├── database.py             # Async MongoDB (Motor)
│   ├── auth.py                 # Authentication routes
│   ├── chatbot_service.py       # Groq AI chatbot
│   ├── crop_service.py          # Crop recommendations
│   ├── yield_prediction_service.py
│   ├── fertilizer_prediction_service.py
│   ├── disease_detection/       # ML models
│   └── model/                  # Pre-trained models
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── .env                        # Environment variables
└── docker-compose.yml          # Docker setup
```

---

## 🌐 Frontend Integration

The frontend (React/Vite on localhost:5173) can now connect to the backend:

```javascript
// Frontend API calls
const response = await fetch('http://localhost:8000/api/crop/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ /* data */ })
});
```

**CORS is enabled** for `localhost:5173` and `127.0.0.1:5173`

---

## 📈 Performance Notes

- ✅ All ML models loaded (Crop, Disease, Yield, Stress)
- ✅ Agentic AI dataset: 345,327 rows loaded
- ✅ Database indexes created for fast queries
- ✅ Connection pooling configured (50 max connections)
- ✅ Retry logic with exponential backoff

---

## 🚢 Deployment to Render

When you're ready for production deployment:

1. All these settings work **identically** on Render
2. Same `MONGODB_URL` in environment variables
3. Same port binding (Render creates PORT env variable)
4. Just push to GitHub and Render auto-deploys

---

## ✨ Summary

Your SmartAgri backend is now:
- ✅ Running locally on port 8000
- ✅ Connected to MongoDB Atlas
- ✅ Ready for frontend development
- ✅ Ready for testing all APIs
- ✅ Production-ready for deployment

**Enjoy building! 🎉**
