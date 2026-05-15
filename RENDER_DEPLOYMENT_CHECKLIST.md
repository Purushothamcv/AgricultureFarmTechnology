# Render Deployment - Quick Checklist

## Changes Made

### Backend Files Modified
- ✅ `backend/main_fastapi.py` - Safe imports, early logging, guarded startup
- ✅ `backend/Dockerfile` - Simplified CMD for reliable binding
- ✅ `backend/__init__.py` - Already exists (required for Python module)

---

## Pre-Deployment Verification (Local)

```bash
# 1. Test that app imports correctly
cd backend
python -c "from main_fastapi import app; print('SUCCESS')"

# 2. Test startup
python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000

# 3. In another terminal, test endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/test-mongodb
```

**All working?** → Continue to deployment

---

## Render Dashboard Setup

### Required Environment Variables
Set these in Render Dashboard → Environment:

1. **MONGODB_URL** (REQUIRED)
   ```
   mongodb+srv://Purushotham:Purushotham123@cluster0.bpdrfrc.mongodb.net/?retryWrites=true&w=majority
   ```

2. **GROQ_API_KEY** (Required for chatbot)
   ```
   [Your Groq API key]
   ```

3. **SECRET_KEY** (JWT authentication)
   ```
   [Let Render auto-generate]
   ```

### Service Configuration
- **Service**: smartagri-backend
- **Type**: Docker
- **Dockerfile**: `./backend/Dockerfile`
- **Docker Context**: `./backend`
- **Port**: Leave as default (Render assigns dynamically)

---

## Deployment Process

### Option 1: Automatic (Recommended)

```bash
# 1. Commit changes
git add backend/main_fastapi.py backend/Dockerfile
git commit -m "fix: Render deployment - safe imports and early logging"

# 2. Push to main branch
git push origin main

# 3. Render auto-deploys from main branch
```

**Render will automatically**:
- Detect push to main
- Build Docker image
- Start container
- Bind to PORT environment variable

### Option 2: Manual Redeploy

1. Go to Render Dashboard
2. Click "smartagri-backend" service
3. Click "Manual Deploy" button
4. Select "main" branch
5. Click "Deploy"

---

## Monitor Deployment

### Watch the Logs

In Render Dashboard → Logs tab:

**Expected sequence**:
```
[START] SmartAgri-AI FastAPI Backend Initialization
[DEBUG] Python Path: ...
[DEBUG] Current Directory: /app
[DEBUG] Main File Location: /app/main_fastapi.py
[DEBUG] PORT env var: (some port, e.g., 10000)

[INFO] SmartAgri-AI Backend starting...

======================================================================
[DB] MongoDB Atlas Connection
[OK] MongoDB Atlas Connected Successfully!
======================================================================

[OK] Crop model loaded successfully
[CREATED] FastAPI app instance created and ready to bind

[INFO] Registering API routes...
[OK] Auth routes registered
[OK] Fruit disease routes registered
...

[START] Starting SmartAgri API initialization...
[OK] MongoDB Connected
[OK] Fruit disease service initialized
[OK] Production fruit disease service initialized
...

[OK] Startup complete - API ready to accept requests

INFO: Started server process [PID]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:PORT
```

**If you see**:
- ✅ "Application startup complete" → SUCCESS
- ✅ "Uvicorn running on" → SUCCESS
- ❌ App crashes or timeouts → Check error in logs

---

## Test After Deployment

Once Render shows "Live":

```bash
# Replace URL with your actual Render URL
export BACKEND_URL="https://smartagri-backend-ckcz.onrender.com"

# Test root
curl $BACKEND_URL/

# Test health
curl $BACKEND_URL/health

# Test MongoDB
curl $BACKEND_URL/test-mongodb
```

**Expected responses**:
```json
{
  "status": "ok",
  "message": "SmartAgri API is running",
  "database": "connected"
}
```

---

## Common Issues & Fixes

### 1. "No open ports detected"

**Cause**: App crashed before binding to port

**Fix**:
- Check Render logs for error
- Common causes:
  - Missing MONGODB_URL env var
  - Import error in main_fastapi.py
  - Model file missing in Docker image

**Check**:
```bash
# Local test
python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000
```

### 2. "ModuleNotFoundError"

**Cause**: Missing dependency

**Fix**:
1. Install locally: `pip install [module]`
2. Update requirements.txt: `pip freeze > requirements.txt`
3. Commit and redeploy

### 3. App slow to start (>60 seconds)

**Cause**: Model loading takes time

**Normal**:
- TensorFlow loading: ~15-30s
- sklearn models: ~5-10s
- MongoDB init: ~5-10s
- Total: ~30-60s

**If > 90s**:
- Check Render logs for hanging process
- Render may auto-restart if health check times out
- Consider upgrading Render plan (more CPU = faster)

### 4. "Connection refused" or "MongoDB timeout"

**Cause**: 
- MONGODB_URL not set in Render
- Connection string is wrong
- MongoDB Atlas cluster not running
- Network block (MongoDB Atlas IP whitelist)

**Fix**:
1. Verify MONGODB_URL in Render Dashoard
2. Test locally: `curl http://localhost:8000/test-mongodb`
3. Whitelist Render IP in MongoDB Atlas (or use 0.0.0.0/0 for testing)

---

## Rollback if Issues

If deployment has problems:

```bash
# Revert commit
git revert HEAD

# Force new deployment
git push origin main

# OR manually redeploy previous version in Render Dashboard
```

---

## Performance Tips

1. **Faster startup**: Upgrade Render plan (more CPU)
2. **Reduce model size**: Consider quantized/compressed models
3. **Lazy load**: Load models on first request instead of startup
4. **Cache layers**: Use Redis cache (separate Render service)

---

## Next Steps

1. **Verify locally** → `python -m uvicorn backend.main_fastapi:app --port 8000`
2. **Commit & Push** → `git push origin main`
3. **Monitor Render** → Check logs in dashboard
4. **Test endpoints** → `curl https://smartagri-backend-ckcz.onrender.com/`
5. **Connect frontend** → React app should now work

---

## Success Indicators

✅ Render Dashboard shows "Live" status  
✅ Logs show "Application startup complete"  
✅ Logs show "Uvicorn running on" message  
✅ Root endpoint returns status: "ok"  
✅ Health endpoint returns status: "healthy"  
✅ MongoDB test shows collections are accessible  
✅ Frontend can connect and make API calls  

**You're deployed!** 🚀
