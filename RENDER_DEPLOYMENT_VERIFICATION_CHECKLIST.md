Render Deployment Verification Checklist
==========================================

## Pre-Deployment (Local Testing)

### Environment Setup
- [ ] Clone latest code from GitHub
- [ ] Create/activate Python virtual environment
- [ ] Run: `pip install -r backend/requirements.txt`
- [ ] Verify Python 3.10+ installed: `python --version`
- [ ] Verify key packages: `pip list | grep -E "fastapi|tensorflow|pymongo"`

### Local Backend Test
```bash
# Set environment variables
export LOW_MEMORY_MODE=true
export ENVIRONMENT=production
export TF_CPP_MIN_LOG_LEVEL=3
export MONGODB_URL=your_mongodb_uri
export PORT=8000

# Run backend
cd backend
python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --workers 1
```

### Quick Endpoint Tests
```bash
# Test 1: Health Check (should respond immediately)
curl http://localhost:8000/health
# Expected: {"status": "healthy", "database": "connected"...}

# Test 2: Model Stats (should show 0 cached models initially)
curl http://localhost:8000/api/models/stats
# Expected: {"cached_models": [], "model_count": 0, ...}

# Test 3: Crop Prediction (loads crop model)
curl "http://localhost:8000/recommend_crop?N=50&P=30&K=40&temperature=25&humidity=60&ph=6.5&rainfall=100&ozone=35"
# Expected: {"recommended_crop": "..."}

# Test 4: Model Stats Again (should show crop_model cached)
curl http://localhost:8000/api/models/stats
# Expected: {"cached_models": ["crop_model"], "model_count": 1}

# Test 5: Yield Prediction (loads yield model)
curl "http://localhost:8000/predict_yield?lat=28&lon=77&ozone=40&soil=0.5"
# Expected: {"result": "Predicted Potato Yield: ... tonnes/hectare"}

# Test 6: Check Memory Usage
ps aux | grep uvicorn
# Note memory usage - should be reasonable (~150-300MB with 2-3 models loaded)
```

### Monitoring Startup
- [ ] Note startup time - should be 3-5 seconds
- [ ] Look for these log messages:
  ```
  [START] Starting SmartAgri API
  [OK] Port ready - services loading in background
  [BACKGROUND] Starting service initialization
  [SKIP] ... (low memory mode)
  [OK] MongoDB Connected
  ```
- [ ] No TensorFlow log spam should appear
- [ ] No errors or warnings during startup

## Pre-Render Deployment Steps

### Code Review
- [ ] Reviewed `model_manager.py` - lazy loading system
- [ ] Reviewed `logging_config.py` - logging suppression
- [ ] Reviewed `main_fastapi.py` changes - uses model_manager
- [ ] Reviewed `requirements.txt` - optimized packages
- [ ] Reviewed `Dockerfile` - optimized settings

### Git Preparation
- [ ] All changes committed to GitHub
- [ ] Branch pushed to remote
- [ ] Verified files on GitHub match local

### Environment Variables List
Create in Render dashboard:
```
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/FinalProject
GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_secret_here
GROQ_API_KEY=gsk_your_key_here (OPTIONAL - skip if using auth only)
LOW_MEMORY_MODE=true
ENVIRONMENT=production
TF_CPP_MIN_LOG_LEVEL=3
```

## Render Deployment

### Create Render Service
- [ ] Go to render.com
- [ ] Create new "Web Service"
- [ ] Connect GitHub account
- [ ] Select SmartAgri-AI repository
- [ ] Select branch: `main`
- [ ] Select build: Docker
- [ ] Dockerfile path: `./backend/Dockerfile`
- [ ] Docker context: `./backend`
- [ ] Plan: Free (512MB RAM)
- [ ] Region: Oregon (or nearest)

### Configure Environment
- [ ] Go to "Environment" tab
- [ ] Add all variables listed above
- [ ] Verify each variable is set correctly
- [ ] Save configuration

### Deploy
- [ ] Click "Deploy"
- [ ] Wait for build to complete (~3-5 minutes)
- [ ] Check logs for:
  ```
  === Building Docker image
  [+] Building X.X s
  Successfully tagged
  === Deploying...
  ```

## Post-Deployment Verification

### Check Render Logs
- [ ] Go to Render dashboard
- [ ] Select your service: `smartagri-backend`
- [ ] Click "Logs" tab
- [ ] Verify you see these messages:
  ```
  [STARTUP] SmartAgri-AI FastAPI Backend Initialization
  [OK] FastAPI app instance created
  [OK] Port ready - services loading in background
  [OK] MongoDB Connected
  [BACKGROUND] Service initialization complete
  ```
- [ ] No errors or restarts shown

### Test Deployment
Get your service URL from Render (format: `https://smartagri-backend-xxxxx.onrender.com`)

```bash
# Test 1: Health Check
curl https://smartagri-backend-xxxxx.onrender.com/health
# Expected: {"status": "healthy", ...}

# Test 2: Model Stats
curl https://smartagri-backend-xxxxx.onrender.com/api/models/stats
# Expected: {"cached_models": [], ...}

# Test 3: Prediction (loads model first time)
curl "https://smartagri-backend-xxxxx.onrender.com/recommend_crop?N=50&P=30&K=40&temperature=25&humidity=60&ph=6.5&rainfall=100&ozone=35"
# Expected: {"recommended_crop": "..."}
```

### Monitor Resource Usage
- [ ] Check Render dashboard "Metrics" tab
- [ ] Memory usage should stay under 512MB
- [ ] No sudden spikes or restarts
- [ ] CPU usage should be low when idle

### Performance Verification
- [ ] First prediction: ~8-12 seconds (model loading)
- [ ] Second prediction: <1 second (cached)
- [ ] Zero downtime or restarts over 24 hours

## Frontend Integration

### Update Frontend API URL
In your frontend code, update API endpoints:
```javascript
// Before:
const API = "http://localhost:8000"

// After:
const API = "https://smartagri-backend-xxxxx.onrender.com"
```

### Test Full Stack
- [ ] Login works: `/auth/login`
- [ ] Crop prediction works: `/recommend_crop`
- [ ] Yield prediction works: `/predict_yield`
- [ ] Image upload works: `/api/fruit-disease/predict` (if applicable)
- [ ] All features functional

## Success Criteria (All Must Pass ✓)

- [ ] Startup time < 5 seconds
- [ ] Memory at startup < 100MB
- [ ] Memory with all models loaded < 512MB
- [ ] No memory limit exceeded errors
- [ ] Zero restarts over 24 hours
- [ ] All endpoints return 200 OK
- [ ] Models load correctly on first call
- [ ] Models cached for fast subsequent calls
- [ ] TensorFlow logs suppressed
- [ ] Predictions return correct results
- [ ] No "out of memory" errors in logs
- [ ] Database connection stable
- [ ] Authentication working
- [ ] Frontend can call all endpoints

## Troubleshooting Guide

### Problem: Backend not starting
**Check in Render logs:**
```
ERROR: [Errno 2] No such file or directory
```
**Solution**: Verify model files exist in Docker build
- Check: `docker build -f backend/Dockerfile -t test backend/`
- Verify: Model files in `/backend/model/` directory

### Problem: Memory exceeds 512MB quickly
**Check**:
```bash
curl https://your-service/api/models/stats
```
**Solutions**:
- Verify `LOW_MEMORY_MODE=true` is set
- Confirm `--workers 1` in Dockerfile
- Check only one model shouldn't use >300MB

### Problem: Slow responses (30+ seconds)
**Check logs for:**
- TensorFlow loading (expected on first call)
- Database connection delays
- Model loading delays

**Solutions**:
- Wait: First model load takes 8-12 seconds
- Cache: Subsequent calls should be <1 second
- Monitor: Check if models cached after first call

### Problem: "Model not available" errors
**Solutions**:
- Check model files in Docker build
- Verify path: `model/model_name.pkl`
- Use `/api/models/stats` to see what's loaded

### Problem: Getting "503 Service Unavailable"
**Causes**:
- Service still starting up (wait 30-60 seconds)
- Out of memory (restart service)
- Model file missing

**Solutions**:
- Wait for startup to complete
- Check Render logs
- Verify all model files included

## Rollback Plan

If deployment fails:

1. **Check Render Dashboard**
   - Go to service page
   - Click "Suspend" to stop
   - Fix the issue
   - Click "Resume" to restart

2. **If Quick Fix Needed**
   ```bash
   git revert <commit-hash>
   git push
   # Render auto-redeploys
   ```

3. **If Need to Debug**
   - Keep the failed deployment active
   - Use "Logs" tab to diagnose
   - Connect with SSH if available

## Performance Optimization Tips

### If Still Slow:
1. Use Render Pro plan for 2x memory (1GB)
2. Add Redis caching layer
3. Implement model preloading in background

### If Memory Issues:
1. Disable unused services (chatbot, remedies)
2. Reduce max request timeout
3. Upgrade to Render Pro plan

## Maintenance

### Weekly Checks
- [ ] Check Render logs for errors
- [ ] Verify memory usage is stable
- [ ] Test key endpoints still working
- [ ] Monitor response times

### Monthly Checks
- [ ] Review model performance
- [ ] Check for package updates
- [ ] Verify security settings
- [ ] Update documentation if needed

## Documentation Files

Essential docs created for this deployment:

1. **MEMORY_OPTIMIZATION_SUMMARY.md**
   - Complete overview of all changes
   - Detailed implementation details
   - Performance comparison

2. **RENDER_MEMORY_OPTIMIZATION.md**
   - Deployment guide
   - Environment setup
   - Troubleshooting tips

3. **RENDER_DEPLOYMENT_VERIFICATION_CHECKLIST.md** (this file)
   - Step-by-step deployment process
   - Verification tests
   - Post-deployment checks

## Success!

When you see:
```
✓ Backend starts in 3-5 seconds
✓ Memory at startup ~50-80MB
✓ All endpoints working
✓ Models loading on demand
✓ Zero out-of-memory errors
✓ 24/7 stable operation
```

**Congratulations! Your backend is successfully optimized for Render! 🎉**

---

Need help?
- Check logs: Render Dashboard → Logs tab
- Test endpoint: `/api/models/stats`
- Review: RENDER_MEMORY_OPTIMIZATION.md
- Verify: All environment variables set

Good luck! 🚀
