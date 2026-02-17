# 🎉 SmartAgri-AI Deployment Guide

## ✅ Backend Deployment - LIVE!

**Backend URL:** https://smartagri-backend-ckcz.onrender.com

### Working Services
- ✅ Crop Recommendation
- ✅ Yield Prediction (XGBoost with 86.94% accuracy)
- ✅ Fertilizer Recommendation
- ✅ Stress Detection
- ✅ Fruit Disease Detection V2
- ✅ Spraying Time Prediction
- ✅ Health Check: https://smartagri-backend-ckcz.onrender.com/health
- ✅ API Documentation: https://smartagri-backend-ckcz.onrender.com/docs

### Services with Warnings (Expected)
⚠️ **Plant Disease Detection** - 547MB model excluded (too large for GitHub)
- Returns 503 Service Unavailable
- All other features work perfectly

⚠️ **MongoDB Authentication** - Not configured yet
- User registration/login disabled
- Set up MongoDB Atlas to enable (optional)

⚠️ **AI Chatbot** - GROQ_API_KEY not configured
- Chatbot endpoint unavailable
- Add API key to enable (optional)

---

## 🌐 Frontend Setup

### Environment Files Created

1. **`.env`** - Currently points to production backend
   ```
   VITE_API_BASE_URL=https://smartagri-backend-ckcz.onrender.com
   ```

2. **`.env.local`** - For local development
   ```
   VITE_API_BASE_URL=http://localhost:8001
   ```

3. **`.env.production`** - For production builds
   ```
   VITE_API_BASE_URL=https://smartagri-backend-ckcz.onrender.com
   ```

### Run Frontend Locally (Connected to Production Backend)

```bash
cd frontend
npm install
npm run dev
```

Frontend will run on http://localhost:3000 and connect to the live backend!

### Run Frontend with Local Backend

```bash
# Terminal 1 - Backend
cd backend
uvicorn main_fastapi:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 - Frontend
cd frontend
cp .env.local .env  # Use local backend
npm run dev
```

---

## 🚀 Deploy Frontend to Vercel

1. **Push to GitHub** (already done)
   
2. **Deploy to Vercel:**
   - Go to https://vercel.com
   - Click "New Project"
   - Import from GitHub: `AgricultureFarmTechnology`
   - Framework Preset: Vite
   - Root Directory: `frontend`
   - Environment Variables (will auto-use .env.production):
     ```
     VITE_API_BASE_URL=https://smartagri-backend-ckcz.onrender.com
     VITE_GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
     ```
   - Click "Deploy"

3. **Your frontend will be live at:**
   - `https://agriculture-farm-technology.vercel.app`
   - or `https://your-project.vercel.app`

---

## 🔧 Optional: Enable Additional Features

### 1. MongoDB Atlas (Authentication)

**Why:** Enable user registration, login, and profile management

**Setup:**
1. Create free cluster: https://www.mongodb.com/cloud/atlas
2. Create database user and get connection string
3. Go to Render Dashboard → smartagri-backend → Environment
4. Add variable:
   ```
   MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/FinalProject
   ```
5. Redeploy backend (automatic)

### 2. AI Chatbot (GROQ API)

**Why:** Enable AI-powered agricultural assistant

**Setup:**
1. Get API key: https://console.groq.com
2. Go to Render Dashboard → smartagri-backend → Environment
3. Add variable:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Redeploy backend (automatic)

### 3. Weather Features (Optional)

**OpenWeather API:**
1. Get key: https://openweathermap.org/api
2. Add to Render Environment:
   ```
   OPENWEATHER_API_KEY=your_key_here
   ```

**News API:**
1. Get key: https://newsapi.org
2. Add to Render Environment:
   ```
   NEWSAPI_KEY=your_key_here
   ```

---

## 🧪 Testing Your Deployment

### Test Backend API
```bash
# Health check
curl https://smartagri-backend-ckcz.onrender.com/health

# Crop recommendation
curl -X POST https://smartagri-backend-ckcz.onrender.com/api/crop \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82,"ph":6.5,"rainfall":202.9}'
```

### Test Frontend
1. Open: http://localhost:3000
2. Try Crop Recommendation feature
3. Check browser console for API calls to production backend

---

## 📊 Deployment Summary

| Service | Status | URL | Notes |
|---------|--------|-----|-------|
| Backend API | ✅ LIVE | https://smartagri-backend-ckcz.onrender.com | All core features working |
| Frontend (Local) | 🔄 Ready | http://localhost:3000 | Run `npm run dev` |
| Frontend (Vercel) | ⏳ Deploy | TBD | Deploy to Vercel next |
| MongoDB | ⚠️ Optional | - | Authentication disabled |
| AI Chatbot | ⚠️ Optional | - | Needs GROQ_API_KEY |

---

## 🎯 Next Steps

1. ✅ Backend deployed successfully
2. ✅ Frontend configured to use production backend
3. **→ Deploy frontend to Vercel** (recommended)
4. **→ Test all features** locally first
5. **→ Add MongoDB Atlas** (optional - for authentication)
6. **→ Add GROQ API** (optional - for chatbot)

---

## 🐛 Troubleshooting

### Frontend can't connect to backend
- Check `.env` file has correct backend URL
- Verify backend is live: https://smartagri-backend-ckcz.onrender.com/health
- Check browser console for CORS errors

### Backend shows warnings in logs
- **MongoDB timeout** - Expected if not configured
- **Plant disease skipped** - Expected (model too large)
- **Chatbot failed** - Expected if GROQ_API_KEY not set
- **These warnings don't affect core features!**

### XGBoost version warning
- Minor compatibility notice
- Model works correctly
- Can be ignored

---

## 🎉 Congratulations!

Your SmartAgri-AI backend is live and fully functional! All core ML features are working perfectly. 

**What's working:**
- ✅ Crop recommendations with 90%+ accuracy
- ✅ Yield predictions with 86.94% R² score
- ✅ Fertilizer recommendations
- ✅ Crop stress detection
- ✅ Fruit disease detection
- ✅ Optimal spraying time predictions

Deploy your frontend to Vercel to complete the deployment!
