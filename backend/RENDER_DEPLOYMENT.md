# 🚀 Render Backend Deployment Guide

## Prerequisites
- GitHub account with backend code pushed
- Render account (sign up at [render.com](https://render.com))
- MongoDB Atlas account (for cloud database)

---

## 📋 Step 1: Set Up MongoDB Atlas (Cloud Database)

### 1.1 Create MongoDB Atlas Cluster
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up / Log in
3. Create a **FREE** M0 Cluster:
   - Provider: AWS
   - Region: Choose closest to your Render region (e.g., Oregon)
   - Cluster Name: `SmartAgri`

### 1.2 Configure Database Access
1. **Database Access** → **Add New Database User**
   - Username: `smartagri_user`
   - Password: Generate secure password (save it!)
   - Database User Privileges: `Read and write to any database`

### 1.3 Configure Network Access
1. **Network Access** → **Add IP Address**
   - Click **"Allow Access from Anywhere"** (0.0.0.0/0)
   - This allows Render to connect

### 1.4 Get Connection String
1. **Database** → **Connect** → **Connect your application**
2. Copy connection string:
   ```
   mongodb+srv://smartagri_user:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
3. Replace `<password>` with your actual password
4. Add database name:
   ```
   mongodb+srv://smartagri_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/FinalProject?retryWrites=true&w=majority
   ```

---

## 🌐 Step 2: Deploy Backend on Render

### 2.1 Create New Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository:
   - Repository: `Purushothamcv/AgricultureFarmTechnology`
   - Click **"Connect"**

### 2.2 Configure Service Settings

**Basic Settings:**
- **Name**: `smartagri-backend`
- **Region**: Oregon (US West) - *Free tier available*
- **Branch**: `main`
- **Root Directory**: `backend`
- **Runtime**: `Python 3`

**Build Settings:**
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```

**Start Command:**
- **Start Command**: 
  ```bash
  uvicorn main_fastapi:app --host 0.0.0.0 --port $PORT
  ```

**Instance Type:**
- Select **"Free"** (512 MB RAM, sleeps after 15 min inactivity)

### 2.3 Add Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

| Key | Value | Notes |
|-----|-------|-------|
| `PYTHON_VERSION` | `3.10.0` | Python version |
| `MONGODB_URL` | `mongodb+srv://...` | Your MongoDB Atlas connection string |
| `DATABASE_NAME` | `FinalProject` | Database name |

**Important**: Paste your **full MongoDB connection string** from Step 1.4

### 2.4 Deploy!
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for first deployment
3. Check deployment logs for errors

---

## ✅ Step 3: Verify Deployment

### 3.1 Test Your Backend
Your backend will be live at:
```
https://smartagri-backend.onrender.com
```

**Test endpoints:**
- API Docs: `https://smartagri-backend.onrender.com/docs`
- Health Check: `https://smartagri-backend.onrender.com/`
- Weather API: `https://smartagri-backend.onrender.com/api/weather?lat=28.6139&lon=77.2090`

### 3.2 Check Logs
- Go to Render Dashboard → Your Service → **"Logs"**
- Look for:
  ```
  ✅ Successfully connected to MongoDB database: FinalProject
  INFO: Application startup complete.
  ```

---

## 🔗 Step 4: Connect Frontend to Backend

### 4.1 Update Vercel Environment Variables
1. Go to Vercel Dashboard → Your Project
2. **Settings** → **Environment Variables**
3. Update `VITE_API_BASE_URL`:
   ```
   https://smartagri-backend.onrender.com
   ```
4. **Redeploy** your frontend (Vercel will auto-deploy)

### 4.2 Update Backend CORS (Already Done!)
Your `main_fastapi.py` already includes:
```python
allow_origins=[
    "https://agriculture-farm-technology.vercel.app",
    "https://*.vercel.app"
]
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError"
**Solution**: Check `requirements.txt` has all dependencies
```bash
# Locally test requirements
pip install -r backend/requirements.txt
```

### Issue 2: "Can't connect to MongoDB"
**Solution**: 
- Verify MongoDB Atlas IP whitelist (0.0.0.0/0)
- Check connection string format
- Ensure password has no special characters that need encoding

### Issue 3: "Application failed to respond"
**Solution**:
- Check Render logs for errors
- Ensure start command uses `$PORT` variable
- Verify Python version compatibility

### Issue 4: "CORS Error" from Frontend
**Solution**:
- Add your Vercel URL to CORS origins in `main_fastapi.py`
- Redeploy backend after CORS changes

### Issue 5: "Service Sleeping"
**Solution**: 
- Free tier sleeps after 15 min inactivity
- First request after sleep takes ~30 seconds
- Upgrade to paid plan for always-on service

---

## 📊 MongoDB Atlas Tips

### View Data
1. MongoDB Atlas → **Collections**
2. Browse your data:
   - `users` collection
   - `plant_disease_predictions`
   - `weather_logs`

### Monitor Usage
- Free tier: 512 MB storage
- Monitor in Atlas Dashboard → **Metrics**

---

## 🔄 Updating Your Backend

### Method 1: Git Push (Auto-Deploy)
```bash
cd backend
# Make changes
git add .
git commit -m "Update: description"
git push origin main
# Render automatically redeploys!
```

### Method 2: Manual Redeploy
1. Render Dashboard → Your Service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 💰 Cost Optimization

### Free Tier Limits:
- ✅ 750 hours/month (sufficient for one service)
- ✅ Sleeps after 15 min inactivity
- ✅ No credit card required

### Upgrade Options:
- **Starter Plan**: $7/month
  - Always on (no sleep)
  - 512 MB RAM
- **Standard Plan**: $25/month
  - 2 GB RAM
  - Better performance

---

## 🎯 Post-Deployment Checklist

- [ ] Backend accessible at Render URL
- [ ] API docs working (`/docs`)
- [ ] MongoDB connection successful
- [ ] Frontend env variable updated
- [ ] Login/Register working
- [ ] Weather API responding
- [ ] Crop recommendation working
- [ ] All modules tested
- [ ] No CORS errors in browser console

---

## 📱 Mobile Testing

Test your deployed app on:
- [ ] Desktop browser
- [ ] Mobile browser
- [ ] Different networks (WiFi, 4G)

---

## 🔐 Security Best Practices

1. **Never commit `.env` files**
2. **Use strong MongoDB passwords**
3. **Rotate credentials periodically**
4. **Monitor Render logs for suspicious activity**
5. **Keep dependencies updated**

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **MongoDB Docs**: https://docs.mongodb.com

---

## 🎉 Success!

Your backend is now live at:
```
🌐 Backend: https://smartagri-backend.onrender.com
📚 API Docs: https://smartagri-backend.onrender.com/docs
```

Your full-stack app is deployed! 🚀
