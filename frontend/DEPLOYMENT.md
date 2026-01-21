# 🚀 Deployment Guide

## Prerequisites
- GitHub account
- Vercel account (sign up at vercel.com)
- Backend deployed (e.g., Render, Railway, etc.)

---

## 📦 Step 1: Prepare Environment Variables

Before deploying, you need your **backend URL**.

### For Vercel Deployment:
1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add this variable:
   ```
   VITE_API_BASE_URL=https://your-backend-url.onrender.com
   ```
   ⚠️ **Replace** `your-backend-url.onrender.com` with your actual backend URL

---

## 🔗 Step 2: Push to GitHub

### Option A: New Repository
```bash
cd frontend
git init
git add .
git commit -m "Initial commit: SmartAgri-AI Frontend"
git branch -M main
git remote add origin https://github.com/Purushothamcv/AgricultureFarmTechnology.git
git push -u origin main
```

### Option B: Existing Repository (Update)
```bash
cd frontend
git add .
git commit -m "Update: Clean frontend for Vercel deployment"
git push origin main
```

---

## 🌐 Step 3: Deploy on Vercel

### Method 1: Via Vercel Dashboard (Recommended)
1. Go to [vercel.com](https://vercel.com)
2. Click **"Add New"** → **"Project"**
3. Import your GitHub repository: `Purushothamcv/AgricultureFarmTechnology`
4. Configure project:
   - **Framework Preset**: Vite
   - **Root Directory**: `./` (leave as is)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Add Environment Variables:
   - Key: `VITE_API_BASE_URL`
   - Value: `https://your-backend-url.onrender.com`
6. Click **"Deploy"**

### Method 2: Via Vercel CLI
```bash
npm install -g vercel
cd frontend
vercel login
vercel --prod
```

---

## ✅ Step 4: Verify Deployment

1. **Check Build Logs**: Ensure no errors during build
2. **Test Live Site**: 
   - Visit your Vercel URL (e.g., `smartagri-ai.vercel.app`)
   - Try logging in
   - Check if API calls work
3. **Check Network Tab** (F12 in browser):
   - Verify API calls go to your backend URL
   - Check for CORS errors

---

## 🔧 Troubleshooting

### Issue 1: API Not Connecting
**Solution**: Check CORS settings on your backend
```python
# In main_fastapi.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 2: Environment Variables Not Working
**Solution**: 
- Redeploy after adding variables
- Variables must start with `VITE_` prefix
- Check spelling: `VITE_API_BASE_URL`

### Issue 3: 404 on Page Refresh
**Solution**: Already handled by `vercel.json` rewrites

### Issue 4: Build Fails
**Solution**: 
```bash
# Test build locally first
npm run build
npm run preview
```

---

## 📱 Post-Deployment Checklist

- [ ] Backend URL is correct in Vercel environment variables
- [ ] Login/Register works
- [ ] Dashboard loads weather data
- [ ] All modules are accessible
- [ ] Maps render correctly
- [ ] No console errors
- [ ] Mobile responsive
- [ ] CORS configured on backend

---

## 🔄 Updating Your Deployment

```bash
# Make changes to your code
git add .
git commit -m "Update: Description of changes"
git push origin main

# Vercel auto-deploys on every push to main branch
```

---

## 🌍 Custom Domain (Optional)

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed by Vercel

---

## 📊 Monitoring

- **Vercel Analytics**: Automatic (free plan)
- **Error Tracking**: Check Vercel logs for runtime errors
- **Performance**: Use Lighthouse in Chrome DevTools

---

## 🆘 Need Help?

- Vercel Docs: https://vercel.com/docs
- Vite Docs: https://vitejs.dev/guide/
- GitHub Issues: Open an issue in your repo

---

**Your app will be live at**: `https://agriculture-farm-technology.vercel.app` (or your custom domain)
