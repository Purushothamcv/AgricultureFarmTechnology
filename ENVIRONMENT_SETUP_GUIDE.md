# 🔧 Environment Variables Setup Guide

This guide explains how to configure **GROQ API Key** and **MongoDB** to resolve the persistent warning messages.

## 📋 Table of Contents
- [Quick Overview](#quick-overview)
- [Issue 1: GROQ API Key Not Set](#issue-1-groq-api-key-not-set)
- [Issue 2: MongoDB Not Connected](#issue-2-mongodb-not-connected)
- [Local Development Setup](#local-development-setup)
- [Production (Render) Setup](#production-render-setup)
- [Verification Steps](#verification-steps)

---

## Quick Overview

### What's Happening?
Your backend shows these warnings:
```
⚠️ GROQ_API_KEY not found in environment variables
⚠️ MongoDB connection failed: Connection timeout
```

### Why It Happens?
Both services require API keys/connection strings that aren't committed to GitHub (for security). They must be configured manually in:
1. **Local Development**: `.env` file
2. **Production (Render)**: Render Dashboard Environment Variables

### Impact When Not Configured:
- ❌ **GROQ_API_KEY missing**: AI Chatbot won't work (returns 500 error)
- ❌ **MongoDB missing**: User authentication (login/signup) won't work

---

## Issue 1: GROQ API Key Not Set

### What is GROQ?
Groq provides the AI language model (Llama 3.3) that powers the SmartAgri chatbot for answering agricultural questions.

### 🔑 Step 1: Get Your Free GROQ API Key

1. **Visit**: https://console.groq.com/keys
2. **Sign Up**: Use Google or GitHub account (FREE - no credit card required)
3. **Create API Key**:
   - Click **"Create API Key"**
   - Name it: `SmartAgri-AI`
   - Copy the key (starts with `gsk_...`)
   - ⚠️ Save it somewhere safe - you can't view it again!

**Example Key Format**: `gsk_abc123xyz456...` (long alphanumeric string)

### 📝 Step 2: Configure Locally (Development)

Create `.env` file in `backend/` folder:

```bash
# Navigate to backend folder
cd backend

# Create .env file (copy from example)
cp .env.example .env

# Edit .env file
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

Add your key:
```env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### ☁️ Step 3: Configure on Render (Production)

1. **Go to Render Dashboard**: https://dashboard.render.com/
2. **Select Service**: `smartagri-backend`
3. **Click**: **Environment** (left sidebar)
4. **Add Environment Variable**:
   - Key: `GROQ_API_KEY`
   - Value: `gsk_your_actual_api_key_here`
5. **Save Changes** - Render will auto-redeploy (~2 min)

---

## Issue 2: MongoDB Not Connected

### What is MongoDB Used For?
MongoDB stores:
- User accounts (email, password, profile)
- Authentication sessions
- User preferences and history

### 🗄️ Step 1: Create Free MongoDB Atlas Cluster

#### A. Sign Up for MongoDB Atlas (100% FREE)

1. **Visit**: https://www.mongodb.com/cloud/atlas/register
2. **Create Account**: Use email, Google, or GitHub
3. **Select**: **Free Tier (M0)** - No credit card required!
4. **Choose Provider**: AWS (default) or Google Cloud
5. **Region**: Select closest to you (e.g., `us-east-1`)
6. **Cluster Name**: `SmartAgriCluster` (or any name)
7. **Create Cluster** - Takes ~3-5 minutes

#### B. Configure Database Access

1. **Go to**: **Database Access** (left menu)
2. **Add Database User**:
   - Username: `smartagri_admin`
   - Password: Click **Auto-Generate Secure Password** (save this!)
   - Database User Privileges: **Read and write to any database**
3. **Add User**

#### C. Configure Network Access

1. **Go to**: **Network Access** (left menu)
2. **Add IP Address**:
   - Click **"Allow Access from Anywhere"** (for development)
   - IP: `0.0.0.0/0` (automatically filled)
   - Description: `Allow all (development)`
3. **Confirm**

⚠️ **Production Note**: For stricter security, add only Render's IP addresses later.

#### D. Get Connection String

1. **Go to**: **Database** (left menu)
2. **Click**: **Connect** on your cluster
3. **Choose**: **Connect your application**
4. **Select**: **Driver: Python**, **Version: 3.12 or later**
5. **Copy Connection String**:
   ```
   mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
   ```

6. **Replace Placeholders**:
   ```
   mongodb+srv://smartagri_admin:YOUR_PASSWORD_HERE@smartagricluster.abc123.mongodb.net/?retryWrites=true&w=majority
   ```

**Example Final String**:
```
mongodb+srv://smartagri_admin:Xy7z$9mK2p@smartagricluster.mongodb.net/?retryWrites=true&w=majority
```

### 📝 Step 2: Configure Locally (Development)

Edit `backend/.env`:

```env
# MongoDB Atlas Connection
MONGODB_URL=mongodb+srv://smartagri_admin:YOUR_PASSWORD@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=FinalProject
```

⚠️ **Important**: Replace with your actual connection string!

### ☁️ Step 3: Configure on Render (Production)

1. **Go to Render Dashboard**: https://dashboard.render.com/
2. **Select Service**: `smartagri-backend`
3. **Click**: **Environment** (left sidebar)
4. **Add Environment Variable**:
   - Key: `MONGODB_URL`
   - Value: `mongodb+srv://smartagri_admin:YOUR_PASSWORD@cluster.mongodb.net/?retryWrites=true&w=majority`
5. **Save Changes** - Render will auto-redeploy (~2 min)

---

## Local Development Setup

### Complete `.env` File Example

Create `backend/.env` with all required variables:

```env
# ============================================================================
# REQUIRED: MongoDB Connection
# ============================================================================
MONGODB_URL=mongodb+srv://smartagri_admin:YOUR_PASSWORD@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=FinalProject

# ============================================================================
# REQUIRED: AI Chatbot
# ============================================================================
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# ============================================================================
# REQUIRED: Google OAuth
# ============================================================================
GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# ============================================================================
# OPTIONAL: Additional APIs
# ============================================================================
OPENWEATHER_API_KEY=your_openweather_key_here
NEWSAPI_KEY=your_news_api_key_here

# ============================================================================
# AUTO-CONFIGURED (Leave as is)
# ============================================================================
PORT=8000
JWT_SECRET_KEY=your-secret-key-change-this-in-production-2024
```

### Start Local Backend

```bash
cd backend
python main_fastapi.py
```

**Expected Output** (No errors):
```
✅ MongoDB connected successfully
✅ GROQ API initialized successfully
✅ Backend running on http://localhost:8000
```

---

## Production (Render) Setup

### Required Environment Variables on Render Dashboard

| Variable Name | Where to Get | Example Value |
|---------------|--------------|---------------|
| `MONGODB_URL` | MongoDB Atlas → Connect → Connection String | `mongodb+srv://user:pass@cluster.mongodb.net/...` |
| `GROQ_API_KEY` | Groq Console → API Keys | `gsk_abc123xyz456...` |
| `GOOGLE_CLIENT_SECRET` | Google Cloud Console → Credentials | `GOCSPX-abc123...` |

### How to Add on Render:

1. **Login**: https://dashboard.render.com/
2. **Select**: `smartagri-backend` service
3. **Navigate**: **Environment** tab (left sidebar)
4. **Add Each Variable**:
   - Click **"Add Environment Variable"** or **"Add Secret File"**
   - Enter **Key** and **Value**
   - Click **"Save Changes"**
5. **Deploy**: Render auto-deploys after saving (~2-3 minutes)

### Verify Deployment:

```bash
# Check health endpoint
curl https://smartagri-backend-ckcz.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "mongodb": "connected",
  "groq": "initialized"
}
```

---

## Verification Steps

### 1. Test MongoDB Connection

**Local:**
```bash
cd backend
python -c "
import asyncio
from database import connect_to_mongodb
asyncio.run(connect_to_mongodb())
print('✅ MongoDB connected successfully!')
"
```

**Production:**
```bash
curl https://smartagri-backend-ckcz.onrender.com/api/users/me
# Should return 401 (auth required) instead of 500 (MongoDB error)
```

### 2. Test GROQ API

**Local:**
```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the best fertilizer for rice?","language":"en"}'
```

**Production:**
```bash
curl -X POST https://smartagri-backend-ckcz.onrender.com/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the best fertilizer for rice?","language":"en"}'
```

**Expected Response:**
```json
{
  "response": "For rice cultivation, the best fertilizers typically include...",
  "language": "en"
}
```

### 3. Check Backend Logs

**Render Dashboard:**
- Go to your service → **Logs** tab
- Look for these success messages:
  ```
  ✅ MongoDB connected successfully
  ✅ GROQ client initialized
  ✅ Application startup complete
  ```

**No more warnings:**
  ```
  ⚠️ GROQ_API_KEY not found  ← Should be GONE
  ⚠️ MongoDB connection failed ← Should be GONE
  ```

---

## 🎉 Success Checklist

After completing setup, verify:

- [ ] Created `.env` file in `backend/` folder
- [ ] Added `GROQ_API_KEY` to `.env`
- [ ] Added `MONGODB_URL` to `.env`
- [ ] Started backend locally without errors
- [ ] Added both variables to Render Dashboard
- [ ] Render deployment successful (check logs)
- [ ] Chatbot responds to messages
- [ ] User signup/login works

---

## 🆘 Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution:**
1. Verify `.env` file exists in `backend/` folder
2. Check for typos: `GROQ_API_KEY` (exact spelling)
3. Ensure no spaces: `GROQ_API_KEY=gsk_...` (no space around `=`)
4. Restart backend after editing `.env`

### Issue: "MongoDB connection timeout"

**Possible Causes:**
1. **Wrong connection string** - Double-check from MongoDB Atlas
2. **Password has special characters** - URL-encode them:
   - `@` → `%40`
   - `#` → `%23`
   - `!` → `%21`
3. **IP not whitelisted** - Add `0.0.0.0/0` in Network Access
4. **Cluster still initializing** - Wait 5 minutes after creation

**Test Connection:**
```bash
pip install pymongo
python -c "from pymongo import MongoClient; client = MongoClient('YOUR_MONGODB_URL'); print(client.server_info())"
```

### Issue: "Module 'motor' not found"

```bash
cd backend
pip install -r requirements.txt
```

### Issue: Changes not reflected on Render

1. Check **Environment** tab shows your variables
2. Trigger manual deploy: **Manual Deploy** → **Deploy latest commit**
3. Check **Logs** tab for error messages
4. Verify commit pushed to GitHub: `git push origin main`

---

## 📚 Additional Resources

- **MongoDB Atlas Tutorial**: https://www.mongodb.com/docs/atlas/getting-started/
- **Groq API Documentation**: https://console.groq.com/docs/quickstart
- **Render Environment Variables**: https://render.com/docs/configure-environment-variables
- **FastAPI .env Integration**: https://fastapi.tiangolo.com/advanced/settings/

---

## 🔐 Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Use strong passwords** - Auto-generate from MongoDB Atlas
3. **Rotate API keys regularly** - Every 3-6 months
4. **Limit IP access in production** - Don't use `0.0.0.0/0` long-term
5. **Use Render's "Secret File"** - For sensitive multi-line configs

---

**Need Help?** Check the logs:
- **Local**: Terminal output when running `python main_fastapi.py`
- **Production**: Render Dashboard → Your Service → **Logs** tab
