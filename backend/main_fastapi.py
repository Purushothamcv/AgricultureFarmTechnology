"""
SmartAgri-AI FastAPI Backend - FIXED FOR RENDER
===============================================

Key fixes:
1. No code executes at import time that could block/crash
2. All heavy initialization moved to @app.on_event("startup")
3. Minimal uvicorn startup
4. Production URLs only
5. Services imported with proper error handling
6. Database index creation moved to startup event

"""

import os
import sys
import traceback
from dotenv import load_dotenv

# ============================================================
# PHASE 1: MINIMAL BOOTSTRAP
# ============================================================
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

print("\n" + "="*70)
print("[BOOTSTRAP] SmartAgri Backend Initialization")
print("="*70)
print(f"[BOOTSTRAP] PORT: {os.getenv('PORT', 'NOT SET')}")
print(f"[BOOTSTRAP] Environment: {os.getenv('ENVIRONMENT', 'production')}")
print(f"[BOOTSTRAP] LOW_MEMORY_MODE: {os.getenv('LOW_MEMORY_MODE', 'true')}")
print("="*70)

# ============================================================
# PHASE 2: CORE IMPORTS ONLY (NO BLOCKING)
# ============================================================
print("\n[INIT] Importing FastAPI...")
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import asyncio
    print("[OK] Core FastAPI imports successful")
except Exception as e:
    print(f"[FATAL] FastAPI import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASE 3: CREATE APP FIRST
# ============================================================
print("\n[INIT] Creating FastAPI app...")
try:
    app = FastAPI(
        title="SmartAgri API",
        description="Smart Agriculture Decision Support System",
        version="1.0.0"
    )
    print("[OK] FastAPI app instance created")
except Exception as e:
    print(f"[FATAL] Failed to create app: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASE 4: CONFIGURE CORS
# ============================================================
print("\n[INIT] Configuring CORS...")
try:
    allowed_origins = [
        "https://agriculture-farm-technology.vercel.app",
        "https://smartagri-backend-ckcz.onrender.com",
        "http://localhost:3000",  # Local dev
        "http://localhost:5173",  # Local Vite
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("[OK] CORS configured")
except Exception as e:
    print(f"[WARN] CORS configuration failed: {e}")

# ============================================================
# PHASE 5: CRITICAL ENDPOINTS (NO EXTERNAL CALLS)
# ============================================================
print("\n[INIT] Registering critical endpoints...")

@app.get("/health")
async def health_check():
    """Health check - Render uses this to detect running app"""
    return {
        "status": "ok",
        "app": "SmartAgri-AI",
        "version": "1.0.0",
        "ready": True
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "message": "SmartAgri API",
        "version": "1.0.0"
    }

print("[OK] Critical endpoints registered (/health, /)")

# ============================================================
# PHASE 6: OPTIONAL SERVICE IMPORTS (WITH FALLBACKS)
# ============================================================
print("\n[INIT] Loading optional services (non-blocking)...")

# Auth router
auth_router = None
try:
    from auth import router as auth_router
    print("[OK] Auth service imported")
except Exception as e:
    print(f"[SKIP] Auth service: {e}")

# Crop service
crop_router = None
predict_crop_func = None
try:
    from crop_service import predict_crop, fetch_all_location_data, fetch_weather_and_ph_only
    from crop_models import ManualCropInput, LocationCropInput, CropPredictionResponse
    predict_crop_func = predict_crop
    print("[OK] Crop service imported")
except Exception as e:
    print(f"[SKIP] Crop service: {e}")

# Database (for optional features only)
connect_to_mongodb = None
get_database = None
try:
    from database import connect_to_mongodb, get_database, close_mongodb_connection
    print("[OK] Database utilities imported")
except Exception as e:
    print(f"[SKIP] Database utilities: {e}")

# Chatbot service (optional)
chatbot_router = None
try:
    from chatbot_service import router as chatbot_router
    print("[OK] Chatbot service imported")
except Exception as e:
    print(f"[SKIP] Chatbot service: {e}")

# Disease services (optional)
fruit_disease_router = None
try:
    from api_fruit_disease_production import router as fruit_disease_router
    print("[OK] Fruit disease service imported")
except Exception as e:
    print(f"[SKIP] Fruit disease service: {e}")

plant_disease_router = None
try:
    from plant_disease_service import router as plant_disease_router
    print("[OK] Plant disease service imported")
except Exception as e:
    print(f"[SKIP] Plant disease service: {e}")

# Other optional services
remedy_router = None
try:
    from remedy_generation_service import router as remedy_router
    print("[OK] Remedy service imported")
except Exception as e:
    print(f"[SKIP] Remedy service: {e}")

yield_startup = None
try:
    from yield_prediction_service import get_yield_service, startup_event as yield_startup
    print("[OK] Yield service imported")
except Exception as e:
    print(f"[SKIP] Yield service: {e}")

agentic_router = None
try:
    from agentic_ai import router as agentic_router
    print("[OK] Agentic AI service imported")
except Exception as e:
    print(f"[SKIP] Agentic AI service: {e}")

# NEW: Fertilizer prediction service (PHASE 6 Extension)
fertilizer_router = None
try:
    from api_fertilizer import router as fertilizer_router
    print("[OK] Fertilizer service imported")
except Exception as e:
    print(f"[SKIP] Fertilizer service: {e}")

# NEW: Stress prediction service (PHASE 6 Extension)
stress_router = None
try:
    from api_stress import router as stress_router
    print("[OK] Stress service imported")
except Exception as e:
    print(f"[SKIP] Stress service: {e}")

# NEW: Yield API service (PHASE 6 Extension)
yield_router = None
try:
    from api_yield import router as yield_router
    print("[OK] Yield API service imported")
except Exception as e:
    print(f"[SKIP] Yield API service: {e}")

# NEW: Fruit disease API v2 (PHASE 6 Extension)
fruit_disease_v2_router = None
try:
    from fruit_disease_api_v2 import router as fruit_disease_v2_router
    print("[OK] Fruit disease API v2 imported")
except Exception as e:
    print(f"[SKIP] Fruit disease API v2: {e}")

print("\n[OK] Optional services loaded")

# ============================================================
# PHASE 7: REGISTER AVAILABLE ROUTERS
# ============================================================
print("\n[INIT] Registering routers...")

if auth_router:
    app.include_router(auth_router)
    print("[OK] Auth routes registered")

if fruit_disease_router:
    app.include_router(fruit_disease_router)
    print("[OK] Fruit disease routes registered")

if fruit_disease_v2_router:
    app.include_router(fruit_disease_v2_router)
    print("[OK] Fruit disease v2 routes registered")

if plant_disease_router:
    app.include_router(plant_disease_router)
    print("[OK] Plant disease routes registered")

if remedy_router:
    app.include_router(remedy_router)
    print("[OK] Remedy routes registered")

if chatbot_router:
    app.include_router(chatbot_router)
    print("[OK] Chatbot routes registered")

if agentic_router:
    app.include_router(agentic_router)
    print("[OK] Agentic AI routes registered")

if fertilizer_router:
    app.include_router(fertilizer_router)
    print("[OK] Fertilizer routes registered")

if stress_router:
    app.include_router(stress_router)
    print("[OK] Stress routes registered")

if yield_router:
    app.include_router(yield_router)
    print("[OK] Yield API routes registered")

print("[OK] All available routes registered")

# ============================================================
# PHASE 8: EXCEPTION HANDLING MIDDLEWARE
# ============================================================
print("\n[INIT] Setting up middleware...")

@app.middleware("http")
async def catch_unhandled_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        print(f"[ERROR] Exception in {request.url.path}: {exc}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)}
        )

print("[OK] Exception middleware registered")

# ============================================================
# PHASE 9: STARTUP EVENT (INITIALIZATION)
# ============================================================
print("\n[INIT] Configuring startup event...")

@app.on_event("startup")
async def startup_event():
    """
    Non-blocking startup - heavy operations done here asynchronously
    CRITICAL: Return quickly so port binding completes
    """
    print("\n" + "="*70)
    print("[STARTUP] FastAPI application startup")
    print("="*70)
    
    # Try to connect to MongoDB in background
    if connect_to_mongodb:
        try:
            print("[STARTUP] Attempting MongoDB connection...")
            await asyncio.wait_for(connect_to_mongodb(), timeout=10.0)
            print("[OK] MongoDB connected")
        except asyncio.TimeoutError:
            print("[WARN] MongoDB connection timeout")
        except Exception as e:
            print(f"[WARN] MongoDB connection failed: {e}")
    
    print("[STARTUP] Application ready to serve requests")
    print("="*70 + "\n")

# ============================================================
# PHASE 10: SHUTDOWN EVENT
# ============================================================
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("[SHUTDOWN] Application shutting down")
    if get_database:
        try:
            from database import close_mongodb_connection
            await close_mongodb_connection()
        except:
            pass

# ============================================================
# PHASE 11: CONFIRMATION
# ============================================================
print("\n" + "="*70)
print("[OK] BOOTSTRAP COMPLETE - App ready for uvicorn")
print("="*70 + "\n")

# ============================================================
# LOCAL TESTING ONLY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"
    print(f"\n[LOCAL] Starting uvicorn on {host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
