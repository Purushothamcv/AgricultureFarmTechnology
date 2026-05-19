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

import sys
# FIX: Add UTF-8 encoding support for Windows environments to prevent charmap errors
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import os
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
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    print(f"[CORS] Allowed origins: {allowed_origins}")
    print(f"[CORS] Credentials enabled: True")
    print(f"[CORS] Methods: ['*']")
    print(f"[CORS] Headers: ['*']")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,  # IMPORTANT: Enables credentials, cookies, authorization headers
        allow_methods=["*"],     # Allow all HTTP methods
        allow_headers=["*"],     # Allow all headers
        expose_headers=["*"],    # Expose all response headers
        max_age=3600,            # Cache preflight for 1 hour
    )
    print("[OK] CORS configured - credentials and specific origins enabled")
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

@app.get("/health/models")
async def health_check_models():
    """Health check for ML models"""
    try:
        from plant_disease_service import plant_disease_model, class_mapping
        plant_disease_loaded = plant_disease_model is not None and len(class_mapping) > 0
    except:
        plant_disease_loaded = False
    
    # Check fertilizer
    try:
        from api_fertilizer import get_fertilizer_service
        service = get_fertilizer_service()
        fertilizer_loaded = service is not None and hasattr(service, 'model') and service.model is not None
    except:
        fertilizer_loaded = False
    
    # Check stress
    try:
        from api_stress import get_stress_service
        service = get_stress_service()
        stress_loaded = service is not None
    except:
        stress_loaded = False
    
    return {
        "status": "ok" if plant_disease_loaded else "degraded",
        "models": {
            "plant_disease": plant_disease_loaded,
            "fertilizer": fertilizer_loaded,
            "stress": stress_loaded
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "message": "SmartAgri API",
        "version": "1.0.0"
    }

# ============================================================
# DIRECT YIELD ENDPOINTS (No prefix, for frontend compatibility)
# ============================================================
@app.get("/yield/states")
async def get_states_direct():
    """
    Get list of Indian states for yield prediction
    Direct endpoint (non-prefixed) for frontend compatibility
    """
    try:
        INDIA_STATES = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
            "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
            "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
            "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
            "Uttar Pradesh", "Uttarakhand", "West Bengal"
        ]
        return {
            "success": True,
            "states": INDIA_STATES,
            "message": "States loaded successfully"
        }
    except Exception as e:
        print(f"[ERROR] /yield/states: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

print("[OK] Critical endpoints registered (/health, /, /yield/states)")

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

# Weather & Location router
weather_location_router = None
try:
    from weather_location import router as weather_location_router
    print("[OK] Weather & Location service imported")
except Exception as e:
    print(f"[SKIP] Weather & Location service: {e}")

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

# Crop prediction API router
crop_prediction_router = None
try:
    from api_crop_prediction import router as crop_prediction_router
    print("[OK] Crop prediction API imported")
except Exception as e:
    print(f"[SKIP] Crop prediction API: {e}")

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

if weather_location_router:
    app.include_router(weather_location_router)
    print("[OK] Weather & Location routes registered")

if crop_prediction_router:
    app.include_router(crop_prediction_router)
    print("[OK] Crop prediction routes registered")

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

# ============================================================
# PHASE 8b: AUTHENTICATION MIDDLEWARE - SKIP OPTIONS
# ============================================================
@app.middleware("http")
async def skip_options_in_auth(request: Request, call_next):
    """
    Middleware to skip authentication for OPTIONS requests.
    OPTIONS requests are used by CORS preflight and should not require JWT validation.
    
    IMPORTANT: When using withCredentials: true, must NOT use wildcard "*" for
    Access-Control-Allow-Origin. Use specific origins instead.
    """
    if request.method == "OPTIONS":
        # Get the requesting origin
        origin = request.headers.get("origin", "")
        
        # List of allowed origins (must match CORS config)
        allowed_origins = [
            "https://agriculture-farm-technology.vercel.app",
            "https://smartagri-backend-ckcz.onrender.com",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
        
        # Check if origin is allowed
        allowed_origin = origin if origin in allowed_origins else allowed_origins[0]
        
        # Return 200 OK for CORS preflight requests with specific origin
        return JSONResponse(
            status_code=200,
            content={"status": "ok"},
            headers={
                "Access-Control-Allow-Origin": allowed_origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "3600",
            }
        )
    
    # For non-OPTIONS requests, continue with normal processing
    response = await call_next(request)
    return response

print("[OK] OPTIONS request handler registered (with credentials support)")

# ============================================================
# PHASE 9: STARTUP EVENT (INITIALIZATION)
# ============================================================
print("\n[INIT] Configuring startup event...")

def print_registered_routes():
    """Debug function to print all registered routes"""
    print("\n" + "="*70)
    print("[ROUTES] All Registered API Routes:")
    print("="*70)
    
    routes_list = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = list(route.methods) if hasattr(route, 'methods') else ['GET']
            for method in methods:
                if method != "HEAD":  # Skip HEAD methods
                    routes_list.append(f"{method:6} {route.path}")
        elif hasattr(route, 'path'):
            routes_list.append(f"GET    {route.path}")
    
    # Sort and print
    for route in sorted(routes_list):
        print(f"  {route}")
    
    print("="*70)
    print(f"[ROUTES] Total routes registered: {len(routes_list)}")
    print("="*70 + "\n")

@app.on_event("startup")
async def startup_event():
    """
    Non-blocking startup - heavy operations done here asynchronously
    CRITICAL: Return quickly so port binding completes
    """
    print("\n" + "="*70)
    print("[STARTUP] FastAPI application startup")
    print("="*70)
    
    # Print all registered routes for debugging
    print_registered_routes()
    
    # Initialize Groq AI client for chatbot service
    print("\n[STARTUP] Initializing Groq AI client...")
    try:
        from chatbot_service import initialize_groq_client
        success = initialize_groq_client()
        if success:
            print("[OK] Groq AI client initialized successfully")
        else:
            print("[WARN] Groq AI client initialization returned False - GROQ_API_KEY may not be set")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Initialize Plant Disease Detection Service
    print("\n[STARTUP] Initializing ML services...")
    try:
        from plant_disease_service import startup_event as plant_disease_startup
        await plant_disease_startup()
        print("[OK] Plant disease service initialized")
    except Exception as e:
        print(f"[WARN] Plant disease service initialization: {e}")
    
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
    
    print("\n[STARTUP] Application ready to serve requests")
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
