# ============================================
# CRITICAL: SUPPRESS TENSORFLOW EARLY
# ============================================
# Must be before ANY imports (especially TensorFlow)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Configure production logging
import logging_config

import sys
from dotenv import load_dotenv

load_dotenv()  # Load .env file into environment
print(f"[DEBUG] Environment Variables Loaded")
print(f"[DEBUG] GOOGLE_CLIENT_ID: {'OK' if os.getenv('GOOGLE_CLIENT_ID') else 'MISSING'}")
print(f"[DEBUG] GOOGLE_CLIENT_SECRET: {'OK' if os.getenv('GOOGLE_CLIENT_SECRET') else 'MISSING'}")
print(f"[DEBUG] MONGODB_URL: {'OK' if os.getenv('MONGODB_URL') else 'MISSING'}")

# Add backend directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# LAZY LOAD: Import model_manager for on-demand loading
# ============================================
from model_manager import (
    get_yield_model, get_best_window_model, get_stress_model, 
    get_crop_model, get_fert_model,
    cleanup_after_inference, cleanup_all_models,
    get_model_stats, suppress_tensorflow_logging
)

# Suppress TensorFlow logging before any TF imports
suppress_tensorflow_logging()

# Now import FastAPI and other dependencies (NO TensorFlow yet!)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import asyncio
import gc
import traceback

# ============================================
# CRITICAL: LOG EARLY FOR RENDER DEBUGGING
# ============================================
print("\n" + "="*60)
print("[START] SmartAgri-AI FastAPI Backend Initialization")
print("="*60)
print(f"[DEBUG] Python Path: {sys.path[:2]}")
print(f"[DEBUG] Current Directory: {os.getcwd()}")
print(f"[DEBUG] Main File Location: {os.path.abspath(__file__)}")
print(f"[DEBUG] PORT env var: {os.environ.get('PORT', 'Not set (will use 8000)')}")
print("="*60 + "\n")

print("[INFO] SmartAgri-AI Backend starting...")

# ============================================
# SAFE IMPORTS WITH ERROR HANDLING
# ============================================
# Always import core dependencies first
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level
from auth import router as auth_router
from database import connect_to_mongodb, close_mongodb_connection, get_database
from db_helpers import get_database_stats
from crop_models import ManualCropInput, LocationCropInput, CropPredictionResponse, LocationDataResponse, WeatherAndPHDataResponse
from crop_service import predict_crop, fetch_all_location_data, fetch_weather_and_ph_only

# Import services with fallbacks
try:
    from fruit_disease_service import router as fruit_disease_router, startup_event as fruit_startup
    print("[OK] Fruit disease service imported")
except ImportError as e:
    print(f"[SKIP] Fruit disease service not available: {e}")
    fruit_disease_router = None
    fruit_startup = None

try:
    from api_fruit_disease_production import router as fruit_disease_prod_router, startup_event as fruit_prod_startup
    print("[OK] Production fruit disease API imported")
except ImportError as e:
    print(f"[SKIP] Production fruit disease API not available: {e}")
    fruit_disease_prod_router = None
    fruit_prod_startup = None

try:
    from fruit_disease_api_v2 import router as fruit_disease_v2_router, startup_event as fruit_v2_startup
    print("[OK] Fruit disease V2 API imported")
except ImportError as e:
    print(f"[SKIP] Fruit disease V2 API not available: {e}")
    fruit_disease_v2_router = None
    fruit_v2_startup = None

try:
    from fruit_disease_detection import router as fruit_disease_detection_router, startup_event as fruit_detection_startup
    print("[OK] Fruit disease detection service (with selection) imported")
except ImportError as e:
    print(f"[SKIP] Fruit disease detection service not available: {e}")
    fruit_disease_detection_router = None
    fruit_detection_startup = None

try:
    from plant_disease_service import router as plant_disease_router, startup_event as plant_disease_startup
    print("[OK] Plant disease service imported")
except ImportError as e:
    print(f"[SKIP] Plant disease service not available: {e}")
    plant_disease_router = None
    plant_disease_startup = None

try:
    from chatbot_service import router as chatbot_router, startup_event as chatbot_startup
    print("[OK] Chatbot service imported")
except ImportError as e:
    print(f"[SKIP] Chatbot service not available: {e}")
    chatbot_router = None
    chatbot_startup = None

try:
    from remedy_generation_service import router as remedy_router, startup_event as remedy_startup
    print("[OK] Remedy generation service imported")
except ImportError as e:
    print(f"[SKIP] Remedy generation service not available: {e}")
    remedy_router = None
    remedy_startup = None

try:
    from yield_prediction_service import get_yield_service, startup_event as yield_startup
    print("[OK] Yield prediction service imported")
except ImportError as e:
    print(f"[SKIP] Yield prediction service not available: {e}")
    yield_startup = None

try:
    from agentic_ai import router as agentic_router, startup_event as agentic_ai_startup
    print("[OK] Agentic AI service imported")
except ImportError as e:
    print(f"[SKIP] Agentic AI service not available: {e}")
    agentic_router = None
    agentic_ai_startup = None

try:
    from fertilizer_prediction_service import get_fertilizer_service
    print("[OK] Fertilizer prediction service imported")
except ImportError as e:
    print(f"[SKIP] Fertilizer prediction service not available: {e}")
    get_fertilizer_service = None

try:
    from fertilizer_auto_fill_service import get_fertilizer_auto_fill_service
    print("[OK] Fertilizer auto-fill service imported")
except ImportError as e:
    print(f"[SKIP] Fertilizer auto-fill service not available: {e}")
    get_fertilizer_auto_fill_service = None

try:
    from stress_prediction_service import stress_service
    print("[OK] Stress prediction service imported")
except ImportError as e:
    print(f"[SKIP] Stress prediction service not available: {e}")
    stress_service = None

try:
    from soil_data_service import get_soil_data_service
    print("[OK] Soil data service imported")
except ImportError as e:
    print(f"[SKIP] Soil data service not available: {e}")
    get_soil_data_service = None

try:
    from stress_agent import generate_stress_insights
    print("[OK] Stress agent imported")
except ImportError as e:
    print(f"[SKIP] Stress agent not available: {e}")
    generate_stress_insights = None

print("[INFO] All imports successful")

# ============================================================================
# LOW MEMORY MODE (512MB RAM on Render free tier)
# ============================================================================
LOW_MEMORY_MODE = os.getenv("LOW_MEMORY_MODE", "true").lower() == "true"
if LOW_MEMORY_MODE:
    print("[WARN] LOW_MEMORY_MODE enabled - heavy ML models will not load at startup")
else:
    print("[INFO] LOW_MEMORY_MODE disabled - all services will initialize")

app = FastAPI(title="SmartAgri API", description="Smart Agriculture Decision Support System", version="1.0.0")
print("[INFO] FastAPI app instance created and ready for uvicorn startup")


@app.middleware("http")
async def catch_unhandled_exceptions(request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc), "path": str(request.url.path)}
        )

# Background initialization task (doesn't block port binding)
async def initialize_services_background():
    """Initialize all services in the background (non-blocking)"""
    try:
        print("\n[BACKGROUND] Starting service initialization...")
        
        if LOW_MEMORY_MODE:
            print("[WARN] LOW_MEMORY_MODE: Skipping heavy TensorFlow model loading")
    except Exception as e:
        print(f"[ERROR] Exception in background initialization setup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize each service with error handling
    # Skip TensorFlow models in low memory mode
    
    if fruit_startup and not LOW_MEMORY_MODE:
        try:
            await fruit_startup()
            print("[OK] Fruit disease service (legacy) initialized")
        except Exception as e:
            print(f"[WARN] Fruit disease service (legacy) failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Fruit disease service (low memory mode)")
    
    if fruit_prod_startup and not LOW_MEMORY_MODE:
        try:
            print("[INIT] Initializing Production Fruit Disease Detection...")
            await fruit_prod_startup()
            print("[OK] Production fruit disease service initialized")
        except Exception as e:
            print(f"[WARN] Production fruit disease service failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Production fruit disease service (low memory mode)")
    
    if fruit_v2_startup and not LOW_MEMORY_MODE:
        try:
            print("[INIT] Initializing Fruit Disease V2 (Clean Model)...")
            await fruit_v2_startup()
            print("[OK] Fruit Disease V2 service initialized")
        except Exception as e:
            print(f"[WARN] Fruit Disease V2 service failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Fruit Disease V2 service (low memory mode)")
    
    if fruit_detection_startup and not LOW_MEMORY_MODE:
        try:
            print("[INIT] Initializing Fruit Disease Detection (with Selection)...")
            await fruit_detection_startup()
            print("[OK] Fruit disease detection service initialized")
        except Exception as e:
            print(f"[WARN] Fruit disease detection service failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Fruit disease detection service (low memory mode)")
    
    if plant_disease_startup and not LOW_MEMORY_MODE:
        try:
            print("[INIT] Initializing Plant Leaf Disease Detection...")
            await plant_disease_startup()
            print("[OK] Plant disease service initialized")
        except Exception as e:
            print(f"[WARN] Plant disease service failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Plant disease service (low memory mode)")
    
    if chatbot_startup:
        try:
            print("[INIT] Initializing AI Chatbot Service...")
            await chatbot_startup()
            print("[OK] Chatbot service initialized")
        except Exception as e:
            print(f"[WARN] Chatbot service failed (check GROQ_API_KEY): {e}")
    
    if remedy_startup:
        try:
            print("[INIT] Initializing Disease Remedy Generation Service...")
            await remedy_startup()
            print("[OK] Remedy generation service initialized")
        except Exception as e:
            print(f"[WARN] Remedy generation service failed: {e}")
    
    if yield_startup and not LOW_MEMORY_MODE:
        try:
            await yield_startup()
            print("[OK] Yield prediction service initialized")
        except Exception as e:
            print(f"[WARN] Yield service failed: {e}")
    elif LOW_MEMORY_MODE:
        print("[SKIP] Yield prediction service (low memory mode)")

    if agentic_ai_startup:
        try:
            print("[INIT] Initializing Agentic AI Crop Service...")
            await agentic_ai_startup()
            print("[OK] Agentic AI service initialized")
        except Exception as e:
            print(f"[WARN] Agentic AI service failed: {e}")

    if get_fertilizer_service and not LOW_MEMORY_MODE:
        try:
            print("[INIT] Initializing Fertilizer Prediction Service...")
            fertilizer_service = get_fertilizer_service()
            fertilizer_service.load_model()
            print("[OK] Fertilizer service initialized")
        except Exception as e:
            print(f"[WARN] Fertilizer service failed: {e}")
            import traceback
            traceback.print_exc()
    elif LOW_MEMORY_MODE:
        print("[SKIP] Fertilizer service (low memory mode)")
    
    print("\n[BACKGROUND] Service initialization complete\n")

# Event handlers for MongoDB connection
@app.on_event("startup")
async def startup_event():
    """Quick startup - only connect MongoDB, defer service initialization"""
    try:
        print("\n[START] Starting SmartAgri API (fast startup mode)...\n")
        
        # MongoDB connection with error handling - don't block startup if it fails
        try:
            await connect_to_mongodb()
            print("[OK] MongoDB Connected")
        except Exception as e:
            print(f"[WARN] MongoDB connection failed: {e}")
            import traceback
            traceback.print_exc()
            print("[WARN] Continuing startup - database operations will fail until connection restored")
        
        # START service initialization in background (doesn't block port binding)
        asyncio.create_task(initialize_services_background())
        
        print("\n[OK] Port ready - services loading in background...\n")
    except Exception as e:
        print(f"\n[ERROR] CRITICAL: Startup event failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to let FastAPI/uvicorn handle it

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    print("[SHUTDOWN] Shutting down SmartAgri API...")
    cleanup_all_models()
    await close_mongodb_connection()

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:5173",  # Vite dev server (127.0.0.1)
    "https://agriculture-farm-technology.vercel.app",
    "https://smartagri-backend-ckcz.onrender.com",  # Allow backend to call itself
]
allowed_origin_regex = r"^https://.*\.(vercel\.app|onrender\.com)$"

print(f"[CORS] CORS enabled for origins: {allowed_origins}")
print(f"[CORS] CORS enabled for regex origins: {allowed_origin_regex}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers - only include if available
print("[INFO] Registering API routes...")
app.include_router(auth_router)
print("[OK] Auth routes registered")

if fruit_disease_router:
    app.include_router(fruit_disease_router)  # Legacy endpoint
    print("[OK] Fruit disease routes (legacy) registered")

if chatbot_router:
    app.include_router(chatbot_router)  # AI Chatbot with voice assistance
    print("[OK] Chatbot routes registered")

if fruit_disease_prod_router:
    app.include_router(fruit_disease_prod_router)  # PRODUCTION endpoint (frozen model)
    print("[OK] Fruit disease production routes registered")

if fruit_disease_v2_router:
    app.include_router(fruit_disease_v2_router)  # V2 endpoint (NEW clean trained model - 92%+)
    print("[OK] Fruit disease V2 routes registered")

if fruit_disease_detection_router:
    app.include_router(fruit_disease_detection_router)  # Fruit disease detection with fruit selection
    print("[OK] Fruit disease detection routes registered")

if plant_disease_router:
    app.include_router(plant_disease_router)  # Plant Leaf Disease Detection
    print("[OK] Plant disease routes registered")

if remedy_router:
    app.include_router(remedy_router)  # Disease Remedy Generation Service
    print("[OK] Remedy generation routes registered")

if agentic_router:
    app.include_router(agentic_router)  # Agentic AI crop fetching
    print("[OK] Agentic AI routes registered")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API requests
class SprayRequest(BaseModel):
    temperature: float
    humidity: float
    windSpeed: float
    rainfall: float
    timeOfDay: str = ""

# Yield Prediction Request Models
class YieldPredictionRequest(BaseModel):
    """Request model for APY-based yield prediction"""
    state: str
    district: str
    crop: str
    year: int
    season: str
    area: float

class LegacyYieldRequest(BaseModel):
    """Legacy yield prediction request (for backward compatibility)"""
    crop: str = 'potato'
    area: float = 1
    soilMoisture: float = 0.5
    ozone: float = 40
    temperature: float = None
    humidity: float = None
    rainfall: float = None
    lat: float = None
    lon: float = None


class FertilizerAutoFillRequest(BaseModel):
    latitude: float
    longitude: float

# ============================================
# LAZY LOADING: Models are now loaded on-demand via model_manager
# NO global model loading to save startup memory
# ============================================

# ====================
# Main Application Routes
# ====================
# Note: Authentication routes (/auth/register, /auth/login) are now handled 
# by the auth.py module and automatically included via app.include_router(auth_router)

@app.get("/")
async def root():
    """Root endpoint - health check"""
    try:
        get_database()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return {
        "status": "ok",
        "message": "SmartAgri API is running",
        "version": "1.0.0",
        "database": db_status
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        get_database()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return {
        "status": "running",
        "database": db_status,
        "backend": "healthy" if db_status == "connected" else "degraded"
    }

@app.get("/test-db")
async def test_database_connection():
    """
    Test async MongoDB connection (used by FastAPI startup)
    Diagnostic endpoint for troubleshooting connection issues
    """
    try:
        from database import client as async_client
        if async_client is None:
            return {
                "status": "error",
                "connection_type": "Async (Motor)",
                "message": "Async MongoDB client not initialized",
                "advice": "Check if startup event completed successfully"
            }
        
        # Test ping
        await async_client.admin.command('ping')
        
        return {
            "status": "success",
            "connection_type": "Async MongoDB (Motor)",
            "message": "MongoDB Atlas Connected via Motor (async)",
            "database": "Connected",
            "details": "Motor async connection working correctly"
        }
    except Exception as e:
        return {
            "status": "error",
            "connection_type": "Async (Motor)",
            "message": str(e),
            "error_type": type(e).__name__,
            "suggestions": [
                "1. Check if MONGODB_URL environment variable is set",
                "2. Verify MongoDB Atlas cluster is running and accessible",
                "3. Check IP whitelist includes Render service IP",
                "4. Verify credentials in connection string",
                "5. Check network connectivity to mongodb.net"
            ]
        }

@app.get("/api/database/stats")
async def get_db_stats():
    """Get FinalProject database statistics"""
    stats = await get_database_stats()
    return stats

@app.get("/api/models/stats")
async def get_models_stats():
    """Get ML models cache statistics and status"""
    return get_model_stats()

@app.get("/test-mongodb")
def test_mongodb_connection():
    """
    Direct MongoDB Atlas connection test
    Tests the synchronous PyMongo connection from db.py
    """
    try:
        from db import client
        client.admin.command('ping')
        
        # Also check if we can access collections
        from db import users_collection, chat_sessions_collection
        
        return {
            "status": "success",
            "message": "MongoDB Atlas Connected",
            "connection_type": "MongoDB Atlas (PyMongo)",
            "database": "Connected",
            "collections": {
                "users": "accessible",
                "chat_sessions": "accessible"
            },
            "details": "PyMongo synchronous connection working"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "suggestions": [
                "1. Check MongoDB Atlas cluster is running",
                "2. Verify MONGODB_URL in .env file",
                "3. Check Network Access whitelist includes your IP",
                "4. Verify username and password are correct"
            ]
        }

# ====================
# HTML Page Routes
# ====================

@app.get("/crop-recommendation", response_class=HTMLResponse)
async def crop_recommendation_page(request: Request):
    """Serve the crop recommendation HTML page"""
    return templates.TemplateResponse("crop_recommendation.html", {"request": request})

# ====================
# Crop Recommendation Endpoints
# ====================

@app.post("/predict/manual", response_model=CropPredictionResponse)
async def predict_crop_manual(input_data: ManualCropInput):
    """
    Crop recommendation based on manual input
    
    User provides all parameters manually through a form
    """
    try:
        # Make prediction
        crop, confidence = predict_crop(
            nitrogen=input_data.nitrogen,
            phosphorus=input_data.phosphorus,
            potassium=input_data.potassium,
            temperature=input_data.temperature,
            humidity=input_data.humidity,
            ph=input_data.ph,
            rainfall=input_data.rainfall,
            ozone=input_data.ozone
        )
        
        return CropPredictionResponse(
            success=True,
            crop=crop,
            confidence=confidence,
            input_values=input_data.dict(),
            message="Crop recommendation generated successfully from manual input"
        )
    
    except Exception as e:
        return CropPredictionResponse(
            success=False,
            crop="Unknown",
            input_values=input_data.dict(),
            message=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/location", response_model=CropPredictionResponse)
async def predict_crop_location(input_data: LocationCropInput):
    """
    Crop recommendation based on location (map selection) with manual NPK input
    
    Important: NPK values MUST be provided by the user from soil test.
    Only weather and pH data can be fetched from location.
    
    Args:
        input_data: LocationCropInput with:
          - REQUIRED: latitude, longitude, nitrogen, phosphorus, potassium
          - OPTIONAL: temperature, humidity, ph, rainfall, ozone (will fetch if not provided)
    """
    try:
        # NPK values are REQUIRED - they come from user's soil test
        # (LocationCropInput model now enforces this)
        nitrogen = input_data.nitrogen
        phosphorus = input_data.phosphorus
        potassium = input_data.potassium
        
        # Fetch weather and pH data if not all provided
        if any(v is None for v in [
            input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall, input_data.ozone
        ]):
            location_data = await fetch_weather_and_ph_only(
                input_data.latitude,
                input_data.longitude
            )
        else:
            location_data = {}
        
        # Use provided values or fallback to fetched values (for weather/pH only)
        temperature = input_data.temperature if input_data.temperature is not None else location_data.get("temperature", 25)
        humidity = input_data.humidity if input_data.humidity is not None else location_data.get("humidity", 70)
        ph = input_data.ph if input_data.ph is not None else location_data.get("ph", 6.5)
        rainfall = input_data.rainfall if input_data.rainfall is not None else location_data.get("rainfall", 100)
        ozone = input_data.ozone if input_data.ozone is not None else location_data.get("ozone", 30)
        
        # Make prediction
        crop, confidence = predict_crop(
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            ozone=ozone
        )
        
        return CropPredictionResponse(
            success=True,
            crop=crop,
            confidence=confidence,
            input_values={
                "latitude": input_data.latitude,
                "longitude": input_data.longitude,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "ozone": ozone
            },
            message="Crop recommendation generated successfully (NPK from soil test, weather data auto-filled)"
        )
    
    except Exception as e:
        return CropPredictionResponse(
            success=False,
            crop="Unknown",
            input_values={"latitude": input_data.latitude, "longitude": input_data.longitude},
            message=f"Prediction failed: {str(e)}"
        )


@app.get("/api/location-data", response_model=WeatherAndPHDataResponse)
async def get_location_data(latitude: float, longitude: float):
    """
    Fetch weather and pH data for crop recommendation (NO NPK values)
    
    Used to auto-populate weather fields when user selects location on map.
    
    Important: NPK values are INTENTIONALLY excluded as they must come from 
    the user's soil test. The frontend should show NPK as manual-entry-only fields.
    """
    try:
        data = await fetch_weather_and_ph_only(latitude, longitude)
        return WeatherAndPHDataResponse(**data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch location data: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get_agri_data")
def get_agri_data(lat: float, lon: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'error': 'Weather data unavailable'}, status_code=500)
    recommendations = "Use the dashboard features for yield, fertilizer, and stress prediction."
    return {"weather": weather, "recommendations": recommendations}

@app.get("/predict_yield")
def predict_yield(lat: float, lon: float, ozone: float, soil: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'result': None}, status_code=400)
    temp = weather['temp']
    rain = weather['rain']
    features = pd.DataFrame([[ozone, temp, rain, soil]], columns=["ozone", "temp", "rain", "soil"])
    
    try:
        yield_model = get_yield_model()
        if yield_model is None:
            return JSONResponse({'error': 'Yield model not available'}, status_code=503)
        prediction = yield_model.predict(features)[0]
        cleanup_after_inference()
        return {"result": f"Predicted Potato Yield: {prediction:.2f} tonnes/hectare"}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/recommend_fertilizer")
def recommend_fertilizer_api(lat: float, lon: float, ozone: float, soil: float, ph: float, stage: str):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'result': None}, status_code=400)
    temp = weather['temp']
    rain = weather['rain']
    input_df = pd.DataFrame([{
        "ozone": ozone,
        "temp": temp,
        "rain": rain,
        "soil": soil,
        "ph": ph,
        "stage": stage
    }])
    
    try:
        fert_model = get_fert_model()
        if fert_model is None:
            return JSONResponse({'error': 'Fertilizer model not available'}, status_code=503)
        result = recommend_fertilizer(input_df, fert_model)
        cleanup_after_inference()
        return {"result": f"Recommended Fertilizer: {result}"}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/predict_stress")
def predict_stress(lat: float, lon: float, ozone: float, temp: float, humidity: float, color: str, symptom: str):
    input_df = pd.DataFrame([[ozone, temp, humidity, color, symptom]],
                            columns=["ozone", "temp", "humidity", "color", "symptom"])
    
    try:
        stress_model = get_stress_model()
        if stress_model is None:
            return JSONResponse({'error': 'Stress model not available'}, status_code=503)
        level, explanation = predict_stress_level(stress_model, input_df)
        cleanup_after_inference()
        return {"result": f"Stress Level: {level}", "explanation": explanation}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/recommend_crop")
def recommend_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float, ozone: float):
    features = [[N, P, K, temperature, humidity, ph, rainfall, ozone]]
    try:
        crop_model = get_crop_model()
        if crop_model is None:
            return JSONResponse({'error': 'Crop model not available'}, status_code=503)
        pred = crop_model.predict(features)[0]
        known_crops = set(str(c) for c in crop_model.classes_)
        cleanup_after_inference()
        if str(pred).strip().lower() in (c.strip().lower() for c in known_crops):
            return {"recommended_crop": pred}
        else:
            return {"recommended_crop": None, "message": "No preferred crop available for the given conditions."}
    except Exception as e:
        return {"recommended_crop": None, "message": f"Prediction error: {e}"}
# API Endpoints for Frontend
@app.get("/api/weather")
def get_weather(lat: float, lon: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        raise HTTPException(status_code=500, detail="Weather data unavailable")
    return weather

@app.post("/api/crop/recommend")
def api_recommend_crop(data: dict):
    N = data.get('N', 0)
    P = data.get('P', 0)
    K = data.get('K', 0)
    temperature = data.get('temperature', 0)
    humidity = data.get('humidity', 0)
    ph = data.get('ph', 0)
    rainfall = data.get('rainfall', 0)
    ozone = data.get('ozone', 0)
    
    features = [[N, P, K, temperature, humidity, ph, rainfall, ozone]]
    try:
        crop_model = get_crop_model()
        if crop_model is None:
            raise HTTPException(status_code=503, detail="Crop model not available")
        pred = crop_model.predict(features)[0]
        cleanup_after_inference()
        return {"crop": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/yield/predict")
def api_predict_yield(data: dict):
    # Get user inputs
    crop = data.get('crop', 'potato')
    area = data.get('area', 1)
    soil = data.get('soilMoisture', 0.5)
    ozone = data.get('ozone', 40)
    
    # Weather params - use provided values OR auto-fetch
    temp = data.get('temperature')
    humidity = data.get('humidity')
    rain = data.get('rainfall')
    
    # If weather not provided, auto-fetch using lat/lon
    if temp is None or humidity is None or rain is None:
        lat = data.get('lat', 20.5937)  # Default: India center
        lon = data.get('lon', 78.9629)
        
        weather = fetch_weather_data(lat, lon)
        if not weather:
            raise HTTPException(status_code=400, detail="Weather data unavailable")
        
        temp = weather['temp'] if temp is None else temp
        humidity = weather['humidity'] if humidity is None else humidity
        rain = weather['rain'] if rain is None else rain
    
    # Prepare features for model
    features = pd.DataFrame(
        [[ozone, temp, rain, soil]], 
        columns=["ozone", "temp", "rain", "soil"]
    )
    
    # Predict yield with lazy loading
    try:
        yield_model = get_yield_model()
        if yield_model is None:
            # Fallback calculation if model not available
            fallback_prediction = float(area) * (30 + (temp * 0.5) + (rain * 0.3))
            yield_value = round(max(fallback_prediction, 0.0), 2)
        else:
            prediction = yield_model.predict(features)[0]
            yield_value = round(max(float(prediction), 0.0), 2)
            cleanup_after_inference()
    except Exception as e:
        # Fallback calculation if model fails
        fallback_prediction = float(area) * (30 + (temp * 0.5) + (rain * 0.3))
        yield_value = round(max(fallback_prediction, 0.0), 2)
    
    return {
        "yield": f"{yield_value} tonnes/hectare",
        "value": yield_value,
        "crop": crop,
        "area": area,
        "weather_used": {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rain
        }
    }


# ====================
# NEW: APY-Based Yield Prediction Endpoints
# ====================

@app.post("/predict-yield")
async def predict_yield_apy(request: YieldPredictionRequest):
    """
    NEW: Predict crop yield using APY dataset-trained model
    
    Uses real historical data (State, District, Crop, Year, Season, Area)
    to predict yield with high accuracy
    """
    try:
        service = get_yield_service()
        
        result = service.predict_yield(
            state=request.state,
            district=request.district,
            crop=request.crop,
            year=request.year,
            season=request.season,
            area=request.area
        )
        
        if not result.get('success', False):
            raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/yield/options")
async def get_yield_prediction_options():
    """
    Get available options for yield prediction dropdowns
    
    Returns lists of: States, Districts, Crops, Seasons
    """
    try:
        service = get_yield_service()
        options = service.get_available_values()
        
        return {
            "success": True,
            "states": options.get('State', []),
            "districts": options.get('District', []),
            "crops": options.get('Crop', []),
            "seasons": options.get('Season', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load options: {str(e)}")


@app.get("/yield/states")
async def get_yield_states():
    """
    Get list of unique states for yield prediction
    """
    try:
        service = get_yield_service()
        options = service.get_available_values()
        
        return {
            "success": True,
            "states": options.get('State', [])
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Yield states unavailable",
                "detail": str(e)
            }
        )


@app.get("/yield/districts/{state}")
async def get_yield_districts_by_state(state: str):
    """
    Get districts filtered by selected state
    
    Args:
        state: State name to filter districts
    
    Returns:
        List of districts in the selected state
    """
    try:
        service = get_yield_service()
        districts = service.get_districts_by_state(state)
        
        return {
            "success": True,
            "state": state,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load districts: {str(e)}")


@app.get("/yield/crops/{state}")
async def get_yield_crops_by_state(state: str):
    """
    Get crops filtered by selected state.

    Args:
        state: State name to filter crops

    Returns:
        Unique crops for the selected state sorted alphabetically
    """
    try:
        service = get_yield_service()
        print(f"[API] API request /yield/crops/{{state}} => {state}")
        crops = service.get_crops_by_state(state)
        print(f"[API] API response crop count => {len(crops)}")

        if not crops:
            return {
                "crops": [],
                "message": "No crops found for this state"
            }

        return {
            "crops": crops
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load crops: {str(e)}")


@app.get("/api/yield/model-info")
async def get_yield_model_info():
    """Get information about the yield prediction model"""
    try:
        service = get_yield_service()
        info = service.get_model_info()
        
        return {
            "success": True,
            **info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/api/fertilizer/recommend")
def api_recommend_fertilizer(data: dict):
    """
    ML-based fertilizer recommendation using trained model
    
    Required inputs from frontend:
    - Soil_Type, Soil_pH
    - Nitrogen_Level, Phosphorus_Level, Potassium_Level
    - Crop_Type, Crop_Growth_Stage, Season
    - Temperature, Humidity, Rainfall
    - Irrigation_Type, Previous_Crop, Region
    
    Optional/Hidden inputs (use defaults if not provided):
    - Soil_Moisture (default: 50)
    - Organic_Carbon (default: 0.8)
    - Electrical_Conductivity (default: 1.2)
    """
    try:
        fertilizer_service = get_fertilizer_service()
        
        # Extract all required features from request with defaults for hidden fields
        inputs = {
            'Soil_Type': data.get('Soil_Type'),
            'Soil_pH': float(data.get('Soil_pH')),
            # Hidden fields with defaults
            'Soil_Moisture': float(data.get('Soil_Moisture', 50)),
            'Organic_Carbon': float(data.get('Organic_Carbon', 0.8)),
            'Electrical_Conductivity': float(data.get('Electrical_Conductivity', 1.2)),
            'Nitrogen_Level': float(data.get('Nitrogen_Level')),
            'Phosphorus_Level': float(data.get('Phosphorus_Level')),
            'Potassium_Level': float(data.get('Potassium_Level')),
            'Crop_Type': data.get('Crop_Type'),
            'Crop_Growth_Stage': data.get('Crop_Growth_Stage'),
            'Season': data.get('Season'),
            'Temperature': float(data.get('Temperature')),
            'Humidity': float(data.get('Humidity')),
            'Rainfall': float(data.get('Rainfall')),
            'Irrigation_Type': data.get('Irrigation_Type'),
            'Previous_Crop': data.get('Previous_Crop'),
            'Region': data.get('Region')
        }
        
        # Get prediction
        result = fertilizer_service.predict(inputs)
        
        return {
            "success": True,
            "fertilizer": result['fertilizer'],
            "confidence": result['confidence'],
            "confidence_percentage": result['confidence_percentage'],
            "top_3_recommendations": result['top_3_recommendations'],
            "all_probabilities": result['all_probabilities'],
            "inputs_used": inputs
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/fertilizer/auto-fill")
def fertilizer_auto_fill(payload: FertilizerAutoFillRequest):
    """
    Auto-fill fertilizer soil/nutrient fields from location.

    Returns only the requested fields. On failure, returns all fields as null.
    """
    try:
        if not (-90 <= payload.latitude <= 90) or not (-180 <= payload.longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid latitude or longitude")

        service = get_fertilizer_auto_fill_service()
        return service.get_auto_fill(payload.latitude, payload.longitude)
    except HTTPException:
        raise
    except Exception:
        return {
            "soil_pH": None,
            "soil_moisture": None,
            "organic_carbon": None,
            "electrical_conductivity": None,
            "nitrogen": None,
            "phosphorus": None,
            "potassium": None,
        }


@app.get("/api/fertilizer/options")
def get_fertilizer_options():
    """Get all valid options for categorical features"""
    try:
        fertilizer_service = get_fertilizer_service()
        if not fertilizer_service.encoders:
            if not fertilizer_service.load_model():
                return JSONResponse(
                    status_code=503,
                    content={"status": "error", "message": "Fertilizer model unavailable"}
                )
        options = fertilizer_service.get_feature_options()
        return {
            "success": True,
            "options": options
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Fertilizer model unavailable"}
        )


@app.get("/api/fertilizer/model-info")
def get_fertilizer_model_info():
    """Get fertilizer model information and metrics"""
    try:
        fertilizer_service = get_fertilizer_service()
        if not fertilizer_service.model:
            if not fertilizer_service.load_model():
                return JSONResponse(
                    status_code=503,
                    content={"status": "error", "message": "Fertilizer model unavailable"}
                )
        info = fertilizer_service.get_model_info()
        return {
            "success": True,
            **info
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Fertilizer model unavailable"}
        )


@app.post("/api/fertilizer/location-data")
def get_fertilizer_location_data(data: dict):
    """
    Get location, weather, and SOIL data for fertilizer recommendation based on coordinates
    Uses reverse geocoding, weather APIs, and SoilGrids API for comprehensive data
    """
    try:
        import requests
        from datetime import datetime
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is None or longitude is None:
            raise HTTPException(status_code=400, detail="latitude and longitude are required")
        
        # Validate coordinates
        try:
            lat = float(latitude)
            lng = float(longitude)
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError("Invalid coordinates")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid latitude or longitude values")
        
        result = {
            "success": True,
            "latitude": lat,
            "longitude": lng,
            "region": None,
            "state": None,
            "district": None,
            "temperature": None,
            "humidity": None,
            "rainfall": 0,
            # Soil characteristics
            "soil_pH": None,
            "soil_moisture": None,
            "organic_matter": None,
            "organic_carbon": None,
            "soil_type": None,
            "elevation": None,
            "electrical_conductivity": None
        }
        
        # Fetch soil data from external APIs (SoilGrids, OpenElevation, etc.)
        try:
            print(f"[DATA] Fetching soil data for coordinates: {lat}, {lng}")
            soil_service = get_soil_data_service()
            soil_data = soil_service.get_soil_data(lat, lng)
            
            # Merge soil data into result
            if soil_data:
                for key in ['soil_pH', 'soil_moisture', 'organic_matter', 'organic_carbon', 
                           'soil_type', 'elevation', 'electrical_conductivity']:
                    if soil_data.get(key) is not None:
                        result[key] = soil_data[key]
                
                print(f"[OK] Soil data fetched: pH={result['soil_pH']}, Type={result['soil_type']}, Elevation={result['elevation']}m")
        except Exception as e:
            print(f"G��n+� Failed to fetch soil data: {e}")
            # Continue without soil data - user can enter manually
        
        # 1. Reverse Geocoding using Nominatim (OpenStreetMap)
        try:
            geocode_url = f"https://nominatim.openstreetmap.org/reverse"
            geocode_params = {
                'lat': lat,
                'lon': lng,
                'format': 'json',
                'addressdetails': 1
            }
            geocode_headers = {
                'User-Agent': 'SmartAgri-Fertilizer/1.0'
            }
            
            geocode_response = requests.get(
                geocode_url, 
                params=geocode_params, 
                headers=geocode_headers,
                timeout=5
            )
            
            if geocode_response.status_code == 200:
                geo_data = geocode_response.json()
                address = geo_data.get('address', {})
                
                # Extract state
                state = (address.get('state') or 
                        address.get('ISO3166-2-lvl4', '').split('-')[-1] or 
                        address.get('region'))
                
                # Extract district
                district = (address.get('state_district') or 
                           address.get('county') or 
                           address.get('district'))
                
                result['state'] = state
                result['district'] = district
                
                # Map state to region for fertilizer model
                state_to_region = {
                    # North (including ISO codes)
                    'Punjab': 'North', 'PB': 'North',
                    'Haryana': 'North', 'HR': 'North',
                    'Himachal Pradesh': 'North', 'HP': 'North',
                    'Jammu and Kashmir': 'North', 'JK': 'North',
                    'Delhi': 'North', 'DL': 'North', 'NCT of Delhi': 'North',
                    'Uttarakhand': 'North', 'UT': 'North', 'UK': 'North',
                    'Uttar Pradesh': 'North', 'UP': 'North',
                    'Chandigarh': 'North', 'CH': 'North',
                    
                    # South (including ISO codes)
                    'Tamil Nadu': 'South', 'TN': 'South',
                    'Karnataka': 'South', 'KA': 'South',
                    'Kerala': 'South', 'KL': 'South',
                    'Andhra Pradesh': 'South', 'AP': 'South',
                    'Telangana': 'South', 'TG': 'South', 'TS': 'South',
                    'Puducherry': 'South', 'PY': 'South',
                    
                    # East (including ISO codes)
                    'West Bengal': 'East', 'WB': 'East',
                    'Odisha': 'East', 'OR': 'East', 'OD': 'East',
                    'Bihar': 'East', 'BR': 'East',
                    'Jharkhand': 'East', 'JH': 'East',
                    'Assam': 'East', 'AS': 'East',
                    'Sikkim': 'East', 'SK': 'East',
                    'Arunachal Pradesh': 'East', 'AR': 'East',
                    'Nagaland': 'East', 'NL': 'East',
                    'Manipur': 'East', 'MN': 'East',
                    'Mizoram': 'East', 'MZ': 'East',
                    'Tripura': 'East', 'TR': 'East',
                    'Meghalaya': 'East', 'ML': 'East',
                    
                    # West (including ISO codes)
                    'Maharashtra': 'West', 'MH': 'West',
                    'Gujarat': 'West', 'GJ': 'West',
                    'Goa': 'West', 'GA': 'West',
                    'Rajasthan': 'West', 'RJ': 'West',
                    'Daman and Diu': 'West', 'DD': 'West',
                    
                    # Central (including ISO codes)
                    'Madhya Pradesh': 'Central', 'MP': 'Central',
                    'Chhattisgarh': 'Central', 'CT': 'Central', 'CG': 'Central'
                }
                
                if state:
                    result['region'] = state_to_region.get(state, 'Central')
                
        except requests.Timeout:
            print("[TIMEOUT] Geocoding timeout")
        except Exception as e:
            print(f"[WARN] Geocoding error: {e}")
        
        # 2. Fetch Weather Data using OpenWeatherMap API (if API key available)
        try:
            # Check if OpenWeatherMap API key is available
            import os
            weather_api_key = os.getenv('OPENWEATHER_API_KEY', '90e50f067196b6d46932c52869d83ed6')
            
            print(f"[WEATHER] Attempting to fetch weather data for: {lat}, {lng}")
            print(f"[AUTH] Using API key: {weather_api_key[:10]}...")
            
            if weather_api_key:
                weather_url = "https://api.openweathermap.org/data/2.5/weather"
                weather_params = {
                    'lat': lat,
                    'lon': lng,
                    'appid': weather_api_key,
                    'units': 'metric'
                }
                
                weather_response = requests.get(
                    weather_url,
                    params=weather_params,
                    timeout=10
                )
                
                print(f"[API] Weather API Response Status: {weather_response.status_code}")
                
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    
                    # Extract temperature
                    result['temperature'] = weather_data.get('main', {}).get('temp')
                    
                    # Extract humidity
                    result['humidity'] = weather_data.get('main', {}).get('humidity')
                    
                    # Extract rainfall (if available in last hour)
                    rain_data = weather_data.get('rain', {})
                    result['rainfall'] = rain_data.get('1h', 0) or rain_data.get('3h', 0) or 0
                    
                    print(f"[OK] Weather data fetched: Temp={result['temperature']}-�C, Humidity={result['humidity']}%, Rainfall={result['rainfall']}mm")
                else:
                    print(f"[ERROR] Weather API failed: Status {weather_response.status_code}")
                    print(f"Response: {weather_response.text[:200]}")
                    
        except requests.Timeout:
            print("[TIMEOUT] Weather API timeout - request took longer than 10 seconds")
        except Exception as e:
            print(f"[WARN] Weather API error: {e}")
            import traceback
            traceback.print_exc()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get location data: {str(e)}")


# ====================================================================
# STRESS PREDICTION API - Simplified Farmer-Friendly Model
# ====================================================================

@app.get("/api/stress/options")
def get_stress_options():
    """Get dropdown options for stress prediction form"""
    try:
        options = stress_service.get_options()
        return {
            "success": True,
            "options": options
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/stress/predict")
def api_predict_stress(data: dict):
    """Predict crop stress level using ML model with AI-powered explanations"""
    try:
        # Auto-fetch weather data if location provided and weather not included
        if 'lat' in data and 'lng' in data:
            if 'temperature' not in data or data.get('temperature') is None:
                lat = data['lat']
                lng = data['lng']
                weather = fetch_weather_data(lat, lng)
                
                if weather:
                    data.setdefault('temperature', weather.get('temp', 25))
                    data.setdefault('humidity', weather.get('humidity', 60))
                    data.setdefault('rainfall', weather.get('rain', 50))
                    data.setdefault('wind_speed', weather.get('wind', 10))
        
        # Step 1: Get ML model prediction
        print("[ML] Running ML model prediction...")
        ml_prediction = stress_service.predict(data)
        
        # Check if ML prediction was successful
        if not ml_prediction.get('success', False):
            return ml_prediction
        
        # Step 2: Generate AI-powered explanation and recommendations
        print("=��� Generating AI insights using Groq LLM...")
        try:
            ai_insights = generate_stress_insights(data, ml_prediction)
            
            # Merge AI insights with ML prediction
            result = {
                **ml_prediction,
                "ai_explanation": ai_insights.get('explanation', ''),
                "ai_recommendations": ai_insights.get('recommendations', []),
                "reasoning_source": ai_insights.get('reasoning_source', ''),
                "enhanced_with_ai": True
            }
        except Exception as e:
            print(f"[WARN] AI explanation generation failed: {e}")
            # Return ML prediction without AI enhancement if LLM fails
            result = {
                **ml_prediction,
                "ai_explanation": "",
                "ai_recommendations": [],
                "reasoning_source": "ML Model Only",
                "enhanced_with_ai": False
            }
        
        print("[OK] Stress prediction with AI insights complete")
        return result
        
    except Exception as e:
        print(f"[ERROR] Stress prediction error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/stress/location-data")
async def get_stress_location_data(data: dict):
    """
    Fetch location, weather, and SOIL data for stress prediction
    Enhanced with soil data from external APIs
    """
    try:
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not latitude or not longitude:
            return {
                "success": False,
                "error": "Latitude and longitude are required"
            }
        
        lat = float(latitude)
        lng = float(longitude)
        
        result = {
            "success": True,
            "latitude": lat,
            "longitude": lng,
            "temperature": None,
            "humidity": None,
            "rainfall": None,
            "wind_speed": None,
            "elevation": None,
            "water_flow": 50,  # Default, would need sensor data
            "drainage": 70,    # Default, would need sensor data
            # Soil data
            "soil_pH": None,
            "soil_moisture": None,
            "organic_matter": None
        }
        
        # Fetch soil data from external APIs
        try:
            print(f"[DATA] Fetching soil data for stress prediction: {lat}, {lng}")
            soil_service = get_soil_data_service()
            soil_data = soil_service.get_soil_data(lat, lng)
            
            if soil_data:
                result["soil_pH"] = soil_data.get("soil_pH")
                result["soil_moisture"] = soil_data.get("soil_moisture")
                result["organic_matter"] = soil_data.get("organic_matter")
                result["elevation"] = soil_data.get("elevation")
                
                print(f"[OK] Soil data for stress: pH={result['soil_pH']}, Moisture={result['soil_moisture']}")
        except Exception as e:
            print(f"[WARN] Failed to fetch soil data for stress: {e}")
        
        # Fetch weather data
        weather = fetch_weather_data(lat, lng)
        
        if weather:
            result["temperature"] = weather.get('temp', 25)
            result["humidity"] = weather.get('humidity', 60)
            result["rainfall"] = weather.get('rain', 50)
            result["wind_speed"] = weather.get('wind', 10)
        
        # If elevation not from soil data, use default
        if result["elevation"] is None:
            result["elevation"] = 500
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Location data error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/spray/recommend")
def api_recommend_spray_time(data: SprayRequest):
    temperature = data.temperature
    humidity = data.humidity
    windSpeed = data.windSpeed
    rainfall = data.rainfall
    timeOfDay = data.timeOfDay
    
    issues = []
    
    if temperature > 30:
        issues.append("Temperature too high (>30-�C)")
    if temperature < 10:
        issues.append("Temperature too low (<10-�C)")
    if humidity < 50:
        issues.append("Humidity too low (<50%)")
    if windSpeed > 15:
        issues.append("Wind speed too high (>15 km/h)")
    if rainfall > 0:
        issues.append("Rain expected")
    
    # Determine best time
    best_time = "Early morning (6-8 AM) or late evening (5-7 PM)"
    if not issues and timeOfDay:
        best_time = timeOfDay
    
    if issues:
        return {
            "is_safe": False,
            "recommendation": "Not recommended - wait for better conditions",
            "best_time": best_time,
            "factors": {
                "wind": "Too high" if windSpeed > 15 else "Favorable",
                "temperature": "Too high" if temperature > 30 else "Too low" if temperature < 10 else "Optimal",
                "rainfall": "Rain expected" if rainfall > 0 else "No rain"
            }
        }
    else:
        return {
            "is_safe": True,
            "recommendation": "Safe to spray - conditions are favorable",
            "best_time": best_time,
            "factors": {
                "wind": "Favorable",
                "temperature": "Optimal",
                "rainfall": "No rain"
            }
        }

@app.post("/api/disease/fruit")
def api_detect_fruit_disease(data: dict):
    # Mock response - in production, process the image
    return {
        "disease": "Healthy",
        "confidence": 95.5,
        "treatment": "No treatment needed. Continue regular monitoring."
    }

@app.post("/api/disease/leaf")
def api_detect_leaf_disease(data: dict):
    # Mock response - in production, process the image
    return {
        "disease": "Healthy",
        "confidence": 92.3,
        "treatment": "No treatment needed. Maintain current care practices."
    }

@app.post("/api/chatbot")
def api_chatbot(data: dict):
    message = data.get('message', '').lower()
    
    # Simple chatbot responses
    if 'weather' in message:
        return {"response": "You can check real-time weather data on the Dashboard. Click on the map to select your location."}
    elif 'crop' in message or 'recommend' in message:
        return {"response": "Use the Crop Recommendation module to get AI-based crop suggestions based on soil and weather conditions."}
    elif 'yield' in message:
        return {"response": "The Yield Prediction module helps estimate your potato crop yield based on environmental factors."}
    elif 'fertilizer' in message:
        return {"response": "The Fertilizer Recommendation module suggests optimal fertilizer types based on your soil nutrient levels."}
    elif 'stress' in message:
        return {"response": "The Stress Prediction module monitors environmental factors that may stress your crops."}
    elif 'spray' in message:
        return {"response": "The Best Time to Spray module analyzes weather conditions to recommend optimal spraying times."}
    elif 'disease' in message:
        return {"response": "Use our Disease Detection modules to identify fruit and leaf diseases by uploading images."}
    elif 'hello' in message or 'hi' in message:
        return {"response": "Hello! I'm your Smart Agri AI assistant. How can I help you today?"}
    else:
        return {"response": "I can help you with weather data, crop recommendations, yield predictions, fertilizer suggestions, stress monitoring, spray timing, and disease detection. What would you like to know?"}


if __name__ == "__main__":
    import uvicorn
    # dotenv already loaded at top of file
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
