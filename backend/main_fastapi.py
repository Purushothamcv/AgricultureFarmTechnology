from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level
from auth import router as auth_router
from database import connect_to_mongodb, close_mongodb_connection
from db_helpers import get_database_stats
from crop_models import ManualCropInput, LocationCropInput, CropPredictionResponse, LocationDataResponse
from crop_service import predict_crop, fetch_all_location_data
from fruit_disease_service import router as fruit_disease_router, startup_event as fruit_startup
# Import PRODUCTION fruit disease API (frozen model, inference-only)
from api_fruit_disease_production import router as fruit_disease_prod_router, startup_event as fruit_prod_startup
# Import NEW V2 API (clean trained model - 92%+ accuracy)
from fruit_disease_api_v2 import router as fruit_disease_v2_router, startup_event as fruit_v2_startup
# Import Plant Leaf Disease Detection Service
from plant_disease_service import router as plant_disease_router, startup_event as plant_disease_startup
# Import AI Chatbot Service
from chatbot_service import router as chatbot_router, startup_event as chatbot_startup
# Import Yield Prediction Service (APY Dataset-based)
from yield_prediction_service import get_yield_service, startup_event as yield_startup

app = FastAPI(title="SmartAgri API", description="Smart Agriculture Decision Support System", version="1.0.0")

# Event handlers for MongoDB connection
@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection and ML models on application startup"""
    print("🚀 Starting SmartAgri API...")
    await connect_to_mongodb()
    # Initialize fruit disease detection model (legacy)
    await fruit_startup()
    # Initialize PRODUCTION fruit disease model (frozen, inference-only)
    print("🔬 Initializing Production Fruit Disease Detection...")
    await fruit_prod_startup()
    # Initialize NEW V2 fruit disease model (clean trained - 92%+)
    print("🍎 Initializing Fruit Disease V2 (Clean Model)...")
    await fruit_v2_startup()
    # Initialize Plant Leaf Disease Detection
    print("🌿 Initializing Plant Leaf Disease Detection...")
    await plant_disease_startup()
    # Initialize AI Chatbot Service
    print("🤖 Initializing AI Chatbot Service...")
    await chatbot_startup()
    # Initialize Yield Prediction Service
    await yield_startup()
    print("✅ All services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on application shutdown"""
    print("🛑 Shutting down SmartAgri API...")
    await close_mongodb_connection()

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "https://agriculture-farm-technology.vercel.app",  # Your Vercel frontend
    "https://*.vercel.app"  # Allow all Vercel preview deployments
]

print(f"🌐 CORS enabled for origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(auth_router)
app.include_router(fruit_disease_router)  # Legacy endpoint
app.include_router(chatbot_router)  # AI Chatbot with voice assistance
app.include_router(fruit_disease_prod_router)  # PRODUCTION endpoint (frozen model)
app.include_router(fruit_disease_v2_router)  # V2 endpoint (NEW clean trained model - 92%+)
app.include_router(plant_disease_router)  # Plant Leaf Disease Detection

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

# Load ML models
yield_model = joblib.load("model/yield_model.pkl")
time_model = joblib.load("model/best_window_model.pkl")
fert_model = joblib.load("model/fert_model.pkl")
stress_model = joblib.load("model/stress_model.pkl")
crop_model = joblib.load("model/crop_model.pkl")

# ====================
# Main Application Routes
# ====================
# Note: Authentication routes (/auth/register, /auth/login) are now handled 
# by the auth.py module and automatically included via app.include_router(auth_router)

@app.get("/")
async def root():
    """Root endpoint - health check"""
    try:
        db_status = "connected" if database and database.client else "disconnected"
    except:
        db_status = "unknown"
    
    return {
        "status": "ok",
        "message": "SmartAgri API is running",
        "version": "1.0.0",
        "database": db_status
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    db_status = "connected" if database else "disconnected"
    return {
        "status": "healthy" if database else "degraded",
        "database": db_status,
        "api": "ok"
    }

@app.get("/api/database/stats")
async def get_db_stats():
    """Get FinalProject database statistics"""
    stats = await get_database_stats()
    return stats

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
    Crop recommendation based on location (map selection)
    
    Fetches weather and soil data based on coordinates
    User can override auto-fetched values
    """
    try:
        # Fetch location data if not all values are provided
        if any(v is None for v in [
            input_data.nitrogen, input_data.phosphorus, input_data.potassium,
            input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall, input_data.ozone
        ]):
            location_data = await fetch_all_location_data(
                input_data.latitude,
                input_data.longitude
            )
        else:
            location_data = {}
        
        # Use provided values or fallback to fetched values
        nitrogen = input_data.nitrogen if input_data.nitrogen is not None else location_data.get("nitrogen", 50)
        phosphorus = input_data.phosphorus if input_data.phosphorus is not None else location_data.get("phosphorus", 50)
        potassium = input_data.potassium if input_data.potassium is not None else location_data.get("potassium", 50)
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
            message="Crop recommendation generated successfully from location"
        )
    
    except Exception as e:
        return CropPredictionResponse(
            success=False,
            crop="Unknown",
            input_values={"latitude": input_data.latitude, "longitude": input_data.longitude},
            message=f"Prediction failed: {str(e)}"
        )


@app.get("/api/location-data", response_model=LocationDataResponse)
async def get_location_data(latitude: float, longitude: float):
    """
    Fetch weather and soil data for a given location
    
    Used to auto-populate form fields when user selects location on map
    """
    try:
        data = await fetch_all_location_data(latitude, longitude)
        return LocationDataResponse(**data)
    
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
    prediction = yield_model.predict(features)[0]
    return {"result": f"Predicted Potato Yield: {prediction:.2f} tonnes/hectare"}

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
    result = recommend_fertilizer(input_df, fert_model)
    return {"result": f"Recommended Fertilizer: {result}"}

@app.get("/predict_stress")
def predict_stress(lat: float, lon: float, ozone: float, temp: float, humidity: float, color: str, symptom: str):
    input_df = pd.DataFrame([[ozone, temp, humidity, color, symptom]],
                            columns=["ozone", "temp", "humidity", "color", "symptom"])
    level, explanation = predict_stress_level(stress_model, input_df)
    return {"result": f"Stress Level: {level}", "explanation": explanation}

@app.get("/recommend_crop")
def recommend_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float, ozone: float):
    features = [[N, P, K, temperature, humidity, ph, rainfall, ozone]]
    try:
        pred = crop_model.predict(features)[0]
        known_crops = set(str(c) for c in crop_model.classes_)
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
        pred = crop_model.predict(features)[0]
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
    
    # Predict yield
    try:
        prediction = yield_model.predict(features)[0]
        yield_value = round(float(prediction), 2)
    except Exception as e:
        # Fallback calculation if model fails
        yield_value = round(float(area) * (30 + (temp * 0.5) + (rain * 0.3)), 2)
    
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
        raise HTTPException(status_code=500, detail=f"Failed to load states: {str(e)}")


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
    # Get user inputs
    N = data.get('N', 0)
    P = data.get('P', 0)
    K = data.get('K', 0)
    crop = data.get('crop', 'potato')
    soilMoisture = data.get('soilMoisture', 0.5)
    
    # Weather params - optional, auto-fetch if not provided
    temp = data.get('temperature')
    humidity = data.get('humidity')
    rainfall = data.get('rainfall')
    
    # If weather not provided, auto-fetch
    if temp is None or humidity is None or rainfall is None:
        lat = data.get('lat', 20.5937)
        lon = data.get('lon', 78.9629)
        
        weather = fetch_weather_data(lat, lon)
        if weather:
            temp = weather['temp'] if temp is None else temp
            humidity = weather['humidity'] if humidity is None else humidity
            rainfall = weather['rain'] if rainfall is None else rainfall
        else:
            temp = temp or 25
            humidity = humidity or 60
            rainfall = rainfall or 0
    
    # Enhanced fertilizer recommendation logic
    fertilizers = []
    recommendations = []
    
    # NPK-based recommendations
    if N < 50:
        fertilizers.append("Urea (Nitrogen)")
        recommendations.append(f"Apply 50-100 kg/ha Urea to increase Nitrogen (Current: {N})")
    elif N > 100:
        recommendations.append(f"Nitrogen levels are high ({N}). Reduce nitrogen fertilizer use.")
    
    if P < 30:
        fertilizers.append("DAP (Phosphorus)")
        recommendations.append(f"Apply 40-60 kg/ha DAP to increase Phosphorus (Current: {P})")
    elif P > 80:
        recommendations.append(f"Phosphorus levels are sufficient ({P}). Maintain current practices.")
    
    if K < 40:
        fertilizers.append("MOP (Potassium)")
        recommendations.append(f"Apply 30-50 kg/ha MOP to increase Potassium (Current: {K})")
    elif K > 100:
        recommendations.append(f"Potassium levels are high ({K}). No additional potash needed.")
    
    # Weather-based adjustments
    if rainfall > 100:
        recommendations.append("⚠️ High rainfall: Apply fertilizer in split doses to prevent leaching")
    if temp > 35:
        recommendations.append("⚠️ High temperature: Consider foliar application for better absorption")
    if soilMoisture < 0.3:
        recommendations.append("⚠️ Low soil moisture: Irrigate before fertilizer application")
    
    if not fertilizers:
        fertilizers.append("Balanced NPK (19-19-19)")
        recommendations.append("Soil nutrient levels are balanced. Use maintenance dose of NPK.")
    
    return {
        "fertilizer": ", ".join(fertilizers),
        "recommendations": recommendations,
        "npk_status": {
            "nitrogen": N,
            "phosphorus": P,
            "potassium": K
        },
        "weather_used": {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "soilMoisture": soilMoisture
        }
    }

@app.post("/api/stress/predict")
def api_predict_stress(data: dict):
    # Get user inputs
    soilMoisture = data.get('soilMoisture', 0.5)
    ozone = data.get('ozone', 40)
    
    # Weather params - use provided values OR auto-fetch
    temp = data.get('temperature')
    humidity = data.get('humidity')
    rainfall = data.get('rainfall')
    windSpeed = data.get('windSpeed')
    
    # If weather not provided, auto-fetch using lat/lon
    if temp is None or humidity is None or rainfall is None or windSpeed is None:
        lat = data.get('lat', 20.5937)
        lon = data.get('lon', 78.9629)
        
        weather = fetch_weather_data(lat, lon)
        if weather:
            temp = weather['temp'] if temp is None else temp
            humidity = weather['humidity'] if humidity is None else humidity
            rainfall = weather['rain'] if rainfall is None else rainfall
            windSpeed = weather['wind'] if windSpeed is None else windSpeed
        else:
            # Use defaults if weather fetch fails
            temp = temp or 25
            humidity = humidity or 60
            rainfall = rainfall or 0
            windSpeed = windSpeed or 10
    
    # Simple stress level calculation
    stress_score = 0
    factors = []
    
    if temp > 35 or temp < 10:
        stress_score += 2
        factors.append("Extreme temperature")
    if humidity < 30 or humidity > 90:
        stress_score += 1
        factors.append("Humidity stress")
    if soilMoisture < 0.2:
        stress_score += 2
        factors.append("Low soil moisture")
    if rainfall > 100:
        stress_score += 1
        factors.append("Excessive rainfall")
    if windSpeed > 40:
        stress_score += 1
        factors.append("High wind speed")
    if ozone > 80:
        stress_score += 1
        factors.append("High ozone levels")
    
    if stress_score >= 4:
        level = "High"
    elif stress_score >= 2:
        level = "Moderate"
    else:
        level = "Low"
    
    return {
        "level": level,
        "factors": factors if factors else ["Optimal conditions"],
        "score": stress_score,
        "weather_used": {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "windSpeed": windSpeed,
            "soilMoisture": soilMoisture,
            "ozone": ozone
        }
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
        issues.append("Temperature too high (>30°C)")
    if temperature < 10:
        issues.append("Temperature too low (<10°C)")
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