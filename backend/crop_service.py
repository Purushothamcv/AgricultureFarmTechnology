"""
Crop Recommendation Service
Handles prediction logic, weather API calls, and soil data fetching
"""

import numpy as np
import requests
from typing import Dict, Any, Optional, Tuple
import joblib


# Load the trained crop recommendation model
try:
    crop_model = joblib.load("model/crop_model.pkl")
    print("✅ Crop model loaded successfully")
except Exception as e:
    print(f"❌ Error loading crop model: {e}")
    crop_model = None


def predict_crop(nitrogen: float, phosphorus: float, potassium: float,
                temperature: float, humidity: float, ph: float, rainfall: float, ozone: float) -> Tuple[str, Optional[float]]:
    """
    Predict crop based on input parameters
    
    Args:
        nitrogen: Nitrogen content (kg/ha)
        phosphorus: Phosphorus content (kg/ha)
        potassium: Potassium content (kg/ha)
        temperature: Temperature (°C)
        humidity: Humidity (%)
        ph: pH value
        rainfall: Rainfall (mm)
        ozone: Ozone level (ppb)
    
    Returns:
        Tuple of (predicted_crop, confidence_score)
    """
    if crop_model is None:
        raise ValueError("Crop model not loaded")
    
    # Prepare input features in the same order as training
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, ozone]])
    
    # Make prediction
    prediction = crop_model.predict(features)[0]
    
    # Get confidence score if available
    confidence = None
    if hasattr(crop_model, 'predict_proba'):
        try:
            proba = crop_model.predict_proba(features)[0]
            confidence = float(max(proba))
        except:
            pass
    
    return str(prediction), confidence


async def fetch_weather_data(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Fetch weather data from Open-Meteo API
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Dictionary with temperature, humidity, and rainfall data
    """
    try:
        # Open-Meteo API (free, no API key required)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "precipitation_sum",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract current weather
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        temperature = current.get("temperature_2m", 25.0)
        humidity = current.get("relative_humidity_2m", 70.0)
        
        # Use daily precipitation sum (average of next 7 days)
        rainfall_data = daily.get("precipitation_sum", [])
        rainfall = sum(rainfall_data[:7]) / 7 if rainfall_data else 100.0
        
        return {
            "success": True,
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "rainfall": round(rainfall, 2)
        }
    
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        # Return default values if API fails
        return {
            "success": False,
            "temperature": 25.0,
            "humidity": 70.0,
            "rainfall": 100.0,
            "error": str(e)
        }


def get_soil_data_by_region(latitude: float, longitude: float) -> Dict[str, float]:
    """
    Get soil data based on geographic region
    Uses static mapping based on latitude/longitude ranges
    
    In production, you could integrate with:
    - SoilGrids API (https://rest.isric.org/)
    - FAO Soil Portal
    - Local agricultural databases
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Dictionary with nitrogen, phosphorus, potassium, and pH values
    """
    
    # Default soil values
    default_soil = {
        "nitrogen": 50.0,
        "phosphorus": 50.0,
        "potassium": 50.0,
        "ph": 6.5,
        "ozone": 30.0
    }
    
    # Regional soil mappings (simplified example)
    # India regions
    if 8.0 <= latitude <= 35.0 and 68.0 <= longitude <= 97.0:
        # North India (fertile plains)
        if latitude > 25.0:
            return {
                "nitrogen": 90.0,
                "phosphorus": 60.0,
                "potassium": 70.0,
                "ph": 7.0,
                "ozone": 28.0
            }
        # South India (varied terrain)
        else:
            return {
                "nitrogen": 70.0,
                "phosphorus": 50.0,
                "potassium": 60.0,
                "ph": 6.5,
                "ozone": 32.0
            }
    
    # USA regions
    elif 25.0 <= latitude <= 49.0 and -125.0 <= longitude <= -65.0:
        return {
            "nitrogen": 80.0,
            "phosphorus": 55.0,
            "potassium": 65.0,
            "ph": 6.8,
            "ozone": 35.0
        }
    
    # Europe
    elif 35.0 <= latitude <= 70.0 and -10.0 <= longitude <= 40.0:
        return {
            "nitrogen": 75.0,
            "phosphorus": 52.0,
            "potassium": 68.0,
            "ph": 6.7,
            "ozone": 33.0
        }
    
    # Africa
    elif -35.0 <= latitude <= 37.0 and -20.0 <= longitude <= 52.0:
        return {
            "nitrogen": 45.0,
            "phosphorus": 35.0,
            "potassium": 50.0,
            "ph": 6.0,
            "ozone": 29.0
        }
    
    # South America
    elif -56.0 <= latitude <= 13.0 and -82.0 <= longitude <= -34.0:
        return {
            "nitrogen": 65.0,
            "phosphorus": 45.0,
            "potassium": 55.0,
            "ph": 6.3,
            "ozone": 31.0
        }
    
    # Australia/Oceania
    elif -45.0 <= latitude <= -10.0 and 110.0 <= longitude <= 180.0:
        return {
            "nitrogen": 55.0,
            "phosphorus": 40.0,
            "potassium": 52.0,
            "ph": 6.2,
            "ozone": 34.0
        }
    
    # Default for other regions
    return default_soil


async def fetch_all_location_data(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Fetch both weather and soil data for a given location
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Dictionary with all required parameters for crop prediction
    """
    # Fetch weather data
    weather_data = await fetch_weather_data(latitude, longitude)
    
    # Get soil data
    soil_data = get_soil_data_by_region(latitude, longitude)
    
    # Combine data
    location_data = {
        "success": weather_data.get("success", True),
        "latitude": latitude,
        "longitude": longitude,
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "rainfall": weather_data["rainfall"],
        "nitrogen": soil_data["nitrogen"],
        "phosphorus": soil_data["phosphorus"],
        "potassium": soil_data["potassium"],
        "ph": soil_data["ph"],
        "ozone": soil_data["ozone"],
        "message": "Location data fetched successfully"
    }
    
    if not weather_data.get("success"):
        location_data["message"] = "Weather API unavailable, using default values"
    
    return location_data
