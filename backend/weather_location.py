"""
Weather and Location Data API Endpoints
========================================

Provides endpoints for:
- /api/weather: Fetch weather data for a given location (lat, lon)
- /api/location-data: Fetch location and soil data for a given location (latitude, longitude)

These endpoints use the existing crop_service functions to fetch real-time weather
and location-based soil data.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from crop_service import fetch_weather_data, fetch_all_location_data
import traceback

# Create router with /api prefix
router = APIRouter(prefix="/api", tags=["weather-location"])


@router.get("/weather")
async def get_weather(lat: float = Query(..., description="Latitude"), lon: float = Query(..., description="Longitude")) -> Dict[str, Any]:
    """
    Fetch weather data for a given location
    
    Query Parameters:
    - lat: Latitude coordinate
    - lon: Longitude coordinate
    
    Returns:
    - temperature: Temperature in Celsius
    - humidity: Relative humidity percentage
    - rainfall: Average rainfall in mm
    - success: Whether the API call was successful
    """
    try:
        if lat is None or lon is None:
            raise ValueError("Latitude and longitude are required")
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid latitude or longitude values")
        
        # Fetch weather data
        weather_data = await fetch_weather_data(lat, lon)
        
        return {
            "success": weather_data.get("success", True),
            "temperature": weather_data.get("temperature", 25.0),
            "humidity": weather_data.get("humidity", 70.0),
            "rainfall": weather_data.get("rainfall", 100.0),
            "wind_speed": weather_data.get("wind_speed", 5.6),
            "pressure": weather_data.get("pressure", 1012),
            "message": "Weather data fetched successfully"
        }
    except ValueError as e:
        return {
            "success": False,
            "temperature": 0,
            "humidity": 0,
            "rainfall": 0,
            "wind_speed": 0,
            "pressure": 0,
            "error": str(e),
            "message": "Invalid input parameters"
        }
    except Exception as e:
        print(f"[ERROR] /api/weather error: {str(e)}")
        traceback.print_exc()
        # Return fallback data instead of crashing
        return {
            "success": False,
            "temperature": 25.0,
            "humidity": 70.0,
            "rainfall": 100.0,
            "wind_speed": 5.6,
            "pressure": 1012,
            "error": str(e),
            "message": "Unable to fetch real-time weather data, using default values"
        }


@router.get("/location-data")
async def get_location_data(latitude: float = Query(..., description="Latitude"), longitude: float = Query(..., description="Longitude")) -> Dict[str, Any]:
    """
    Fetch location and soil data for a given latitude and longitude
    
    Query Parameters:
    - latitude: Latitude coordinate
    - longitude: Longitude coordinate
    
    Returns:
    - temperature: Temperature in Celsius
    - humidity: Relative humidity percentage
    - rainfall: Average rainfall in mm
    - nitrogen: Soil nitrogen level
    - phosphorus: Soil phosphorus level
    - potassium: Soil potassium level
    - ph: Soil pH value
    - state: State/Province name (inferred from coordinates)
    - district: District name (inferred from coordinates)
    - soil_type: Type of soil in the region
    - elevation: Elevation in meters
    """
    try:
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude are required")
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise ValueError("Invalid latitude or longitude values")
        
        # Fetch all location data (weather + soil)
        location_data = await fetch_all_location_data(latitude, longitude)
        
        # Determine state, district, soil_type, elevation based on coordinates
        # This is a simplified mapping for Indian states
        state = get_state_from_coordinates(latitude, longitude)
        district = get_district_from_coordinates(latitude, longitude)
        soil_type = get_soil_type(latitude, longitude)
        elevation = get_elevation(latitude, longitude)
        
        return {
            "success": location_data.get("success", True),
            "latitude": latitude,
            "longitude": longitude,
            "temperature": location_data.get("temperature", 25.0),
            "humidity": location_data.get("humidity", 70.0),
            "rainfall": location_data.get("rainfall", 100.0),
            "nitrogen": location_data.get("nitrogen", 45),
            "phosphorus": location_data.get("phosphorus", 35),
            "potassium": location_data.get("potassium", 180),
            "ph": location_data.get("ph", 6.5),
            "state": state,
            "district": district,
            "soil_type": soil_type,
            "elevation": elevation,
            "message": "Location data fetched successfully"
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Invalid input parameters"
        }
    except Exception as e:
        print(f"[ERROR] /api/location-data error: {str(e)}")
        traceback.print_exc()
        # Return fallback data instead of crashing
        return {
            "success": False,
            "state": "Unknown",
            "district": "Unknown",
            "soil_type": "Loamy",
            "elevation": 0,
            "error": str(e),
            "message": "Unable to fetch location data, using default values"
        }


def get_state_from_coordinates(lat: float, lon: float) -> str:
    """
    Determine Indian state from latitude and longitude
    Uses geographic boundaries of Indian states
    """
    # North India
    if lat >= 32 and lon >= 70 and lon <= 78:
        return "Himachal Pradesh"
    if lat >= 30 and lat < 32 and lon >= 75 and lon <= 80:
        return "Punjab"
    if lat >= 27 and lat < 30 and lon >= 75 and lon <= 78:
        return "Haryana"
    if lat >= 26 and lat < 31 and lon >= 77 and lon <= 84:
        return "Uttar Pradesh"
    
    # South India
    if lat >= 11 and lat < 17 and lon >= 74 and lon <= 81:
        return "Karnataka"
    if lat >= 8 and lat < 13 and lon >= 74 and lon <= 79:
        return "Tamil Nadu"
    if lat >= 8 and lat < 14 and lon >= 76 and lon <= 82:
        return "Telangana"
    if lat >= 13 and lat < 19 and lon >= 72 and lon <= 80:
        return "Maharashtra"
    if lat >= 10 and lat < 13 and lon >= 74 and lon <= 77:
        return "Kerala"
    
    # Central India
    if lat >= 20 and lat < 25 and lon >= 77 and lon <= 84:
        return "Madhya Pradesh"
    if lat >= 21 and lat < 26 and lon >= 73 and lon <= 81:
        return "Rajasthan"
    if lat >= 19 and lat < 24 and lon >= 79 and lon <= 86:
        return "Odisha"
    
    # Northeast India
    if lat >= 26 and lat < 29 and lon >= 88 and lon <= 92:
        return "Assam"
    
    # Western India
    if lat >= 20 and lat < 24 and lon >= 68 and lon <= 74:
        return "Gujarat"
    if lat >= 16 and lat < 19 and lon >= 72 and lon <= 76:
        return "Goa"
    
    # Default
    return "India"


def get_district_from_coordinates(lat: float, lon: float) -> str:
    """
    Determine Indian district from latitude and longitude
    Uses geographic boundaries of major Indian districts
    """
    # Karnataka districts
    if lat >= 13 and lat < 15 and lon >= 75 and lon <= 77:
        return "Mysuru"
    if lat >= 12 and lat < 14 and lon >= 77 and lon <= 78:
        return "Bengaluru"
    if lat >= 14 and lat < 16 and lon >= 74 and lon <= 76:
        return "Belagavi"
    
    # Maharashtra districts
    if lat >= 18 and lat < 20 and lon >= 72 and lon <= 74:
        return "Mumbai"
    if lat >= 19 and lat < 21 and lon >= 75 and lon <= 77:
        return "Nagpur"
    
    # Uttar Pradesh districts
    if lat >= 26 and lat < 28 and lon >= 77 and lon <= 79:
        return "Delhi"
    if lat >= 25 and lat < 27 and lon >= 80 and lon <= 82:
        return "Lucknow"
    
    # Tamil Nadu districts
    if lat >= 10 and lat < 12 and lon >= 78 and lon <= 80:
        return "Chennai"
    if lat >= 9 and lat < 11 and lon >= 77 and lon <= 79:
        return "Madurai"
    
    # Default
    return "Unknown"


def get_soil_type(lat: float, lon: float) -> str:
    """
    Determine soil type based on location
    Uses simplified geographic boundaries
    """
    # Loamy soil regions (most of India)
    if lat >= 12 and lat < 20:
        return "Loamy"
    
    # Black soil regions (Deccan)
    if lat >= 16 and lat < 19 and lon >= 72 and lon <= 80:
        return "Black Soil"
    
    # Alluvial soil regions (Indo-Gangetic plains)
    if lat >= 24 and lat < 32 and lon >= 75 and lon <= 90:
        return "Alluvial"
    
    # Red soil regions (South India)
    if lat >= 10 and lat < 16:
        return "Red Soil"
    
    # Default
    return "Loamy"


def get_elevation(lat: float, lon: float) -> int:
    """
    Estimate elevation based on location
    Uses simplified geographic data
    """
    # Himalayas
    if lat >= 30 and lon >= 75 and lon <= 90:
        return 1500
    
    # Western Ghats (high elevation)
    if lon >= 74 and lon <= 77 and lat >= 8 and lat < 16:
        return 1200
    
    # Deccan plateau
    if lat >= 12 and lat < 20 and lon >= 74 and lon <= 82:
        return 700
    
    # Eastern Ghats
    if lon >= 80 and lon <= 82 and lat >= 10 and lat < 20:
        return 600
    
    # Plains (very low elevation)
    if lat >= 24 and lat < 32:
        return 100
    
    # Coastal regions
    if lon >= 68 and lon <= 74 and lat >= 8 and lat < 20:
        return 50
    
    # Default
    return 500
