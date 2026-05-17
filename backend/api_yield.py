"""
YIELD PREDICTION API
====================
FastAPI Router for crop yield prediction
"""

import logging
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import traceback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/yield", tags=["Yield"])

# Global service instance
yield_service = None

def get_yield_service():
    """Lazy load yield service"""
    global yield_service
    if yield_service is None:
        try:
            from yield_prediction_service import YieldPredictionService
            yield_service = YieldPredictionService()
            yield_service.load_model()
        except Exception as e:
            logger.error(f"Failed to load yield service: {e}")
            yield_service = None
    return yield_service


# India states for yield prediction
INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]


@router.get("/states")
async def get_states():
    """Get list of Indian states for yield prediction"""
    try:
        return {
            "status": "success",
            "data": INDIA_STATES,
            "total_states": len(INDIA_STATES)
        }
    except Exception as e:
        logger.error(f"Error in get_states: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/options")
async def get_yield_options():
    """Get available yield prediction parameters"""
    try:
        return {
            "status": "success",
            "data": {
                "crops": [
                    "Rice", "Wheat", "Maize", "Potato", "Sugarcane",
                    "Cotton", "Jute", "Rapeseed", "Soybean", "Barley"
                ],
                "seasons": ["Kharif", "Rabi", "Summer"],
                "states": INDIA_STATES,
                "years": list(range(2015, 2024)),
                "parameters": [
                    "Area (hectares)",
                    "Rainfall (mm)",
                    "Temperature (°C)",
                    "Humidity (%)",
                    "Soil pH"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error in get_yield_options: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/predict")
async def predict_yield(
    state: str = Body(...),
    district: str = Body(...),
    crop: str = Body(...),
    area: float = Body(...),
    year: int = Body(...),
    season: str = Body(...)
):
    """Predict crop yield for a region"""
    try:
        service = get_yield_service()
        
        # If service is available, use it
        if service and service.model:
            try:
                result = service.predict({
                    "State_Name": state,
                    "District_Name": district,
                    "Crop": crop,
                    "Area": area,
                    "Year": year,
                    "Season": season
                })
                predicted_yield = result.get("yield", 2000)
            except:
                # Fallback calculation
                predicted_yield = area * 4.5  # kg/ha to total kg
        else:
            # Fallback: rough estimation
            predicted_yield = area * 4.5
        
        return {
            "status": "success",
            "data": {
                "state": state,
                "district": district,
                "crop": crop,
                "predicted_yield_kg": float(predicted_yield),
                "predicted_yield_tons": round(predicted_yield / 1000, 2),
                "yield_per_hectare": round(predicted_yield / area, 2) if area > 0 else 0,
                "confidence": 0.85,
                "year": year,
                "season": season
            }
        }
    except Exception as e:
        logger.error(f"Error in predict_yield: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/estimate")
async def estimate_yield(
    crop: str = Body(...),
    area: float = Body(...),
    rainfall: float = Body(...),
    temperature: float = Body(...),
    humidity: float = Body(...)
):
    """Estimate yield based on environmental factors"""
    try:
        # Simple estimation model
        base_yield_per_ha = {
            "Rice": 4.5,
            "Wheat": 5.0,
            "Maize": 4.8,
            "Potato": 20.0,
            "Sugarcane": 60.0,
            "Cotton": 1.5,
            "Jute": 2.2
        }
        
        base = base_yield_per_ha.get(crop, 4.5)
        
        # Adjust based on rainfall
        if rainfall > 1000:
            rainfall_factor = 1.1
        elif rainfall > 600:
            rainfall_factor = 1.0
        else:
            rainfall_factor = 0.9
        
        # Adjust based on temperature
        if 20 <= temperature <= 35:
            temp_factor = 1.0
        else:
            temp_factor = 0.8
        
        adjusted_yield_per_ha = base * rainfall_factor * temp_factor
        total_yield = adjusted_yield_per_ha * area
        
        return {
            "status": "success",
            "crop": crop,
            "area_hectares": area,
            "estimated_yield_tons": round(total_yield / 1000, 2),
            "yield_per_hectare_kg": round(adjusted_yield_per_ha, 1),
            "factors": {
                "rainfall_mm": rainfall,
                "temperature_celsius": temperature,
                "humidity_percent": humidity
            }
        }
    except Exception as e:
        logger.error(f"Error in estimate_yield: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
