"""
STRESS PREDICTION API
====================
FastAPI Router for crop stress monitoring
"""

import logging
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import traceback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stress", tags=["Stress"])

# Global service instance
stress_service = None

def get_stress_service():
    """Lazy load stress service"""
    global stress_service
    if stress_service is None:
        try:
            from stress_prediction_service import StressPredictionService
            stress_service = StressPredictionService()
        except Exception as e:
            logger.error(f"Failed to load stress service: {e}")
            stress_service = None
    return stress_service


@router.get("/options")
async def get_stress_options():
    """Get available stress monitoring parameters"""
    try:
        return {
            "status": "success",
            "data": {
                "stress_types": [
                    "Water Stress", "Heat Stress", "Cold Stress",
                    "Nutrient Deficiency", "Pest Damage", "Disease"
                ],
                "indicators": [
                    "Temperature", "Humidity", "Soil Moisture",
                    "Rainfall", "Leaf Color", "Plant Height"
                ],
                "severity_levels": ["Low", "Moderate", "High", "Critical"],
                "crops": ["Maize", "Rice", "Paddy", "Sugarcane", "Cotton", "Wheat", "Potato"]
            }
        }
    except Exception as e:
        logger.error(f"Error in get_stress_options: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/predict")
async def predict_stress(
    temperature: float = Body(...),
    humidity: float = Body(...),
    soil_moisture: float = Body(...),
    rainfall: float = Body(...),
    crop: str = Body(...)
):
    """Predict crop stress levels"""
    try:
        service = get_stress_service()
        
        # Basic stress prediction logic
        stress_level = "Low"
        stress_score = 0.2
        
        if temperature > 35 or temperature < 15:
            stress_level = "High"
            stress_score = 0.75
        elif soil_moisture < 30:
            stress_level = "High"
            stress_score = 0.70
        elif humidity < 40:
            stress_level = "Moderate"
            stress_score = 0.50
        
        return {
            "status": "success",
            "data": {
                "stress_type": "Water Stress" if soil_moisture < 30 else "Heat Stress" if temperature > 35 else "Low Stress",
                "stress_level": stress_level,
                "stress_score": stress_score,
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "recommendation": "Increase irrigation" if soil_moisture < 30 else "Monitor regularly"
            }
        }
    except Exception as e:
        logger.error(f"Error in predict_stress: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/analyze")
async def analyze_stress(
    crop: str = Body(...),
    parameters: Dict[str, float] = Body(...)
):
    """Analyze crop stress with detailed parameters"""
    try:
        # Extract parameters
        temp = parameters.get("temperature", 25)
        humid = parameters.get("humidity", 60)
        moisture = parameters.get("soil_moisture", 50)
        rainfall = parameters.get("rainfall", 0)
        
        # Calculate stress metrics
        stress_factors = []
        if temp > 35:
            stress_factors.append("High Temperature")
        if humid < 40:
            stress_factors.append("Low Humidity")
        if moisture < 30:
            stress_factors.append("Water Stress")
        
        severity = "Low" if len(stress_factors) == 0 else "Moderate" if len(stress_factors) == 1 else "High"
        
        return {
            "status": "success",
            "crop": crop,
            "stress_factors": stress_factors,
            "severity": severity,
            "recommendations": [
                "Monitor soil moisture daily",
                "Adjust irrigation schedule",
                "Check for pest activity"
            ]
        }
    except Exception as e:
        logger.error(f"Error in analyze_stress: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
