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
stress_model_error = None

def get_stress_service():
    """Lazy load stress service"""
    global stress_service, stress_model_error
    if stress_service is None and stress_model_error is None:
        try:
            logger.info("[INIT] Loading stress prediction service...")
            from stress_prediction_service import StressPredictionService
            stress_service = StressPredictionService()
            logger.info("[OK] ✅ Stress service loaded successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load stress service: {e}")
            logger.error(traceback.format_exc())
            stress_model_error = str(e)
            stress_service = None
    return stress_service


@router.get("/options")
async def get_stress_options():
    """Get available stress monitoring parameters"""
    try:
        logger.info("[OK] Stress options requested")
        return {
            "status": "success",
            "data": {
                "crop_types": ["Maize", "Rice", "Paddy", "Sugarcane", "Cotton", "Tobacco", "Wheat", "Potato"],
                "growth_stages": ["Seedling", "Vegetative", "Flowering", "Grain Filling", "Maturity"],
                "stress_types": [
                    "Water Stress", "Heat Stress", "Cold Stress",
                    "Nutrient Deficiency", "Pest Damage", "Disease"
                ],
                "indicators": [
                    "Temperature", "Humidity", "Soil Moisture",
                    "Rainfall", "Leaf Color", "Plant Height"
                ],
                "severity_levels": ["Low", "Moderate", "High", "Critical"]
            }
        }
    except Exception as e:
        logger.error(f"[ERROR] Error in get_stress_options: {e}")
        logger.error(traceback.format_exc())
        # Return default data even if there's an error
        return {
            "status": "success",
            "data": {
                "crop_types": ["Maize", "Rice", "Paddy", "Sugarcane", "Cotton", "Tobacco", "Wheat", "Potato"],
                "growth_stages": ["Seedling", "Vegetative", "Flowering", "Grain Filling", "Maturity"],
                "stress_types": [
                    "Water Stress", "Heat Stress", "Cold Stress",
                    "Nutrient Deficiency", "Pest Damage", "Disease"
                ],
                "indicators": [
                    "Temperature", "Humidity", "Soil Moisture",
                    "Rainfall", "Leaf Color", "Plant Height"
                ],
                "severity_levels": ["Low", "Moderate", "High", "Critical"]
            },
            "warning": "Default data returned - service error"
        }


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
        logger.error(f"[ERROR] Error in predict_stress: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "success",
            "data": {
                "stress_type": "Unknown",
                "stress_level": "Low",
                "stress_score": 0.3,
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "recommendation": "Default recommendation - service error"
            }
        }


@router.post("/analyze")
async def analyze_stress(
    crop: str = Body(...),
    parameters: Dict[str, Any] = Body(...)
):
    """Analyze crop stress with detailed parameters"""
    try:
        logger.info(f"[STRESS] Received crop: {crop}")
        logger.info(f"[STRESS] Received parameters: {parameters}")
        
        # Extract and convert numeric parameters safely
        try:
            temp = float(parameters.get("temperature", 25))
            humid = float(parameters.get("humidity", 60))
            moisture = float(parameters.get("soil_moisture", 50))
            rainfall = float(parameters.get("rainfall", 0))
        except (ValueError, TypeError) as e:
            logger.error(f"[ERROR] Invalid numeric value: {e}")
            return JSONResponse(
                status_code=422,
                content={"status": "error", "message": f"Invalid numeric value: {e}"}
            )
        
        logger.info(f"[STRESS] Parsed: temp={temp}°C, humid={humid}%, moisture={moisture}%, rainfall={rainfall}mm")
        
        # Calculate stress metrics
        stress_factors = []
        if temp > 35:
            stress_factors.append("High Temperature")
        if humid < 40:
            stress_factors.append("Low Humidity")
        if moisture < 30:
            stress_factors.append("Water Stress")
        
        severity = "Low" if len(stress_factors) == 0 else "Moderate" if len(stress_factors) == 1 else "High"
        
        logger.info(f"[STRESS] Calculated severity: {severity}, factors: {stress_factors}")
        
        return {
            "status": "success",
            "crop": crop,
            "stress_factors": stress_factors,
            "severity": severity,
            "stress_level": severity,
            "advice": f"Crop: {crop}. Severity: {severity}. Factors: {', '.join(stress_factors) if stress_factors else 'None'}",
            "confidence_percentage": f"{(100 - len(stress_factors) * 25)}%",
            "recommendations": [
                "Monitor soil moisture daily",
                "Adjust irrigation schedule",
                "Check for pest activity"
            ]
        }
    except Exception as e:
        logger.error(f"[ERROR] Error in analyze_stress: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
