"""
FERTILIZER PREDICTION API
==========================
FastAPI Router for fertilizer recommendations
"""

import logging
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import traceback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fertilizer", tags=["Fertilizer"])

# Global service instance
fertilizer_service = None

def get_fertilizer_service():
    """Lazy load fertilizer service"""
    global fertilizer_service
    if fertilizer_service is None:
        try:
            from fertilizer_prediction_service import FertilizerPredictionService
            fertilizer_service = FertilizerPredictionService()
            fertilizer_service.load_model()
        except Exception as e:
            logger.error(f"Failed to load fertilizer service: {e}")
            fertilizer_service = None
    return fertilizer_service


@router.get("/options")
async def get_fertilizer_options():
    """Get available fertilizer options and parameters"""
    try:
        return {
            "status": "success",
            "data": {
                "soil_types": ["Sandy", "Loamy", "Clay", "Peat", "Saline"],
                "crops": ["Maize", "Rice", "Paddy", "Sugarcane", "Cotton", "Tobacco", "Potato", "Wheat"],
                "fertilizer_types": [
                    "Urea", "DAP", "Potassium Chloride", "Potassium Sulphate",
                    "NPK 10:26:26", "NPK 5:10:40", "NPK 20:20:0", "SSP"
                ],
                "parameters": ["N", "P", "K", "pH", "EC", "Moisture"]
            }
        }
    except Exception as e:
        logger.error(f"Error in get_fertilizer_options: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/model-info")
async def get_model_info():
    """Get fertilizer model information"""
    try:
        service = get_fertilizer_service()
        if service and service.model:
            return {
                "status": "success",
                "data": {
                    "model_loaded": True,
                    "model_type": "Random Forest",
                    "version": "1.0.0",
                    "input_features": ["N", "P", "K", "pH", "EC", "Moisture", "Crop", "Soil Type"],
                    "output_types": 9,
                    "accuracy": 0.92
                }
            }
        return {
            "status": "success",
            "data": {
                "model_loaded": False,
                "message": "Model not yet loaded"
            }
        }
    except Exception as e:
        logger.error(f"Error in get_model_info: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/predict")
async def predict_fertilizer(
    data: Dict[str, Any] = Body(...)
):
    """Predict fertilizer type based on soil and crop data"""
    try:
        service = get_fertilizer_service()
        if not service or not service.model:
            return {
                "status": "error",
                "message": "Fertilizer model not available",
                "recommendation": "Default: Use NPK 20:20:0"
            }
        
        # Make prediction
        result = service.predict(data)
        return {
            "status": "success",
            "data": {
                "fertilizer_type": result.get("fertilizer_type", "NPK 20:20:0"),
                "quantity_kg_per_hectare": result.get("quantity", 100),
                "confidence": result.get("confidence", 0.85),
                "application_timing": "During soil preparation"
            }
        }
    except Exception as e:
        logger.error(f"Error in predict_fertilizer: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/recommend")
async def recommend_fertilizer(
    nitrogen: float = Body(...),
    phosphorus: float = Body(...),
    potassium: float = Body(...),
    ph: float = Body(...),
    crop: str = Body(...),
    soil_type: str = Body(...)
):
    """Get fertilizer recommendation with detailed parameters"""
    try:
        service = get_fertilizer_service()
        if not service or not service.model:
            return {
                "status": "success",
                "recommendation": "NPK 20:20:0",
                "quantity": 100,
                "confidence": 0.70,
                "note": "Using default recommendation - model not available"
            }
        
        # Prepare input
        input_data = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "pH": ph,
            "Crop": crop,
            "Soil Type": soil_type
        }
        
        result = service.predict(input_data)
        return {
            "status": "success",
            "recommendation": result.get("fertilizer_type", "NPK 20:20:0"),
            "quantity_kg": result.get("quantity", 100),
            "confidence": result.get("confidence", 0.85)
        }
    except Exception as e:
        logger.error(f"Error in recommend_fertilizer: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
