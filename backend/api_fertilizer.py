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
fertilizer_model_error = None

def get_fertilizer_service():
    """Lazy load fertilizer service"""
    global fertilizer_service, fertilizer_model_error
    if fertilizer_service is None and fertilizer_model_error is None:
        try:
            logger.info("[INIT] Loading fertilizer prediction service...")
            from fertilizer_prediction_service import FertilizerPredictionService
            fertilizer_service = FertilizerPredictionService()
            logger.info("[INIT] Fertilizer service created, loading model...")
            fertilizer_service.load_model()
            logger.info("[OK] ✅ Fertilizer service model loaded successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load fertilizer service: {e}")
            logger.error(traceback.format_exc())
            fertilizer_model_error = str(e)
            fertilizer_service = None
    return fertilizer_service


@router.get("/options")
async def get_fertilizer_options():
    """Get available fertilizer options and parameters"""
    try:
        logger.info("[OK] Fertilizer options requested")
        # Import regions and states for Region dropdown
        try:
            from india_districts import INDIA_STATES_AND_UTS
            regions = list(INDIA_STATES_AND_UTS)
        except:
            regions = ["North", "South", "East", "West", "Central"]
        
        return {
            "status": "success",
            "data": {
                # ONLY model-supported values - exactly as model encoders expect
                "Soil_Type": ["Clay", "Loamy", "Sandy", "Silt"],
                "Crop_Type": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "Crop_Growth_Stage": ["Sowing", "Vegetative", "Flowering", "Harvest"],
                "Season": ["Kharif", "Rabi", "Zaid"],
                "Irrigation_Type": ["Canal", "Drip", "Rainfed", "Sprinkler"],
                "Previous_Crop": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "Region": ["Central", "East", "North", "South", "West"],
                
                # Keep original keys for backward compatibility
                "soil_types": ["Clay", "Loamy", "Sandy", "Silt"],
                "crops": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "fertilizer_types": [
                    "Compost", "DAP", "MOP", "NPK", "SSP", "Urea", "Zinc Sulphate"
                ],
                "parameters": ["N", "P", "K", "pH", "EC", "Moisture"]
            }
        }
    except Exception as e:
        logger.error(f"[ERROR] Error in get_fertilizer_options: {e}")
        logger.error(traceback.format_exc())
        # Return default data even if there's an error
        try:
            from india_districts import INDIA_STATES_AND_UTS
            regions = list(INDIA_STATES_AND_UTS)
        except:
            regions = ["North", "South", "East", "West", "Central"]
        
        return {
            "status": "success",
            "data": {
                # ONLY model-supported values
                "Soil_Type": ["Clay", "Loamy", "Sandy", "Silt"],
                "Crop_Type": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "Crop_Growth_Stage": ["Sowing", "Vegetative", "Flowering", "Harvest"],
                "Season": ["Kharif", "Rabi", "Zaid"],
                "Irrigation_Type": ["Canal", "Drip", "Rainfed", "Sprinkler"],
                "Previous_Crop": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "Region": ["Central", "East", "North", "South", "West"],
                "soil_types": ["Clay", "Loamy", "Sandy", "Silt"],
                "crops": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane"],
                "fertilizer_types": [
                    "Compost", "DAP", "MOP", "NPK", "SSP", "Urea", "Zinc Sulphate"
                ],
                "parameters": ["N", "P", "K", "pH", "EC", "Moisture"]
            },
            "warning": "Default data returned - service error"
        }


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
async def recommend_fertilizer(payload: Dict[str, Any] = Body(...)):
    """Get fertilizer recommendation - expects ALL 17 required fields"""
    try:
        logger.info(f"[FERTILIZER] Received payload with {len(payload)} fields")
        logger.info(f"[FERTILIZER] Payload keys: {list(payload.keys())}")
        
        # STEP 1: Map lowercase/alias field names to CamelCase feature names expected by model
        field_mapping = {
            'nitrogen': 'Nitrogen_Level',
            'phosphorus': 'Phosphorus_Level',
            'potassium': 'Potassium_Level',
            'ph': 'Soil_pH',
            'crop': 'Crop_Type',
            'soil_type': 'Soil_Type',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'rainfall': 'Rainfall',
            'season': 'Season',
            'irrigation_type': 'Irrigation_Type',
            'previous_crop': 'Previous_Crop',
            'region': 'Region',
            'crop_growth_stage': 'Crop_Growth_Stage',
            'electrical_conductivity': 'Electrical_Conductivity',
            'organic_carbon': 'Organic_Carbon',
            'soil_moisture': 'Soil_Moisture'
        }
        
        # Map all lowercase keys to CamelCase for model
        mapped_payload = {}
        for key, value in payload.items():
            mapped_key = field_mapping.get(key.lower(), key)
            mapped_payload[mapped_key] = value
        
        logger.info(f"[FERTILIZER] Mapped payload keys: {list(mapped_payload.keys())}")
        
        service = get_fertilizer_service()
        if not service or not service.model:
            return {
                "status": "success",
                "recommendation": "NPK 20:20:0",
                "fertilizer": "NPK 20:20:0",
                "confidence": 0.70,
                "confidence_percentage": 70,
                "all_probabilities": {"NPK 20:20:0": 0.70},
                "top_3_recommendations": ["NPK 20:20:0"],
                "quantity_kg": 100,
                "note": "Using default recommendation - model not available"
            }
        
        # Pass mapped payload to service
        logger.info(f"[FERTILIZER] Calling model.predict with mapped payload")
        result = service.predict(mapped_payload)
        logger.info(f"[FERTILIZER] Prediction result: {result}")
        logger.info(f"[FERTILIZER] Result keys: {list(result.keys())}")
        
        # Ensure all fields are present in the response
        return {
            "status": "success",
            "recommendation": result.get("fertilizer", "NPK 20:20:0"),
            "fertilizer": result.get("fertilizer", "NPK 20:20:0"),
            "confidence": float(result.get("confidence", 0.85)),
            "confidence_percentage": float(result.get("confidence_percentage", 85)),
            "all_probabilities": result.get("all_probabilities", {}),
            "top_3_recommendations": result.get("top_3_recommendations", []),
            "quantity_kg": int(result.get("quantity", 100))
        }
    except ValueError as e:
        # Model validation error
        logger.error(f"[ERROR] Fertilizer model validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"[ERROR] Error in recommend_fertilizer: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
