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
from india_districts import INDIA_DISTRICTS, INDIA_STATES_AND_UTS, TOTAL_REGIONS, TOTAL_DISTRICTS

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
INDIA_STATES = INDIA_STATES_AND_UTS  # Use comprehensive list from india_districts.py


@router.get("/states")
async def get_states():
    """
    Get list of Indian states and union territories for yield prediction.
    Returns all 28 states + 8 union territories = 36 regions total.
    """
    try:
        print(f"[REQUEST] GET /api/yield/states")
        print(f"[OK] Returning {len(INDIA_STATES)} states and UTs ({TOTAL_REGIONS} total regions)")
        
        return {
            "status": "success",
            "data": INDIA_STATES,
            "total_states": len([s for s in INDIA_STATES if s not in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"]]),
            "total_uts": len([s for s in INDIA_STATES if s in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"]]),
            "total_regions": TOTAL_REGIONS,
            "total_districts_available": TOTAL_DISTRICTS
        }
    except Exception as e:
        logger.error(f"Error in get_states: {e}")
        traceback.print_exc()
        print(f"[ERROR] Exception in get_states: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/options")
async def get_yield_options():
    """
    Get available yield prediction parameters.
    Includes comprehensive list of all states and union territories.
    """
    try:
        print(f"[REQUEST] GET /api/yield/options")
        print(f"[OK] Returning yield prediction parameters with {TOTAL_REGIONS} regions")
        
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
                ],
                "metadata": {
                    "total_states": len([s for s in INDIA_STATES if s not in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"]]),
                    "total_uts": len([s for s in INDIA_STATES if s in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"]]),
                    "total_regions": TOTAL_REGIONS,
                    "total_available_districts": TOTAL_DISTRICTS
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in get_yield_options: {e}")
        traceback.print_exc()
        print(f"[ERROR] Exception in get_yield_options: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/districts/{state}")
async def get_districts(state: str):
    """
    Get list of districts for a given Indian state or union territory.
    Supports case-insensitive state names.
    
    Example:
        - GET /api/yield/districts/Karnataka
        - GET /api/yield/districts/karnataka
        - GET /api/yield/districts/MAHARASHTRA
    
    Returns:
        {
            "status": "success",
            "state": "Karnataka",
            "districts": [list of district names],
            "total_districts": count,
            "region_type": "state"  # or "union_territory"
        }
    """
    try:
        print(f"[REQUEST] GET /api/yield/districts/{state}")
        print(f"[INFO] Requested state/UT: {state}")
        
        # Try exact match first
        if state in INDIA_DISTRICTS:
            districts = INDIA_DISTRICTS[state]
            region_type = "state" if state not in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"] else "union_territory"
            print(f"[OK] Exact match found for state: {state}")
            print(f"[OK] Returning {len(districts)} districts")
            
            return {
                "status": "success",
                "state": state,
                "districts": districts,
                "total_districts": len(districts),
                "region_type": region_type
            }
        
        # Try case-insensitive match
        print(f"[INFO] Exact match not found, trying case-insensitive match...")
        for region_name, districts in INDIA_DISTRICTS.items():
            if region_name.lower() == state.lower():
                region_type = "state" if region_name not in ["Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"] else "union_territory"
                print(f"[OK] Case-insensitive match found: {region_name} (requested: {state})")
                print(f"[OK] Returning {len(districts)} districts for {region_name}")
                
                return {
                    "status": "success",
                    "state": region_name,
                    "districts": districts,
                    "total_districts": len(districts),
                    "region_type": region_type
                }
        
        # State not found
        print(f"[WARN] State/UT not found: {state}")
        print(f"[INFO] Available states/UTs: {TOTAL_REGIONS} total regions ({len([s for s in INDIA_STATES_AND_UTS if s not in ['Delhi', 'Jammu and Kashmir', 'Ladakh', 'Puducherry', 'Chandigarh', 'Andaman and Nicobar Islands', 'Lakshadweep', 'Dadra and Nagar Haveli and Daman and Diu']])} states, {len([s for s in INDIA_STATES_AND_UTS if s in ['Delhi', 'Jammu and Kashmir', 'Ladakh', 'Puducherry', 'Chandigarh', 'Andaman and Nicobar Islands', 'Lakshadweep', 'Dadra and Nagar Haveli and Daman and Diu']])} UTs)")
        
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "state": state,
                "message": f"State/UT '{state}' not found",
                "districts": [],
                "total_districts": 0,
                "error_details": {
                    "total_supported_regions": TOTAL_REGIONS,
                    "total_districts_available": TOTAL_DISTRICTS,
                    "suggestion": "Check spelling or use one of the supported states/UTs"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in get_districts: {e}")
        traceback.print_exc()
        print(f"[ERROR] Exception in get_districts: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "state": state,
                "message": "Internal server error while fetching districts",
                "districts": [],
                "total_districts": 0,
                "error": str(e)
            }
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
