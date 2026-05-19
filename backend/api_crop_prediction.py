"""
Crop Prediction API
===================

Handles crop recommendation based on location and soil parameters.
Endpoint: POST /predict/location
"""

from fastapi import APIRouter, HTTPException
from crop_models import LocationCropInput, CropPredictionResponse
from crop_service import predict_crop, fetch_all_location_data
import traceback
import os

# Create router (no prefix - path will be /predict/location)
router = APIRouter(tags=["crop-prediction"])

print("[INIT] Checking crop model files...")
print(f"  - crop_model.pkl exists: {os.path.exists('model/crop_model.pkl')}")
print(f"  - scaler.pkl exists: {os.path.exists('model/scaler.pkl')}")
print(f"  - label_encoder.pkl exists: {os.path.exists('model/label_encoder.pkl')}")


@router.post("/predict/location", response_model=CropPredictionResponse)
async def predict_crop_location(input_data: LocationCropInput) -> CropPredictionResponse:
    """
    Crop recommendation based on location (map selection) with manual NPK input.
    
    Important: NPK values MUST be provided by the user from soil test.
    Only weather and pH data can be fetched automatically from location.
    
    Request Body (JSON):
    {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 25.0,  // optional - will auto-fetch if not provided
        "humidity": 80.0,     // optional
        "ph": 6.5,            // optional
        "rainfall": 200.0,    // optional
        "ozone": 30.0         // optional
    }
    
    Response:
    {
        "success": true,
        "crop": "Rice",
        "confidence": 0.95,
        "input_values": { ... },
        "message": "Crop recommendation generated successfully"
    }
    
    Args:
        input_data: LocationCropInput with required NPK values and optional weather/pH
    
    Returns:
        CropPredictionResponse with crop prediction and confidence score
    
    Raises:
        HTTPException: 400 for invalid input, 500 for prediction errors
    """
    print(f"\n[REQUEST] POST /predict/location")
    print(f"  Latitude: {input_data.latitude}")
    print(f"  Longitude: {input_data.longitude}")
    print(f"  Nitrogen: {input_data.nitrogen}")
    print(f"  Phosphorus: {input_data.phosphorus}")
    print(f"  Potassium: {input_data.potassium}")
    
    try:
        # Validate NPK values are provided (required)
        if input_data.nitrogen is None or input_data.phosphorus is None or input_data.potassium is None:
            print("[ERROR] NPK values are required")
            raise HTTPException(
                status_code=400,
                detail="Nitrogen, phosphorus, and potassium values are required (must come from soil test)"
            )
        
        # Fetch missing weather and pH data if not all provided
        if any(v is None for v in [
            input_data.temperature, 
            input_data.humidity, 
            input_data.ph, 
            input_data.rainfall, 
            input_data.ozone
        ]):
            print(f"[INFO] Fetching missing weather/soil data from location...")
            try:
                location_data = await fetch_all_location_data(
                    input_data.latitude,
                    input_data.longitude
                )
            except Exception as e:
                print(f"[WARN] Could not fetch location data: {e}")
                location_data = {}
        else:
            location_data = {}
        
        # Use provided values or fallback to fetched values
        temperature = input_data.temperature if input_data.temperature is not None else location_data.get("temperature", 25.0)
        humidity = input_data.humidity if input_data.humidity is not None else location_data.get("humidity", 70.0)
        ph = input_data.ph if input_data.ph is not None else location_data.get("ph", 6.5)
        rainfall = input_data.rainfall if input_data.rainfall is not None else location_data.get("rainfall", 100.0)
        ozone = input_data.ozone if input_data.ozone is not None else location_data.get("ozone", 30.0)
        
        print(f"[INFO] Final input values:")
        print(f"  Temperature: {temperature}")
        print(f"  Humidity: {humidity}")
        print(f"  pH: {ph}")
        print(f"  Rainfall: {rainfall}")
        print(f"  Ozone: {ozone}")
        
        # Make prediction using crop recommendation model
        print(f"[INFO] Calling predict_crop()...")
        crop, confidence = predict_crop(
            nitrogen=input_data.nitrogen,
            phosphorus=input_data.phosphorus,
            potassium=input_data.potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            ozone=ozone
        )
        
        print(f"[SUCCESS] Prediction: {crop} (confidence: {confidence})")
        
        return CropPredictionResponse(
            success=True,
            crop=crop,
            confidence=confidence,
            input_values={
                "latitude": input_data.latitude,
                "longitude": input_data.longitude,
                "nitrogen": input_data.nitrogen,
                "phosphorus": input_data.phosphorus,
                "potassium": input_data.potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "ozone": ozone
            },
            message="Crop recommendation generated successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        print(f"\n[ERROR] Exception in /predict/location:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(e)}"
        )
