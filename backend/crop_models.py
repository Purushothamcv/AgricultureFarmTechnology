"""
Crop Recommendation Pydantic Models
Defines input/output schemas for crop recommendation endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class ManualCropInput(BaseModel):
    """Schema for manual crop recommendation input"""
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen content (kg/ha)")
    phosphorus: float = Field(..., ge=0, le=200, description="Phosphorus content (kg/ha)")
    potassium: float = Field(..., ge=0, le=200, description="Potassium content (kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=3.0, le=10.0, description="pH value")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)")
    ozone: float = Field(..., ge=0, le=100, description="Ozone level (ppb)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 20.87,
                "humidity": 82.00,
                "ph": 6.50,
                "rainfall": 202.93
            }
        }


class LocationCropInput(BaseModel):
    """Schema for location-based crop recommendation input"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    
    # Optional override values (if user wants to edit auto-fetched data)
    nitrogen: Optional[float] = Field(None, ge=0, le=200)
    phosphorus: Optional[float] = Field(None, ge=0, le=200)
    potassium: Optional[float] = Field(None, ge=0, le=200)
    temperature: Optional[float] = Field(None, ge=-10, le=60)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    ph: Optional[float] = Field(None, ge=3.0, le=10.0)
    rainfall: Optional[float] = Field(None, ge=0, le=500)
    ozone: Optional[float] = Field(None, ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 28.6139,
                "longitude": 77.2090,
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 25.0,
                "humidity": 80.0,
                "ph": 6.5,
                "rainfall": 200.0
            }
        }


class CropPredictionResponse(BaseModel):
    """Schema for crop recommendation response"""
    success: bool
    crop: str
    confidence: Optional[float] = None
    input_values: dict
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "crop": "Rice",
                "confidence": 0.95,
                "input_values": {
                    "nitrogen": 90,
                    "phosphorus": 42,
                    "potassium": 43,
                    "temperature": 20.87,
                    "humidity": 82.00,
                    "ph": 6.50,
                    "rainfall": 202.93
                },
                "message": "Crop recommendation generated successfully"
            }
        }


class LocationDataResponse(BaseModel):
    """Schema for fetched location data"""
    success: bool
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    rainfall: float
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    ozone: float
    location_name: Optional[str] = None
    message: Optional[str] = None
