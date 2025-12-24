"""
Database Helper Functions for FinalProject Collections
Provides utility functions to interact with MongoDB collections
"""

from database import get_database
from datetime import datetime
from typing import Dict, Any, Optional


async def save_disease_prediction(
    user_email: str,
    image_path: str,
    disease_name: str,
    confidence: float,
    recommendations: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save plant disease prediction to MongoDB
    
    Args:
        user_email: Email of user who made the prediction
        image_path: Path to the uploaded image
        disease_name: Detected disease name
        confidence: Prediction confidence score
        recommendations: Treatment recommendations
    
    Returns:
        Dictionary with insertion result
    """
    db = get_database()
    
    prediction_doc = {
        "user_email": user_email,
        "image_path": image_path,
        "disease_name": disease_name,
        "confidence": confidence,
        "recommendations": recommendations,
        "predicted_at": datetime.utcnow()
    }
    
    result = await db.plant_disease_predictions.insert_one(prediction_doc)
    
    return {
        "success": True,
        "prediction_id": str(result.inserted_id),
        "message": "Disease prediction saved successfully"
    }


async def get_user_predictions(user_email: str, limit: int = 10) -> list:
    """
    Get disease predictions history for a user
    
    Args:
        user_email: Email of the user
        limit: Maximum number of predictions to return
    
    Returns:
        List of prediction documents
    """
    db = get_database()
    
    predictions = await db.plant_disease_predictions.find(
        {"user_email": user_email}
    ).sort("predicted_at", -1).limit(limit).to_list(length=limit)
    
    # Convert ObjectId to string for JSON serialization
    for pred in predictions:
        pred["_id"] = str(pred["_id"])
    
    return predictions


async def log_weather_data(
    user_email: str,
    location: Dict[str, float],
    weather_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Log weather data fetched by user
    
    Args:
        user_email: Email of user
        location: Dictionary with lat and lon
        weather_data: Weather information
    
    Returns:
        Dictionary with insertion result
    """
    db = get_database()
    
    log_doc = {
        "user_email": user_email,
        "location": location,
        "weather_data": weather_data,
        "fetched_at": datetime.utcnow()
    }
    
    result = await db.weather_logs.insert_one(log_doc)
    
    return {
        "success": True,
        "log_id": str(result.inserted_id),
        "message": "Weather data logged successfully"
    }


async def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the FinalProject database
    
    Returns:
        Dictionary with collection statistics
    """
    db = get_database()
    
    users_count = await db.users.count_documents({})
    predictions_count = await db.plant_disease_predictions.count_documents({})
    weather_logs_count = await db.weather_logs.count_documents({})
    
    return {
        "database": "FinalProject",
        "collections": {
            "users": users_count,
            "plant_disease_predictions": predictions_count,
            "weather_logs": weather_logs_count
        },
        "total_documents": users_count + predictions_count + weather_logs_count
    }
