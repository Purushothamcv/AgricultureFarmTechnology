"""
Fruit Disease Detection with Fruit Type Selection
==================================================
New comprehensive API endpoint for fruit disease detection with:
- Fruit type selection/validation
- Supported fruit checking
- Clean error handling for unsupported fruits
- Image validation
- Confidence scoring

Endpoint:
- POST /api/fruit-disease/predict-with-selection - Predict with fruit selection

Author: SmartAgri-AI Team
Date: 2026-01-25
"""

import os
import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import sys

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model.fruit_disease_detector import FruitDiseaseDetector
except ImportError:
    try:
        from model.fruit_disease_inference import FruitDiseasePredictor as FruitDiseaseDetector
    except ImportError:
        FruitDiseaseDetector = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/api/fruit-disease-detection",
    tags=["Fruit Disease Detection Selection"]
)

# Global detector instance
detector: Optional[FruitDiseaseDetector] = None

# Supported fruits based on trained model
SUPPORTED_FRUITS = {
    "Apple": ["Blotch", "Healthy", "Rot", "Scab"],
    "Mango": ["Alternaria", "Anthracnose", "Black Mould Rot (Aspergillus)", "Healthy", "Stem and Rot (Lasiodiplodia)"],
    "Pomegranate": ["Alternaria", "Anthracnose", "Bacterial Blight", "Cercospora", "Healthy"],
    "Guava": ["Anthracnose", "Fruitfly", "Healthy"]
}


async def startup_event():
    """Initialize detector at application startup"""
    get_detector()


def get_detector() -> Optional[FruitDiseaseDetector]:
    """Get detector instance"""
    global detector
    if detector is not None:
        return detector
    try:
        logger.info("Loading fruit disease model...")
        detector = FruitDiseaseDetector()
        logger.info("Fruit disease model loaded successfully")
        return detector
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        detector = None
        return None


def extract_fruit_from_class_name(class_name: str) -> str:
    """
    Extract fruit type from class name
    
    Examples:
        "Alternaria_Mango" -> "Mango"
        "Healthy_Apple" -> "Apple"
        "Blotch_Apple" -> "Apple"
    """
    parts = class_name.split('_')
    if len(parts) >= 2:
        return parts[-1]
    return "Unknown"


def is_fruit_supported(fruit_type: str, predicted_class: str) -> bool:
    """
    Check if the predicted class matches the selected fruit type
    
    Args:
        fruit_type: Selected fruit from dropdown
        predicted_class: Class name from model prediction
    
    Returns:
        True if the prediction is for the selected fruit type
    """
    predicted_fruit = extract_fruit_from_class_name(predicted_class)
    return predicted_fruit.lower() == fruit_type.lower()


@router.get("/supported-fruits")
async def get_supported_fruits():
    """
    Get list of supported fruits for detection
    
    Returns:
        List of fruit names with their common diseases
    """
    try:
        return {
            "success": True,
            "data": {
                "fruits": list(SUPPORTED_FRUITS.keys()),
                "supported_fruits_details": SUPPORTED_FRUITS,
                "total_fruits": len(SUPPORTED_FRUITS)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching supported fruits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-with-selection")
async def predict_with_fruit_selection(
    fruit_type: str = Form(..., description="Selected fruit type (e.g., Apple, Mango, Pomegranate, Guava)"),
    file: UploadFile = File(..., description="Fruit image file (JPEG, PNG)"),
    confidence_threshold: float = Form(0.50, description="Minimum confidence threshold"),
    debug: bool = Form(False, description="Enable debug logging")
):
    """
    Predict fruit disease with fruit type selection
    
    Key Features:
    - Validates fruit type is supported
    - Checks if prediction matches selected fruit
    - Returns appropriate error messages
    - Validates image quality
    
    Args:
        fruit_type: Selected fruit type from dropdown
        file: Uploaded image file
        confidence_threshold: Minimum confidence for prediction
        debug: Enable debug logging
    
    Returns:
        JSON with prediction or appropriate error message
        
    Example:
        curl -X POST "http://localhost:8000/api/fruit-disease-detection/predict-with-selection" \
             -F "fruit_type=Mango" \
             -F "file=@mango.jpg" \
             -F "confidence_threshold=0.50"
    """
    try:
        # Step 1: Validate fruit type selection
        if not fruit_type or fruit_type.strip() == "":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please select a fruit type from the dropdown",
                    "data": None
                }
            )
        
        fruit_type = fruit_type.strip()
        
        # Step 2: Check if fruit is supported
        if fruit_type not in SUPPORTED_FRUITS:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"This fruit is currently not supported or not available in the trained model. Supported fruits: {', '.join(SUPPORTED_FRUITS.keys())}",
                    "data": {
                        "selected_fruit": fruit_type,
                        "supported_fruits": list(SUPPORTED_FRUITS.keys())
                    }
                }
            )
        
        # Step 3: Validate image file
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid file type. Please upload an image (JPEG, PNG, etc.)",
                    "data": None
                }
            )
        
        # Step 4: Check file size
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "File too large. Maximum size is 10MB",
                    "data": None
                }
            )
        
        # Step 5: Validate image format
        try:
            image = Image.open(io.BytesIO(contents))
            # Try to load image data to ensure it's valid
            image.load()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Unable to detect disease. Please upload a valid fruit image.",
                    "data": None
                }
            )
        
        # Step 6: Get detector
        det = get_detector()
        if det is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Fruit disease model temporarily unavailable on cloud deployment",
                    "data": None
                }
            )
        
        # Step 7: Make prediction
        try:
            result = det.predict_with_details(
                image,
                top_n=3,
                confidence_threshold=confidence_threshold,
                debug=debug
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Unable to detect disease. Please upload a valid fruit image.",
                    "data": None
                }
            )
        
        # Step 8: Validate prediction matches selected fruit
        predicted_class = result.get('prediction', '')
        predicted_fruit = extract_fruit_from_class_name(predicted_class)
        
        # Check if prediction fruit matches selected fruit
        if not is_fruit_supported(fruit_type, predicted_class):
            logger.warning(
                f"Fruit mismatch: Selected={fruit_type}, Predicted={predicted_fruit}, "
                f"Class={predicted_class}, Confidence={result.get('confidence', 0):.2%}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Unable to detect disease. Please upload a valid fruit image.",
                    "data": {
                        "selected_fruit": fruit_type,
                        "detected_fruit": predicted_fruit,
                        "confidence": result.get('confidence', 0)
                    }
                }
            )
        
        # Step 9: Confidence-based validation (< 55% threshold)
        confidence = result.get('confidence', 0)
        CONFIDENCE_THRESHOLD = 0.55
        
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Low confidence prediction: Selected={fruit_type}, "
                f"Predicted={predicted_class}, Confidence={confidence:.2%}"
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "selected_fruit": fruit_type,
                        "is_low_confidence": True,
                        "confidence": confidence,
                        "message": "Low confidence detected. Please upload a clearer and valid fruit image.",
                        "prediction": result.get('prediction', ''),
                        "top_3": result.get('top_3', [])
                    },
                    "filename": file.filename
                }
            )
        
        # Step 10: Return successful prediction (confidence >= 55%)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "selected_fruit": fruit_type,
                    "is_low_confidence": False,
                    "prediction": result.get('prediction', ''),
                    "confidence": result.get('confidence', 0),
                    "disease_info": result.get('disease_info', {}),
                    "interpretation": result.get('interpretation', ''),
                    "warnings": result.get('warnings', []),
                    "has_warnings": result.get('has_warnings', False),
                    "action_required": result.get('action_required', 'NONE'),
                    "top_3": result.get('top_3', [])
                },
                "filename": file.filename
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in predict_with_fruit_selection: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An unexpected error occurred during prediction",
                "data": None
            }
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service status and availability
    """
    try:
        det = get_detector()
        if det is None:
            return {
                "status": "unhealthy",
                "service": "Fruit Disease Detection (Selection)",
                "error": "Detector not initialized",
                "supported_fruits": list(SUPPORTED_FRUITS.keys())
            }
        
        return {
            "status": "healthy",
            "service": "Fruit Disease Detection (Selection)",
            "detector_ready": True,
            "supported_fruits": list(SUPPORTED_FRUITS.keys()),
            "total_fruits": len(SUPPORTED_FRUITS)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
