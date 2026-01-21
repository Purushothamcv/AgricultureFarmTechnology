"""
Fruit Disease Detection FastAPI Service
=======================================
REST API endpoints for fruit disease prediction

Endpoints:
- POST /api/fruit-disease/predict - Single image prediction
- POST /api/fruit-disease/predict-batch - Batch prediction
- GET /api/fruit-disease/classes - Get available classes
- GET /api/fruit-disease/health - Check service health

Author: SmartAgri-AI Team
Date: 2026-01-21
"""

import os
import io
import logging
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fruit_disease_inference import FruitDiseasePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/fruit-disease", tags=["Fruit Disease Detection"])

# Global predictor instance (loaded once at startup)
predictor: Optional[FruitDiseasePredictor] = None


def get_predictor():
    """Get or initialize predictor instance"""
    global predictor
    if predictor is None:
        try:
            logger.info("Initializing Fruit Disease Predictor...")
            predictor = FruitDiseasePredictor()
            logger.info("✓ Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Model initialization failed: {str(e)}"
            )
    return predictor


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service status
    """
    try:
        pred = get_predictor()
        return {
            "status": "healthy",
            "service": "Fruit Disease Detection",
            "model_loaded": pred.model is not None,
            "num_classes": len(pred.labels) if pred.labels else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/classes")
async def get_classes():
    """
    Get all available disease classes
    
    Returns:
        List of class names organized by fruit
    """
    try:
        pred = get_predictor()
        
        # Organize classes by fruit
        classes_by_fruit = {}
        for idx, class_name in pred.labels.items():
            parts = class_name.split('_')
            fruit = parts[-1] if parts else "Unknown"
            
            if fruit not in classes_by_fruit:
                classes_by_fruit[fruit] = []
            
            classes_by_fruit[fruit].append({
                "index": idx,
                "name": class_name,
                "disease": '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
            })
        
        return {
            "total_classes": len(pred.labels),
            "fruits": list(classes_by_fruit.keys()),
            "classes_by_fruit": classes_by_fruit,
            "all_classes": list(pred.labels.values())
        }
        
    except Exception as e:
        logger.error(f"Error fetching classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(..., description="Fruit image file"),
    top_n: int = 3
):
    """
    Predict fruit disease from uploaded image
    
    Args:
        file: Uploaded image file
        top_n: Number of top predictions to return (default: 3)
    
    Returns:
        Prediction results with confidence scores and recommendations
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict_with_recommendations(image, top_n=top_n)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        # Remove all_probabilities for cleaner response (optional)
        result.pop('all_probabilities', None)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "filename": file.filename
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple fruit images"),
    top_n: int = 3
):
    """
    Batch prediction for multiple images
    
    Args:
        files: List of uploaded image files
        top_n: Number of top predictions per image
    
    Returns:
        List of prediction results
    """
    try:
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
        
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch"
            )
        
        # Get predictor
        pred = get_predictor()
        
        # Process each image
        results = []
        for file in files:
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Invalid file type"
                    })
                    continue
                
                # Read and predict
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                result = pred.predict_with_recommendations(image, top_n=top_n)
                result['filename'] = file.filename
                
                # Remove all_probabilities
                result.pop('all_probabilities', None)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "success": True,
            "total_images": len(files),
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_model_info():
    """
    Get model information and statistics
    
    Returns:
        Model architecture and performance info
    """
    try:
        pred = get_predictor()
        
        return {
            "model_name": "Fruit Disease Detector",
            "architecture": "EfficientNet-B0 (Transfer Learning)",
            "framework": "TensorFlow/Keras",
            "input_size": "224x224",
            "num_classes": len(pred.labels),
            "supported_fruits": ["Apple", "Guava", "Mango", "Pomegranate"],
            "model_path": pred.model_path,
            "labels_path": pred.labels_path
        }
        
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event to preload model
async def startup_event():
    """Initialize predictor on startup"""
    logger.info("Starting Fruit Disease Detection Service...")
    try:
        get_predictor()
        logger.info("✓ Service ready")
    except Exception as e:
        logger.error(f"✗ Service initialization failed: {e}")
