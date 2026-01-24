"""
Fruit Disease Detection - FastAPI Service (Production)
======================================================
Clean, production-ready API endpoints for fruit disease detection

Uses the newly trained EfficientNet-B0 model (92%+ accuracy)
Optimized for fast inference and frontend integration

Endpoints:
- POST /api/v2/fruit-disease/predict - Single image prediction
- POST /api/v2/fruit-disease/predict-batch - Batch prediction
- GET /api/v2/fruit-disease/classes - Get available classes
- GET /api/v2/fruit-disease/health - Health check

Author: SmartAgri-AI
Date: 2026-01-24
"""

import os
import io
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import sys

# Add model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.fruit_disease_detector import FruitDiseaseDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/api/v2/fruit-disease",
    tags=["Fruit Disease Detection V2"]
)

# Global detector instance (loaded once at startup)
detector: Optional[FruitDiseaseDetector] = None


async def startup_event():
    """Initialize detector at application startup"""
    global detector
    try:
        logger.info("🚀 Initializing Fruit Disease Detector...")
        detector = FruitDiseaseDetector()
        logger.info("✅ Detector initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        raise


def get_detector() -> FruitDiseaseDetector:
    """Get detector instance"""
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Fruit disease detection service not initialized"
        )
    return detector


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service status and model information
    """
    try:
        det = get_detector()
        return {
            "status": "healthy",
            "service": "Fruit Disease Detection V2",
            "model": "EfficientNet-B0",
            "model_loaded": det.model is not None,
            "num_classes": len(det.labels),
            "input_size": f"{det.img_width}x{det.img_height}",
            "version": "2.0.0"
        }
    except HTTPException:
        return {
            "status": "unhealthy",
            "error": "Detector not initialized"
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
        det = get_detector()
        
        # Organize classes by fruit
        classes_by_fruit = {}
        all_classes = []
        
        for idx, class_name in det.labels.items():
            all_classes.append(class_name)
            
            # Extract fruit name (last part after underscore)
            parts = class_name.split('_')
            fruit = parts[-1] if parts else "Unknown"
            disease = '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
            
            if fruit not in classes_by_fruit:
                classes_by_fruit[fruit] = []
            
            classes_by_fruit[fruit].append({
                "index": int(idx),
                "full_name": class_name,
                "disease": disease,
                "is_healthy": "Healthy" in class_name
            })
        
        return {
            "total_classes": len(det.labels),
            "fruits": list(classes_by_fruit.keys()),
            "classes_by_fruit": classes_by_fruit,
            "all_classes": sorted(all_classes)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(..., description="Fruit image file (JPEG, PNG)"),
    top_n: int = Form(3, description="Number of top predictions to return"),
    confidence_threshold: float = Form(0.70, description="Minimum confidence for reliable prediction"),
    debug: bool = Form(False, description="Enable debug logging")
):
    """
    Predict fruit disease from uploaded image with robust safety checks
    
    IMPROVEMENTS:
    - Confidence thresholding to detect uncertain predictions
    - Top-3 decision logic to catch potential false negatives
    - Warnings for ambiguous "Healthy" predictions
    - Dynamic severity assessment based on confidence
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        top_n: Number of top predictions to return (default: 3)
        confidence_threshold: Minimum confidence for reliable prediction (default: 0.70)
        debug: Enable detailed debug logging (default: False)
    
    Returns:
        JSON response with prediction results:
        {
            "success": true,
            "data": {
                "prediction": "Disease_Name",
                "confidence": 0.94,
                "top_3": [...],
                "disease_info": {...},
                "interpretation": "...",
                "warnings": [...],
                "has_warnings": true/false,
                "action_required": "NONE|EXPERT_REVIEW|UPLOAD_BETTER_IMAGE"
            },
            "filename": "..."
        }
    
    Example:
        curl -X POST "http://localhost:8000/api/v2/fruit-disease/predict" \
             -F "file=@mango.jpg" \
             -F "top_n=3" \
             -F "confidence_threshold=0.70" \
             -F "debug=false"
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image (JPEG, PNG, etc.)"
            )
        
        # Validate file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB"
            )
        
        # Load image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Get detector
        det = get_detector()
        
        # Make prediction with new parameters
        result = det.predict_with_details(
            image, 
            top_n=top_n,
            confidence_threshold=confidence_threshold,
            debug=debug
        )
        
        # Log prediction with warnings
        log_message = f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})"
        if result.get('has_warnings', False):
            log_message += f" [WARNINGS: {len(result.get('warnings', []))}]"
            logger.warning(log_message)
            for warning in result.get('warnings', []):
                logger.warning(f"  - {warning}")
        else:
            logger.info(log_message)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": result,
                "filename": file.filename
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict-batch")
async def predict_disease_batch(
    files: List[UploadFile] = File(..., description="Multiple fruit images"),
    top_n: int = Form(3, description="Number of top predictions per image")
):
    """
    Predict fruit disease for multiple images
    
    Args:
        files: List of uploaded image files
        top_n: Number of top predictions per image
    
    Returns:
        JSON response with prediction results for each image
    
    Example:
        curl -X POST "http://localhost:8000/api/v2/fruit-disease/predict-batch" \
             -F "files=@mango1.jpg" \
             -F "files=@mango2.jpg" \
             -F "top_n=3"
    """
    try:
        # Validate number of files
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch"
            )
        
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
        
        # Load all images
        images = []
        filenames = []
        
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}"
                )
            
            # Read and load image
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} too large (max 10MB)"
                )
            
            try:
                image = Image.open(io.BytesIO(contents))
                images.append(image)
                filenames.append(file.filename)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file {file.filename}: {str(e)}"
                )
        
        # Get detector
        det = get_detector()
        
        # Make batch prediction
        results = det.predict_batch(images, top_n=top_n)
        
        # Add filenames to results
        for result, filename in zip(results, filenames):
            result["filename"] = filename
        
        logger.info(f"Batch prediction completed for {len(files)} images")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "count": len(results),
                "data": results
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/fruits")
async def get_supported_fruits():
    """
    Get list of supported fruits
    
    Returns:
        List of fruits that can be analyzed
    """
    try:
        det = get_detector()
        
        # Extract unique fruits
        fruits = set()
        for class_name in det.labels.values():
            parts = class_name.split('_')
            fruit = parts[-1] if parts else "Unknown"
            fruits.add(fruit)
        
        return {
            "success": True,
            "fruits": sorted(list(fruits)),
            "count": len(fruits)
        }
        
    except Exception as e:
        logger.error(f"Error fetching fruits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diseases")
async def get_diseases_by_fruit(fruit: str = None):
    """
    Get diseases for a specific fruit or all fruits
    
    Args:
        fruit: Fruit name (optional, e.g., "Mango", "Apple")
    
    Returns:
        List of diseases for the specified fruit
    """
    try:
        det = get_detector()
        
        diseases = {}
        
        for class_name in det.labels.values():
            parts = class_name.split('_')
            current_fruit = parts[-1] if parts else "Unknown"
            disease = '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
            
            if fruit is None or current_fruit.lower() == fruit.lower():
                if current_fruit not in diseases:
                    diseases[current_fruit] = []
                
                diseases[current_fruit].append({
                    "name": disease,
                    "full_name": class_name,
                    "is_healthy": "Healthy" in class_name
                })
        
        if fruit and fruit.lower() not in [f.lower() for f in diseases.keys()]:
            raise HTTPException(
                status_code=404,
                detail=f"Fruit '{fruit}' not found"
            )
        
        return {
            "success": True,
            "fruit": fruit,
            "diseases": diseases
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching diseases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router and startup event
__all__ = ['router', 'startup_event']
