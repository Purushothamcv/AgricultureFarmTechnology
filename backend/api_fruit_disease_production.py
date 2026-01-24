"""
FRUIT DISEASE DETECTION - FASTAPI PRODUCTION ENDPOINT
======================================================
Interview-Ready | Production-Grade | RESTful API

This module provides FastAPI endpoints for fruit disease detection
using the frozen EfficientNet-B0 model trained to 91-92% accuracy.

ENDPOINTS
=========
POST /api/fruit-disease/predict - Single image prediction
POST /api/fruit-disease/predict-batch - Batch prediction
GET /api/fruit-disease/classes - Available disease classes
GET /api/fruit-disease/stats - Model statistics
GET /api/fruit-disease/health - Service health check

FEATURES
========
- Production-ready error handling
- Input validation
- Performance tracking
- CORS enabled
- Structured JSON responses
- Multipart file upload support

Author: SmartAgri-AI Team
Date: January 22, 2026
"""

import io
import time
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from pathlib import Path

# Import production inference engine
from model.production_inference import (
    get_inference_engine,
    predict_fruit_disease,
    predict_fruit_disease_batch
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router with prefix and tags
router = APIRouter(
    prefix="/api/fruit-disease",
    tags=["Fruit Disease Detection (Production)"]
)

# Global engine instance
_engine = None


def initialize_engine():
    """Initialize inference engine at startup"""
    global _engine
    try:
        logger.info("🚀 Initializing Fruit Disease Detection Engine...")
        _engine = get_inference_engine()
        logger.info("✅ Engine initialized successfully")
        logger.info(f"   - Model: EfficientNet-B0")
        logger.info(f"   - Classes: {_engine.num_classes}")
        logger.info(f"   - Status: Frozen (Inference-Only)")
        return True
    except Exception as e:
        logger.error(f"❌ Engine initialization failed: {e}")
        return False


def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If validation fails
    """
    # Check file exists
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check content type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    # Check file size (max 10MB)
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file.size} bytes. Maximum: {MAX_SIZE} bytes (10MB)"
        )


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Load PIL Image from uploaded file
    
    Args:
        file: Uploaded file
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If image loading fails
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns service status and model information
    """
    try:
        if _engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Inference engine not initialized"
                }
            )
        
        stats = _engine.get_statistics()
        
        return {
            "status": "healthy",
            "service": "Fruit Disease Detection",
            "model": {
                "architecture": "EfficientNet-B0",
                "status": "frozen (inference-only)",
                "training_accuracy": "95%",
                "validation_accuracy": "91-92%",
                "num_classes": _engine.num_classes,
                "trained_at_epoch": 29
            },
            "performance": {
                "total_predictions": stats['total_predictions'],
                "average_inference_time_ms": stats['average_inference_time_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/classes")
async def get_available_classes():
    """
    Get all available disease classes organized by fruit
    
    Returns:
        Dictionary mapping fruits to disease lists
    """
    try:
        if _engine is None:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        classes = _engine.get_available_classes()
        
        return {
            "success": True,
            "total_classes": _engine.num_classes,
            "classes_by_fruit": classes,
            "fruits": list(classes.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get classes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve classes: {str(e)}"
        )


@router.get("/stats")
async def get_model_statistics():
    """
    Get model statistics and performance metrics
    
    Returns:
        Model statistics and performance data
    """
    try:
        if _engine is None:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        stats = _engine.get_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "training_story": {
                "architecture": "EfficientNet-B0",
                "training_strategy": "Two-phase transfer learning",
                "phase_1": "Feature extraction (frozen backbone, 30 epochs)",
                "phase_2": "Fine-tuning (unfrozen top layers, stopped at epoch 29)",
                "best_epoch": 29,
                "reason_stopped": "Catastrophic forgetting after epoch 29",
                "validation_accuracy_peak": "91-92%",
                "current_status": "Frozen for production deployment"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    top_n: int = Form(default=3)
):
    """
    Predict fruit disease from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, WebP)
        top_n: Number of top predictions to return (default: 3)
        
    Returns:
        Prediction result with:
        - predicted_class: Top prediction
        - confidence: Confidence score (0-1)
        - top_predictions: List of top N predictions
        - inference_time_ms: Processing time
        
    Example Response:
    {
        "success": true,
        "predicted_class": "Alternaria_Mango",
        "confidence": 0.94,
        "top_predictions": [
            {"class": "Alternaria_Mango", "confidence": 0.94},
            {"class": "Anthracnose_Mango", "confidence": 0.04},
            {"class": "Healthy_Mango", "confidence": 0.01}
        ],
        "inference_time_ms": 87.5,
        "model_info": {
            "architecture": "EfficientNet-B0",
            "validation_accuracy": "91-92%",
            "frozen": true
        }
    }
    """
    try:
        # Validate engine
        if _engine is None:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized. Please try again later."
            )
        
        # Validate input
        validate_image(file)
        
        # Validate top_n parameter
        if not 1 <= top_n <= _engine.num_classes:
            raise HTTPException(
                status_code=400,
                detail=f"top_n must be between 1 and {_engine.num_classes}"
            )
        
        # Load image
        image = await load_image_from_upload(file)
        
        # Run prediction
        start_time = time.time()
        result = predict_fruit_disease(image, top_n=top_n)
        total_time = (time.time() - start_time) * 1000
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'total_processing_time_ms': round(total_time, 2)
        }
        
        logger.info(
            f"Prediction: {result['predicted_class']} "
            f"(confidence: {result['confidence']:.2%}) "
            f"in {total_time:.2f}ms"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict-batch")
async def predict_disease_batch(
    files: List[UploadFile] = File(...),
    top_n: int = Form(default=3)
):
    """
    Batch prediction for multiple images
    
    Args:
        files: List of image files
        top_n: Number of top predictions per image
        
    Returns:
        List of prediction results
        
    Example Response:
    {
        "success": true,
        "batch_size": 5,
        "results": [
            {
                "filename": "mango1.jpg",
                "predicted_class": "Alternaria_Mango",
                "confidence": 0.94,
                "top_predictions": [...]
            },
            ...
        ],
        "total_time_ms": 425.5,
        "average_time_per_image_ms": 85.1
    }
    """
    try:
        # Validate engine
        if _engine is None:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        # Validate batch size
        MAX_BATCH_SIZE = 20
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files uploaded"
            )
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum: {MAX_BATCH_SIZE}"
            )
        
        # Validate top_n
        if not 1 <= top_n <= _engine.num_classes:
            raise HTTPException(
                status_code=400,
                detail=f"top_n must be between 1 and {_engine.num_classes}"
            )
        
        # Start timing
        start_time = time.time()
        
        # Load all images
        images = []
        filenames = []
        
        for file in files:
            validate_image(file)
            image = await load_image_from_upload(file)
            images.append(image)
            filenames.append(file.filename)
        
        # Run batch prediction
        results = predict_fruit_disease_batch(images, top_n=top_n)
        
        # Add filenames to results
        for result, filename in zip(results, filenames):
            result['filename'] = filename
        
        # Calculate timing
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(files)
        
        logger.info(
            f"Batch prediction: {len(files)} images in {total_time:.2f}ms "
            f"({avg_time:.2f}ms/image)"
        )
        
        return {
            "success": True,
            "batch_size": len(files),
            "results": results,
            "timing": {
                "total_time_ms": round(total_time, 2),
                "average_time_per_image_ms": round(avg_time, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================
# STARTUP EVENT HANDLER
# ============================================

async def startup_event():
    """Initialize engine on FastAPI startup"""
    success = initialize_engine()
    if not success:
        logger.error("⚠️  Fruit Disease Detection engine failed to initialize")
    return success


# ============================================
# INTERVIEW TALKING POINTS - API DESIGN
# ============================================
"""
API DESIGN DECISIONS

1. RESTful Design
   - Clear resource naming (/fruit-disease)
   - HTTP methods match operations (POST for predictions, GET for info)
   - Structured JSON responses

2. Error Handling
   - Input validation (file type, size, parameters)
   - Descriptive error messages
   - Appropriate HTTP status codes
   - Graceful degradation

3. Performance
   - Single model load at startup (not per request)
   - Batch processing support
   - Response time tracking
   - Efficient image preprocessing

4. Production Features
   - Health check endpoint
   - Statistics endpoint
   - CORS enabled for frontend integration
   - File size limits (10MB)
   - Batch size limits (20 images)

5. Documentation
   - OpenAPI/Swagger automatic documentation
   - Clear endpoint descriptions
   - Example responses
   - Parameter descriptions
"""
