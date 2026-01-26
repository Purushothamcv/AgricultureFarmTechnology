"""
Plant Leaf Disease Detection Service
=====================================
Production-ready service for plant disease classification using trained CNN model.

Features:
- Dynamic class extraction from dataset folder structure
- Efficient model loading at startup
- Image preprocessing matching training pipeline
- Top-K predictions with confidence scores
- Professional error handling
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "model/plant_disease_prediction_model.h5"
DATASET_PATH = "data/plant-village dataset/plantvillage dataset/color"
IMAGE_SIZE = (224, 224)  # Standard ResNet/VGG input size
TOP_K_PREDICTIONS = 3

# ============================================================================
# GLOBAL MODEL AND CLASS MAPPING
# ============================================================================

plant_disease_model: Optional[keras.Model] = None
class_names: List[str] = []
class_mapping: Dict[int, str] = {}

# ============================================================================
# DATASET CLASS EXTRACTION
# ============================================================================

def extract_class_names_from_dataset(dataset_path: str) -> List[str]:
    """
    Dynamically extract class names from dataset folder structure.
    
    Args:
        dataset_path: Path to dataset directory containing class folders
        
    Returns:
        Sorted list of class names (folder names)
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
        ValueError: If no valid class folders found
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Get all subdirectories (each represents a disease class)
    class_folders = [
        d.name for d in dataset_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    if not class_folders:
        raise ValueError(f"No class folders found in: {dataset_path}")
    
    # Sort alphabetically to match model training order
    class_folders.sort()
    
    logger.info(f"📁 Extracted {len(class_folders)} disease classes from dataset")
    logger.info(f"   Classes: {', '.join(class_folders[:5])}{'...' if len(class_folders) > 5 else ''}")
    
    return class_folders


def create_class_mapping(class_names: List[str]) -> Dict[int, str]:
    """
    Create mapping from class index to class name.
    
    Args:
        class_names: List of class names in order
        
    Returns:
        Dictionary mapping index to class name
    """
    return {idx: name for idx, name in enumerate(class_names)}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_plant_disease_model(model_path: str) -> keras.Model:
    """
    Load trained plant disease detection model.
    
    Args:
        model_path: Path to .h5 model file
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        logger.info(f"🔄 Loading plant disease model from: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        
        # Get model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Input shape: {input_shape}")
        logger.info(f"   Output shape: {output_shape}")
        logger.info(f"   Total classes: {output_shape[-1]}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess image to match model training pipeline.
    
    Critical: This must match the preprocessing used during model training!
    
    Args:
        image: PIL Image object
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Convert to RGB (handles grayscale, RGBA, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1] range (most common for ImageDataGenerator)
    img_array = img_array / 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ============================================================================
# NAME CLEANING UTILITIES
# ============================================================================

def clean_crop_name(raw_crop: str) -> str:
    """
    Clean and normalize crop name for display.
    
    Examples:
        "Cherry_(including_sour)" → "Cherry (Including Sour)"
        "Corn_(maize)" → "Corn (Maize)"
        "Pepper,_bell" → "Pepper, Bell"
    
    Args:
        raw_crop: Raw crop name from dataset
        
    Returns:
        Cleaned, human-readable crop name
    """
    # Replace underscores with spaces
    clean = raw_crop.replace('_', ' ')
    
    # Apply title case
    clean = clean.title()
    
    return clean


def clean_disease_name(raw_disease: str) -> str:
    """
    Clean and normalize disease name for display.
    
    Examples:
        "Powdery_mildew" → "Powdery Mildew"
        "Late_blight" → "Late Blight"
        "healthy" → "Healthy"
    
    Args:
        raw_disease: Raw disease name from dataset
        
    Returns:
        Cleaned, human-readable disease name
    """
    # Replace underscores with spaces
    clean = raw_disease.replace('_', ' ')
    
    # Apply title case
    clean = clean.title()
    
    return clean


# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def predict_disease(
    model: keras.Model,
    image_array: np.ndarray,
    class_mapping: Dict[int, str],
    top_k: int = TOP_K_PREDICTIONS
) -> Dict:
    """
    Run model prediction and format results.
    
    Args:
        model: Loaded Keras model
        image_array: Preprocessed image array
        class_mapping: Index to class name mapping
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary containing prediction results
    """
    # Run prediction
    predictions = model.predict(image_array, verbose=0)[0]  # Get first (only) result
    
    # Get top K predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]  # Descending order
    
    # Format top predictions
    top_predictions = [
        {
            "class": class_mapping[int(idx)],
            "confidence": float(predictions[idx])
        }
        for idx in top_indices
    ]
    
    # Get primary prediction
    primary_idx = top_indices[0]
    primary_class = class_mapping[int(primary_idx)]
    primary_confidence = float(predictions[primary_idx])
    
    # Extract raw plant and disease names (everything before/after '___')
    if '___' in primary_class:
        raw_plant = primary_class.split('___')[0]
        raw_disease = primary_class.split('___')[1]
    else:
        raw_plant = "Unknown"
        raw_disease = primary_class
    
    # Clean names for user-friendly display
    clean_plant = clean_crop_name(raw_plant)
    clean_disease = clean_disease_name(raw_disease)
    
    # Determine severity based on CONFIDENCE (not disease keywords)
    if primary_confidence > 0.85:
        severity = "High"
    elif primary_confidence >= 0.70:
        severity = "Moderate"
    else:
        severity = "Low"
    
    # Override for healthy plants
    if 'healthy' in raw_disease.lower():
        severity = "Healthy"
    
    # Add warning for low confidence predictions
    warning = None
    if primary_confidence < 0.70:
        warning = "Low confidence prediction. Consider uploading a clearer image or different angle."
    
    # Format top 3 predictions with cleaned names
    top_3_formatted = []
    for pred in top_predictions[:3]:
        pred_class = pred["class"]
        if '___' in pred_class:
            pred_plant = pred_class.split('___')[0]
            pred_disease = pred_class.split('___')[1]
        else:
            pred_plant = "Unknown"
            pred_disease = pred_class
        
        top_3_formatted.append({
            "crop": clean_crop_name(pred_plant),
            "disease": clean_disease_name(pred_disease),
            "confidence": pred["confidence"]
        })
    
    return {
        "crop": clean_plant,
        "disease": clean_disease,
        "confidence": primary_confidence,
        "severity": severity,
        "warning": warning,
        "top_3": top_3_formatted
    }


# ============================================================================
# FASTAPI ROUTER
# ============================================================================

router = APIRouter(prefix="/predict", tags=["Plant Disease Detection"])


@router.post("/plant-disease", response_model=Dict)
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Predict plant leaf disease from uploaded image.
    
    **Features:**
    - Automatic disease classification
    - Confidence scores for predictions
    - Top 3 most likely diseases
    - Support for multiple plant types
    
    **Supported formats:** JPG, JPEG, PNG
    
    **Example Response:**
    ```json
    {
        "plant": "Tomato",
        "prediction": "Tomato___Late_blight",
        "confidence": 0.91,
        "top_3": [
            {"class": "Tomato___Late_blight", "confidence": 0.91},
            {"class": "Tomato___Early_blight", "confidence": 0.06},
            {"class": "Tomato___healthy", "confidence": 0.03}
        ]
    }
    ```
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
        
    Raises:
        HTTPException: If file validation or prediction fails
    """
    # Check if model is loaded
    if plant_disease_model is None or not class_mapping:
        raise HTTPException(
            status_code=503,
            detail="Plant disease model not initialized. Please restart the server."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file (JPG, PNG)."
        )
    
    try:
        # Read and open image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run prediction
        result = predict_disease(
            model=plant_disease_model,
            image_array=processed_image,
            class_mapping=class_mapping,
            top_k=TOP_K_PREDICTIONS
        )
        
        # Log prediction with cleaned names
        logger.info(f"✅ Prediction: {result['crop']} - {result['disease']} ({result['confidence']:.2%})")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/plant-disease/health")
async def health_check():
    """
    Health check endpoint for plant disease detection service.
    
    Returns:
        Service status and configuration
    """
    return {
        "status": "healthy" if plant_disease_model is not None else "unhealthy",
        "model_loaded": plant_disease_model is not None,
        "total_classes": len(class_names),
        "image_size": IMAGE_SIZE,
        "sample_classes": class_names[:5] if class_names else []
    }


# ============================================================================
# STARTUP EVENT
# ============================================================================

async def startup_event():
    """
    Initialize plant disease detection service on application startup.
    
    This function:
    1. Extracts class names from dataset folder structure
    2. Creates index-to-class mapping
    3. Loads the trained model
    
    Called automatically when FastAPI starts.
    """
    global plant_disease_model, class_names, class_mapping
    
    try:
        logger.info("🌿 Initializing Plant Disease Detection Service...")
        
        # Extract class names from dataset
        class_names = extract_class_names_from_dataset(DATASET_PATH)
        
        # Create class mapping
        class_mapping = create_class_mapping(class_names)
        
        # Load model
        plant_disease_model = load_plant_disease_model(MODEL_PATH)
        
        # Validate model output matches number of classes
        model_classes = plant_disease_model.output_shape[-1]
        dataset_classes = len(class_names)
        
        if model_classes != dataset_classes:
            logger.warning(
                f"⚠️  Model output classes ({model_classes}) != "
                f"Dataset classes ({dataset_classes}). "
                f"This may cause prediction errors!"
            )
        
        logger.info("✅ Plant Disease Detection Service initialized successfully!")
        logger.info(f"   Ready to detect {len(class_names)} disease classes")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Plant Disease Detection: {str(e)}")
        logger.error("   Service will not be available until this is fixed.")
        raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import io  # Import at module level

def get_model_info() -> Dict:
    """Get information about loaded model and classes."""
    if plant_disease_model is None:
        return {"status": "not_loaded"}
    
    return {
        "status": "loaded",
        "model_path": MODEL_PATH,
        "total_classes": len(class_names),
        "input_shape": str(plant_disease_model.input_shape),
        "output_shape": str(plant_disease_model.output_shape),
        "classes": class_names
    }


if __name__ == "__main__":
    # For testing purposes
    import asyncio
    
    async def test_service():
        """Test service initialization"""
        await startup_event()
        print("\n" + "="*60)
        print("Plant Disease Detection Service - Test Results")
        print("="*60)
        print(f"\n✅ Service Status: {'Ready' if plant_disease_model else 'Failed'}")
        print(f"📊 Total Classes: {len(class_names)}")
        print(f"📁 Dataset Path: {DATASET_PATH}")
        print(f"🤖 Model Path: {MODEL_PATH}")
        print(f"\n🏷️  Sample Classes:")
        for i, cls in enumerate(class_names[:10], 1):
            print(f"   {i}. {cls}")
        print("="*60)
    
    asyncio.run(test_service())
