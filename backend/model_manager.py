"""
Model Manager for Lazy Loading
================================
Handles lazy loading and caching of ML models to reduce startup memory usage.
Models are loaded only when their endpoints are first called.

Features:
- Lazy initialization of all ML/DL models
- In-memory caching after first load
- Automatic cleanup and garbage collection
- Error handling with fallback behaviors
- Thread-safe operations
"""

import os
import gc
import joblib
import logging
import numpy as np
from typing import Optional, Dict, Any
from threading import Lock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Dict[str, Any] = {}
_cache_lock = Lock()

# Model paths
MODEL_DIR = "model"
MODEL_PATHS = {
    "yield_model": f"{MODEL_DIR}/yield_model.pkl",
    "best_window_model": f"{MODEL_DIR}/best_window_model.pkl",
    "stress_model": f"{MODEL_DIR}/stress_model.pkl",
    "crop_model": f"{MODEL_DIR}/crop_model.pkl",
    "fert_model": f"{MODEL_DIR}/fert_model.pkl",
}

# TensorFlow models (loaded on-demand)
TENSORFLOW_MODELS = {
    "fruit_disease": "model/fruit_disease_prediction_model.h5",
    "plant_disease": "model/plant_disease_prediction_model.h5",
}


def load_joblib_model(model_name: str, model_path: str) -> Optional[Any]:
    """
    Load a joblib model with caching.
    
    Args:
        model_name: Name/key for the model
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading fails
    """
    with _cache_lock:
        # Check if already cached
        if model_name in _model_cache:
            logger.debug(f"[CACHE HIT] {model_name}")
            return _model_cache[model_name]
        
        # Load from disk
        if not os.path.exists(model_path):
            logger.warning(f"[WARN] Model file not found: {model_path}")
            return None
        
        try:
            logger.info(f"[LOADING] {model_name} from {model_path}")
            model = joblib.load(model_path)
            _model_cache[model_name] = model
            logger.info(f"[OK] {model_name} loaded and cached")
            return model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {model_name}: {e}")
            return None


def load_tensorflow_model(model_name: str, model_path: str) -> Optional[Any]:
    """
    Load a TensorFlow model with lazy initialization.
    TensorFlow import is deferred until first call to reduce startup memory.
    
    Args:
        model_name: Name/key for the model
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading fails
    """
    with _cache_lock:
        # Check if already cached
        if model_name in _model_cache:
            logger.debug(f"[CACHE HIT] TensorFlow {model_name}")
            return _model_cache[model_name]
        
        if not os.path.exists(model_path):
            logger.warning(f"[WARN] Model file not found: {model_path}")
            return None
        
        try:
            # Import TensorFlow only when needed
            logger.info(f"[LOADING] TensorFlow {model_name}...")
            import tensorflow as tf
            logger.info(f"[LOADING] {model_name} from {model_path}")
            model = tf.keras.models.load_model(model_path)
            _model_cache[model_name] = model
            logger.info(f"[OK] TensorFlow {model_name} loaded and cached")
            return model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load TensorFlow {model_name}: {e}")
            return None


# ============================================================================
# Public API for Model Access
# ============================================================================

def get_yield_model():
    """Get or load the yield prediction model"""
    return load_joblib_model("yield_model", MODEL_PATHS["yield_model"])


def get_best_window_model():
    """Get or load the best window model"""
    return load_joblib_model("best_window_model", MODEL_PATHS["best_window_model"])


def get_stress_model():
    """Get or load the stress prediction model"""
    return load_joblib_model("stress_model", MODEL_PATHS["stress_model"])


def get_crop_model():
    """Get or load the crop recommendation model"""
    return load_joblib_model("crop_model", MODEL_PATHS["crop_model"])


def get_fert_model():
    """Get or load the fertilizer model"""
    return load_joblib_model("fert_model", MODEL_PATHS["fert_model"])


def get_fruit_disease_model():
    """Get or load the fruit disease detection model"""
    return load_tensorflow_model(
        "fruit_disease",
        TENSORFLOW_MODELS["fruit_disease"]
    )


def get_plant_disease_model():
    """Get or load the plant disease detection model"""
    return load_tensorflow_model(
        "plant_disease",
        TENSORFLOW_MODELS["plant_disease"]
    )


# ============================================================================
# Memory Management
# ============================================================================

def cleanup_model(model_name: str) -> bool:
    """
    Remove a model from cache and free memory.
    
    Args:
        model_name: Name of the model to remove
        
    Returns:
        True if removed, False otherwise
    """
    with _cache_lock:
        if model_name in _model_cache:
            try:
                del _model_cache[model_name]
                gc.collect()
                logger.info(f"[OK] {model_name} removed from cache, memory freed")
                return True
            except Exception as e:
                logger.error(f"[ERROR] Failed to cleanup {model_name}: {e}")
                return False
    return False


def cleanup_all_models() -> None:
    """Remove all models from cache and free memory"""
    with _cache_lock:
        try:
            _model_cache.clear()
            gc.collect()
            logger.info("[OK] All models cleared from cache")
        except Exception as e:
            logger.error(f"[ERROR] Failed to cleanup all models: {e}")


def get_cached_model_names() -> list:
    """Get list of currently cached models"""
    with _cache_lock:
        return list(_model_cache.keys())


def get_model_stats() -> Dict[str, Any]:
    """Get statistics about cached models"""
    with _cache_lock:
        return {
            "cached_models": list(_model_cache.keys()),
            "model_count": len(_model_cache),
            "available_models": {
                "joblib": list(MODEL_PATHS.keys()),
                "tensorflow": list(TENSORFLOW_MODELS.keys()),
            }
        }


# ============================================================================
# Heavy Operations Memory Cleanup
# ============================================================================

def cleanup_after_inference() -> None:
    """
    Clean up memory after inference operations.
    Call this after heavy predictions.
    """
    try:
        gc.collect()
        logger.debug("[MEMORY] Garbage collection completed")
    except Exception as e:
        logger.warning(f"[WARN] Failed to run garbage collection: {e}")


def optimize_array_memory(arr: np.ndarray) -> np.ndarray:
    """
    Optimize numpy array memory usage.
    Convert to lower precision if possible.
    """
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


# ============================================================================
# TensorFlow Configuration
# ============================================================================

def suppress_tensorflow_logging() -> None:
    """
    Suppress verbose TensorFlow startup logs to reduce console spam.
    Call this before importing TensorFlow-dependent modules.
    """
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
        
        # Also configure TensorFlow logging
        import logging as tf_logging
        tf_logging.getLogger('tensorflow').setLevel(tf_logging.ERROR)
        
        logger.info("[OK] TensorFlow logging suppressed")
    except Exception as e:
        logger.warning(f"[WARN] Could not suppress TensorFlow logging: {e}")


# Initialize at module load time
suppress_tensorflow_logging()
