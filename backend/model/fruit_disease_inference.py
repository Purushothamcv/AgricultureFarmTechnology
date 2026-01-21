"""
Fruit Disease Detection - Inference Module
==========================================
Deployment-ready inference for FastAPI integration

This module provides optimized prediction functions for the trained
EfficientNet-B0 fruit disease detection model.

Features:
- Fast single image prediction
- Batch prediction support
- Confidence scores
- Top-N predictions
- Error handling
- Image preprocessing

Author: SmartAgri-AI Team
Date: 2026-01-21
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FruitDiseasePredictor:
    """
    Fruit disease detection predictor class
    
    Handles model loading, preprocessing, and inference
    """
    
    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize predictor with model and labels
        
        Args:
            model_path: Path to trained model (.h5 file)
            labels_path: Path to class labels JSON
        """
        # Default paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, 'model', 'fruit_disease_model.h5')
        if labels_path is None:
            labels_path = os.path.join(base_dir, 'model', 'fruit_disease_labels.json')
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = None
        self.img_height = 224
        self.img_width = 224
        
        # Load labels first
        self._load_labels()
        # Then load model
        self._load_model()
    
    def _load_model(self):
        """Load trained Keras model with multiple fallback strategies"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Strategy 1: Direct load without compilation
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logger.info("✓ Model loaded successfully (Strategy 1: Direct load)")
            except Exception as e1:
                logger.warning(f"Strategy 1 failed: {str(e1)[:100]}")
                
                # Strategy 2: Load with safe_mode=False
                try:
                    import warnings
                    warnings.filterwarnings('ignore')
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        compile=False,
                        safe_mode=False
                    )
                    logger.info("✓ Model loaded successfully (Strategy 2: safe_mode=False)")
                except Exception as e2:
                    logger.warning(f"Strategy 2 failed: {str(e2)[:100]}")
                    
                    # Strategy 3: Load with TF2 SavedModel format
                    try:
                        self.model = tf.saved_model.load(self.model_path)
                        self.model = self.model.signatures['serving_default']
                        logger.info("✓ Model loaded successfully (Strategy 3: SavedModel)")
                    except Exception as e3:
                        logger.warning(f"Strategy 3 failed: {str(e3)[:100]}")
                        
                        # Strategy 4: Manual architecture reconstruction with weight loading
                        logger.info("Attempting Strategy 4: Manual reconstruction...")
                        try:
                            # Build the exact architecture from training
                            base_model = EfficientNetB0(
                                include_top=False,
                                weights=None,  # Don't load ImageNet weights
                                input_shape=(224, 224, 3)
                            )
                            
                            self.model = models.Sequential([
                                base_model,
                                layers.GlobalAveragePooling2D(),
                                layers.Dense(256, activation='relu'),
                                layers.Dropout(0.5),
                                layers.Dense(17, activation='softmax')
                            ])
                            
                            # Build the model by calling it
                            dummy_input = tf.zeros((1, 224, 224, 3))
                            _ = self.model(dummy_input)
                            
                            # Now load weights
                            self.model.load_weights(self.model_path)
                            logger.info("✓ Model loaded successfully (Strategy 4: Manual reconstruction)")
                        except Exception as e4:
                            logger.error(f"All strategies failed. Last error: {e4}")
                            raise Exception("Could not load model with any strategy")
            
            # Model loaded successfully, now prepare for inference
            if not isinstance(self.model, tf.keras.Model):
                # If it's a SavedModel signature, wrap it
                logger.warning("Model is not a Keras model, wrapping for inference")
            
            logger.info("✓ Model ready for inference")
            
        except Exception as e:
            logger.error(f"Fatal error loading model: {e}")
            raise
    
    def _load_labels(self):
        """Load class labels mapping"""
        try:
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels not found at: {self.labels_path}")
            
            with open(self.labels_path, 'r') as f:
                self.labels = json.load(f)
            
            # Convert string keys to integers
            self.labels = {int(k): v for k, v in self.labels.items()}
            logger.info(f"✓ Loaded {len(self.labels)} class labels")
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input using EfficientNet preprocessing
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                img = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                img = image_path
            else:
                raise ValueError("Input must be image path or PIL Image")
            
            # Convert to RGB (in case of RGBA or grayscale)
            img = img.convert('RGB')
            
            # Resize
            img = img.resize((self.img_width, self.img_height))
            
            # Convert to array
            img_array = np.array(img)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Use EfficientNet preprocessing (IMPORTANT)
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path, top_n=3):
        """
        Predict fruit disease from image
        
        Args:
            image_path: Path to image file or PIL Image object
            top_n: Number of top predictions to return
            
        Returns:
            Dictionary containing:
                - predicted_class: Most likely class name
                - confidence: Confidence score (0-1)
                - top_predictions: List of top N predictions with scores
                - all_probabilities: All class probabilities
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get top prediction
            top_idx = np.argmax(predictions)
            predicted_class = self.labels[top_idx]
            confidence = float(predictions[top_idx])
            
            # Get top N predictions
            top_indices = np.argsort(predictions)[::-1][:top_n]
            top_predictions = [
                {
                    'class': self.labels[int(idx)],
                    'confidence': float(predictions[idx]),
                    'percentage': f"{float(predictions[idx]) * 100:.2f}%"
                }
                for idx in top_indices
            ]
            
            # Get fruit type and disease from class name
            fruit_info = self._parse_class_name(predicted_class)
            
            # Prepare result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'fruit_type': fruit_info['fruit'],
                'disease': fruit_info['disease'],
                'is_healthy': fruit_info['is_healthy'],
                'confidence': confidence,
                'confidence_percentage': f"{confidence * 100:.2f}%",
                'top_predictions': top_predictions,
                'all_probabilities': {
                    self.labels[i]: float(predictions[i]) 
                    for i in range(len(predictions))
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, image_paths, top_n=3):
        """
        Predict multiple images in batch
        
        Args:
            image_paths: List of image paths
            top_n: Number of top predictions per image
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, top_n)
            results.append(result)
        return results
    
    def _parse_class_name(self, class_name):
        """
        Parse class name to extract fruit type and disease
        
        Args:
            class_name: Full class name (e.g., "Blotch_Apple")
            
        Returns:
            Dictionary with fruit, disease, and health status
        """
        # Split by underscore
        parts = class_name.split('_')
        
        # Last part is usually the fruit name
        fruit = parts[-1] if parts else "Unknown"
        
        # Everything before is the disease
        disease = '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
        
        # Check if healthy
        is_healthy = 'healthy' in class_name.lower()
        
        if is_healthy:
            disease = "Healthy"
        
        return {
            'fruit': fruit,
            'disease': disease,
            'is_healthy': is_healthy
        }
    
    def get_treatment_recommendation(self, predicted_class):
        """
        Get treatment recommendations based on disease
        
        Args:
            predicted_class: Predicted disease class
            
        Returns:
            Treatment recommendation string
        """
        # Treatment database (can be expanded)
        treatments = {
            # Apple diseases
            'Blotch_Apple': "Apply fungicides like Captan or Mancozeb. Remove infected fruits. Improve air circulation.",
            'Rot_Apple': "Remove infected fruits immediately. Apply copper-based fungicides. Ensure proper drainage.",
            'Scab_Apple': "Apply fungicides during bud break. Remove fallen leaves. Use resistant varieties.",
            'Healthy_Apple': "No treatment needed. Continue regular monitoring and good agricultural practices.",
            
            # Guava diseases
            'Anthracnose_Guava': "Apply Carbendazim or Mancozeb. Remove infected parts. Maintain proper spacing.",
            'Fruitfly_Guava': "Use fruit fly traps. Apply protein bait sprays. Bag fruits for protection.",
            'Healthy_Guava': "No treatment needed. Continue regular monitoring and good agricultural practices.",
            
            # Mango diseases
            'Alternaria_Mango': "Apply Azoxystrobin or Mancozeb. Remove infected leaves. Improve ventilation.",
            'Anthracnose_Mango': "Apply Copper oxychloride. Remove infected fruits. Harvest at proper maturity.",
            'Black Mould Rot (Aspergillus)_Mango': "Apply post-harvest fungicides. Store at low temperature. Handle carefully to avoid injuries.",
            'Stem and Rot (Lasiodiplodia)_Mango': "Remove infected branches. Apply Thiophanate-methyl. Improve tree health.",
            'Healthy_Mango': "No treatment needed. Continue regular monitoring and good agricultural practices.",
            
            # Pomegranate diseases
            'Alternaria_Pomegranate': "Apply Mancozeb or Chlorothalonil. Remove infected parts. Improve drainage.",
            'Anthracnose_Pomegranate': "Apply Carbendazim. Prune infected branches. Maintain proper spacing.",
            'Bacterial_Blight_Pomegranate': "Apply Streptocycline. Remove infected parts. Avoid overhead irrigation.",
            'Cercospora_Pomegranate': "Apply Copper-based fungicides. Remove infected leaves. Improve air circulation.",
            'Healthy_Pomegranate': "No treatment needed. Continue regular monitoring and good agricultural practices."
        }
        
        return treatments.get(predicted_class, 
                             "Consult with agricultural expert for specific treatment recommendations.")
    
    def predict_with_recommendations(self, image_path, top_n=3):
        """
        Predict with treatment recommendations
        
        Args:
            image_path: Path to image file
            top_n: Number of top predictions
            
        Returns:
            Dictionary with predictions and recommendations
        """
        # Get basic prediction
        result = self.predict(image_path, top_n)
        
        # Add treatment recommendation
        if result['success']:
            result['treatment'] = self.get_treatment_recommendation(
                result['predicted_class']
            )
        
        return result


# Convenience function for quick predictions
def predict_fruit_disease(image_path, model_path=None, labels_path=None, top_n=3):
    """
    Quick prediction function
    
    Args:
        image_path: Path to image file
        model_path: Optional custom model path
        labels_path: Optional custom labels path
        top_n: Number of top predictions
        
    Returns:
        Prediction result dictionary
    """
    predictor = FruitDiseasePredictor(model_path, labels_path)
    return predictor.predict_with_recommendations(image_path, top_n)


# Example usage
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print(" "*15 + "FRUIT DISEASE DETECTION - INFERENCE TEST")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = FruitDiseasePredictor()
        print("\n✓ Predictor initialized successfully")
        print(f"✓ Model loaded: {predictor.model_path}")
        print(f"✓ Number of classes: {len(predictor.labels)}")
        
        # Test with image if provided
        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            print(f"\n📸 Testing with image: {test_image}")
            print("-"*70)
            
            result = predictor.predict_with_recommendations(test_image)
            
            if result['success']:
                print(f"\n✅ Prediction: {result['predicted_class']}")
                print(f"🍎 Fruit Type: {result['fruit_type']}")
                print(f"🦠 Disease: {result['disease']}")
                print(f"📊 Confidence: {result['confidence_percentage']}")
                print(f"💊 Treatment: {result['treatment']}")
                
                print("\n📈 Top Predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"   {i}. {pred['class']}: {pred['percentage']}")
            else:
                print(f"\n❌ Prediction failed: {result['error']}")
        else:
            print("\n💡 Usage: python fruit_disease_inference.py <image_path>")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
