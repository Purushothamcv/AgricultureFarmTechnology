"""
Fertilizer Prediction Service
Provides ML-based fertilizer recommendations using trained model
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


class FertilizerPredictionService:
    """Singleton service for fertilizer predictions"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.encoders = None
        self.label_encoder = None
        self.feature_info = None
        self._initialized = True
        
    def load_model(self):
        """Load trained model and encoders"""
        try:
            # Get the directory where this service file is located
            service_dir = Path(__file__).parent
            model_dir = service_dir / "model"
            
            # Load model
            model_path = model_dir / "fertilizer_model.pkl"
            self.model = joblib.load(model_path)
            
            # Load feature encoders
            encoders_path = model_dir / "fertilizer_encoders.pkl"
            self.encoders = joblib.load(encoders_path)
            
            # Load label encoder
            label_encoder_path = model_dir / "fertilizer_label_encoder.pkl"
            self.label_encoder = joblib.load(label_encoder_path)
            
            # Load feature info
            feature_info_path = model_dir / "fertilizer_feature_info.json"
            import json
            with open(feature_info_path, 'r') as f:
                self.feature_info = json.load(f)
            
            print(f"✅ Fertilizer model loaded successfully")
            print(f"   - Features: {len(self.feature_info['feature_columns'])}")
            print(f"   - Classes: {len(self.feature_info['fertilizer_classes'])}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found. Please train the model first.")
            print(f"   Run: python train_fertilizer_model.py")
            return False
        except Exception as e:
            print(f"❌ Error loading fertilizer model: {e}")
            return False
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input features"""
        required_features = set(self.feature_info['original_features'] + 
                              self.feature_info['numerical_features'])
        
        # Check all required features present
        missing = required_features - set(inputs.keys())
        if missing:
            return False, f"Missing required features: {', '.join(missing)}"
        
        # Validate categorical values
        for feature, encoder in self.encoders.items():
            value = str(inputs.get(feature, ''))
            if value not in encoder.classes_:
                valid_values = ', '.join(encoder.classes_[:5])
                return False, f"Invalid value '{value}' for {feature}. Valid: {valid_values}..."
        
        return True, None
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fertilizer recommendation
        
        Args:
            inputs: Dictionary with all required features
                - Soil_Type: str
                - Soil_pH: float
                - Soil_Moisture: float
                - Organic_Carbon: float
                - Electrical_Conductivity: float
                - Nitrogen_Level: float
                - Phosphorus_Level: float
                - Potassium_Level: float
                - Crop_Type: str
                - Crop_Growth_Stage: str
                - Season: str
                - Temperature: float
                - Humidity: float
                - Rainfall: float
                - Irrigation_Type: str
                - Previous_Crop: str
                - Region: str
                
        Returns:
            Dictionary with:
                - fertilizer: str (recommended fertilizer name)
                - confidence: float (prediction probability 0-1)
                - all_probabilities: dict (all class probabilities)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Prepare feature vector
        feature_values = []
        
        for original_feature in self.feature_info['original_features']:
            # Encode categorical feature
            value = str(inputs[original_feature])
            encoded_value = self.encoders[original_feature].transform([value])[0]
            feature_values.append(encoded_value)
        
        for numerical_feature in self.feature_info['numerical_features']:
            # Add numerical feature as-is
            feature_values.append(float(inputs[numerical_feature]))
        
        # Create feature array
        X = np.array([feature_values])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Decode prediction
        fertilizer = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Get all probabilities
        all_probs = {
            self.label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        # Sort probabilities
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'fertilizer': fertilizer,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 2),
            'all_probabilities': sorted_probs,
            'top_3_recommendations': list(sorted_probs.keys())[:3]
        }
    
    def get_feature_options(self) -> Dict[str, Any]:
        """Get all valid options for categorical features"""
        if not self.encoders:
            raise RuntimeError("Encoders not loaded. Call load_model() first.")
        
        options = {}
        for feature, encoder in self.encoders.items():
            options[feature] = sorted(encoder.classes_.tolist())
        
        return options
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            model_dir = Path("model")
            metrics_path = model_dir / "fertilizer_model_metrics.json"
            
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            return {
                'model_type': metrics.get('model_type', 'RandomForestClassifier'),
                'accuracy': metrics.get('accuracy', 0),
                'accuracy_percentage': round(metrics.get('accuracy', 0) * 100, 2),
                'f1_score': metrics.get('f1_score', 0),
                'n_features': metrics.get('n_features', 0),
                'n_classes': metrics.get('n_classes', 0),
                'fertilizer_classes': self.feature_info['fertilizer_classes']
            }
        except:
            return {
                'model_type': 'RandomForestClassifier',
                'status': 'loaded'
            }


# Global instance
fertilizer_service = FertilizerPredictionService()


def get_fertilizer_service() -> FertilizerPredictionService:
    """Get global fertilizer service instance"""
    return fertilizer_service
