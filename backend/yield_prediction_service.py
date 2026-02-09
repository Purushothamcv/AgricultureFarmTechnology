"""
Yield Prediction Service
Production-ready service for yield prediction using APY-trained model
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class YieldPredictionService:
    """Service for predicting crop yields based on APY dataset model"""
    
    def __init__(
        self,
        model_path: str = "model/yield_prediction_model.pkl",
        encoders_path: str = "model/yield_encoders.pkl",
        feature_info_path: str = "model/yield_feature_info.json",
        metrics_path: str = "model/yield_model_metrics.json"
    ):
        """Initialize service and load model artifacts"""
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.feature_info_path = feature_info_path
        self.metrics_path = metrics_path
        
        self.model = None
        self.encoders = {}
        self.feature_info = {}
        self.metrics = {}
        self.is_loaded = False
        
    def load_model(self):
        """Load trained model and encoders"""
        try:
            print(f"🔄 Loading Yield Prediction model...")
            
            # Load model
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f"  ✅ Model loaded from {self.model_path}")
            
            # Load encoders
            if not Path(self.encoders_path).exists():
                raise FileNotFoundError(f"Encoders file not found: {self.encoders_path}")
            self.encoders = joblib.load(self.encoders_path)
            print(f"  ✅ Encoders loaded: {list(self.encoders.keys())}")
            
            # Load feature info
            if Path(self.feature_info_path).exists():
                with open(self.feature_info_path, 'r') as f:
                    self.feature_info = json.load(f)
                print(f"  ✅ Feature info loaded")
            
            # Load metrics
            if Path(self.metrics_path).exists():
                with open(self.metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                print(f"  ✅ Model metrics:")
                print(f"     - Type: {self.metrics.get('model_type', 'Unknown')}")
                print(f"     - R² Score: {self.metrics.get('r2_score', 0):.4f}")
                print(f"     - RMSE: {self.metrics.get('rmse', 0):.4f}")
                print(f"     - MAE: {self.metrics.get('mae', 0):.4f}")
            
            self.is_loaded = True
            print(f"✅ Yield Prediction Service ready!")
            
        except Exception as e:
            print(f"❌ Failed to load yield prediction model: {e}")
            raise
    
    def get_available_values(self) -> Dict[str, List[str]]:
        """Get available values for categorical features"""
        if not self.is_loaded:
            self.load_model()
        
        available = {}
        for feature, encoder in self.encoders.items():
            available[feature] = sorted(encoder.classes_.tolist())
        
        return available
    
    def get_districts_by_state(self, state: str) -> List[str]:
        """
        Get districts filtered by selected state from APY dataset
        
        Args:
            state: State name to filter districts
        
        Returns:
            List of district names in the selected state
        """
        if not self.is_loaded:
            self.load_model()
        
        # Load APY dataset to filter districts by state
        try:
            import pandas as pd
            df = pd.read_csv("data/APY.csv")
            df.columns = df.columns.str.strip()
            df['State'] = df['State'].str.strip()
            df['District'] = df['District'].str.strip()
            
            # Filter districts by state
            districts = df[df['State'] == state]['District'].unique().tolist()
            return sorted(districts)
        except Exception as e:
            print(f"⚠️  Warning: Could not load districts for state {state}: {e}")
            # Fallback: return all districts
            return sorted(self.encoders.get('District', {}).classes_.tolist())
    
    def validate_input(
        self,
        state: str,
        district: str,
        crop: str,
        season: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input values against known categories
        
        Returns:
            (is_valid, error_message)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Check State
        if state not in self.encoders['State'].classes_:
            available_states = sorted(self.encoders['State'].classes_.tolist())[:10]
            return False, f"Invalid State '{state}'. Try one of: {', '.join(available_states)}..."
        
        # Check District
        if district not in self.encoders['District'].classes_:
            # Try to suggest districts from the same state
            return False, f"Invalid District '{district}'. Please check district name."
        
        # Check Crop
        if crop not in self.encoders['Crop'].classes_:
            available_crops = sorted(self.encoders['Crop'].classes_.tolist())[:15]
            return False, f"Invalid Crop '{crop}'. Try one of: {', '.join(available_crops)}..."
        
        # Check Season
        if season not in self.encoders['Season'].classes_:
            available_seasons = sorted(self.encoders['Season'].classes_.tolist())
            return False, f"Invalid Season '{season}'. Try one of: {', '.join(available_seasons)}"
        
        return True, None
    
    def predict_yield(
        self,
        state: str,
        district: str,
        crop: str,
        year: int,
        season: str,
        area: float
    ) -> Dict[str, Any]:
        """
        Predict crop yield for given parameters
        
        Args:
            state: State name
            district: District name
            crop: Crop name
            year: Crop year (e.g., 2024)
            season: Season name
            area: Area in hectares
        
        Returns:
            Dictionary containing:
                - predicted_yield: Predicted yield value
                - confidence: Model confidence (R² score)
                - unit: Unit of measurement
                - input_values: Echo of input parameters
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Clean inputs
            state = state.strip()
            district = district.strip()
            crop = crop.strip()
            season = season.strip()
            
            # Validate inputs
            is_valid, error_message = self.validate_input(state, district, crop, season)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_message,
                    'predicted_yield': None
                }
            
            # Encode categorical features
            state_encoded = int(self.encoders['State'].transform([state])[0])
            district_encoded = int(self.encoders['District'].transform([district])[0])
            crop_encoded = int(self.encoders['Crop'].transform([crop])[0])
            season_encoded = int(self.encoders['Season'].transform([season])[0])
            
            # Prepare feature vector
            # Order must match training: State, District, Crop, Crop_Year, Season, Area
            features = pd.DataFrame([[
                state_encoded,
                district_encoded,
                crop_encoded,
                year,
                season_encoded,
                area
            ]], columns=[
                'State_encoded',
                'District_encoded',
                'Crop_encoded',
                'Crop_Year',
                'Season_encoded',
                'Area'
            ])
            
            # Make prediction
            predicted_yield = float(self.model.predict(features)[0])
            
            # Get confidence (R² score from training)
            confidence = self.metrics.get('r2_score', 0.0)
            
            # Calculate total production estimate
            estimated_production = predicted_yield * area
            
            return {
                'success': True,
                'predicted_yield': round(predicted_yield, 2),
                'confidence': round(confidence, 4),
                'unit': 'tonnes/hectare',
                'estimated_production': round(estimated_production, 2),
                'production_unit': 'tonnes',
                'model_type': self.metrics.get('model_type', 'Unknown'),
                'input_values': {
                    'state': state,
                    'district': district,
                    'crop': crop,
                    'year': year,
                    'season': season,
                    'area': area
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}",
                'predicted_yield': None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            self.load_model()
        
        return {
            'model_type': self.metrics.get('model_type', 'Unknown'),
            'r2_score': self.metrics.get('r2_score', 0),
            'rmse': self.metrics.get('rmse', 0),
            'mae': self.metrics.get('mae', 0),
            'train_samples': self.metrics.get('train_samples', 0),
            'test_samples': self.metrics.get('test_samples', 0),
            'features': self.feature_info.get('feature_columns', []),
            'available_states': len(self.encoders.get('State', {}).classes_) if 'State' in self.encoders else 0,
            'available_districts': len(self.encoders.get('District', {}).classes_) if 'District' in self.encoders else 0,
            'available_crops': len(self.encoders.get('Crop', {}).classes_) if 'Crop' in self.encoders else 0,
            'available_seasons': len(self.encoders.get('Season', {}).classes_) if 'Season' in self.encoders else 0
        }


# Global service instance (singleton pattern)
_yield_service = None


def get_yield_service() -> YieldPredictionService:
    """Get or create yield prediction service instance"""
    global _yield_service
    if _yield_service is None:
        _yield_service = YieldPredictionService()
        try:
            _yield_service.load_model()
        except Exception as e:
            print(f"⚠️  Warning: Could not load yield model on startup: {e}")
            print(f"   Model will be loaded on first prediction request")
    return _yield_service


# Startup event for FastAPI
async def startup_event():
    """Initialize yield prediction service on application startup"""
    print("🌾 Initializing Yield Prediction Service...")
    try:
        service = get_yield_service()
        if service.is_loaded:
            print("✅ Yield Prediction Service initialized successfully")
        else:
            service.load_model()
    except Exception as e:
        print(f"⚠️  Yield Prediction Service initialization warning: {e}")
        print(f"   Service will attempt to load model on first request")
