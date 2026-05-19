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
    DEFAULT_YIELD_MIN = 0.0
    
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
        self._apy_df: Optional[pd.DataFrame] = None

    def _load_apy_dataset(self) -> pd.DataFrame:
        """Load and cache APY dataset used for state-based dropdown filters."""
        if self._apy_df is None:
            df = pd.read_csv("data/APY.csv")
            df.columns = df.columns.str.strip()

            # Clean categorical fields for reliable matching.
            for col in ['State', 'District', 'Crop']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)

            # Drop rows without crop values and enforce consistent casing.
            if 'Crop' in df.columns:
                df = df.dropna(subset=['Crop'])
                df = df[df['Crop'].astype(str).str.strip() != '']

            if 'State' in df.columns:
                df['State'] = df['State'].astype(str).str.strip().str.title()

            if 'Crop' in df.columns:
                df['Crop'] = df['Crop'].astype(str).str.strip().str.title()

            if 'District' in df.columns:
                df['District'] = df['District'].astype(str).str.strip()

            if 'Yield' in df.columns:
                df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
                df = df.dropna(subset=['Yield'])
                df = df[df['Yield'] >= 0]

            self._apy_df = df

        return self._apy_df

    def _sanitize_predicted_yield(self, predicted_yield: float) -> float:
        """Guarantee non-negative yield output without hard upper capping."""
        return float(max(float(predicted_yield), self.DEFAULT_YIELD_MIN))

    def _dataset_median_and_p95(self, state: str, district: str, crop: str) -> Tuple[float, float, str]:
        """
        Resolve dataset median and p95 for smart correction.

        Fallback hierarchy:
        1) state + district + crop
        2) state-level
        3) crop-level
        4) global
        """
        df = self._load_apy_dataset()

        state_l = (state or '').strip().lower()
        district_l = (district or '').strip().lower()
        crop_l = (crop or '').strip().lower()

        def _stats(mask: pd.Series) -> Tuple[Optional[float], Optional[float]]:
            subset = df.loc[mask, 'Yield']
            if subset.empty:
                return None, None
            median_val = float(subset.median())
            p95_val = float(subset.quantile(0.95))
            if not np.isfinite(median_val) or not np.isfinite(p95_val):
                return None, None
            return median_val, p95_val

        median_val, p95_val = _stats(
            (df['State'].str.lower() == state_l) &
            (df['District'].str.lower() == district_l) &
            (df['Crop'].str.lower() == crop_l)
        )
        if median_val is not None:
            return median_val, p95_val if p95_val is not None else median_val, 'state_district_crop'

        median_val, p95_val = _stats(df['State'].str.lower() == state_l)
        if median_val is not None:
            return median_val, p95_val if p95_val is not None else median_val, 'state'

        median_val, p95_val = _stats(df['Crop'].str.lower() == crop_l)
        if median_val is not None:
            return median_val, p95_val if p95_val is not None else median_val, 'crop'

        median_val, p95_val = _stats(df['Yield'] >= 0)
        if median_val is not None:
            return median_val, p95_val if p95_val is not None else median_val, 'global'

        return 0.0, 0.0, 'none'
        
    def load_model(self):
        """Load trained model and encoders"""
        try:
            print(f"🔄 Loading Yield Prediction model...")
            
            # Load model
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f"  [SUCCESS] Model loaded from {self.model_path}")
            
            # Load encoders
            if not Path(self.encoders_path).exists():
                raise FileNotFoundError(f"Encoders file not found: {self.encoders_path}")
            self.encoders = joblib.load(self.encoders_path)
            print(f"  [SUCCESS] Encoders loaded: {list(self.encoders.keys())}")
            
            # Load feature info
            if Path(self.feature_info_path).exists():
                with open(self.feature_info_path, 'r') as f:
                    self.feature_info = json.load(f)
                print(f"  [SUCCESS] Feature info loaded")
            
            # Load metrics
            if Path(self.metrics_path).exists():
                with open(self.metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                print(f"  [SUCCESS] Model metrics:")
                print(f"     - Type: {self.metrics.get('model_type', 'Unknown')}")
                print(f"     - R² Score: {self.metrics.get('r2_score', 0):.4f}")
                print(f"     - RMSE: {self.metrics.get('rmse', 0):.4f}")
                print(f"     - MAE: {self.metrics.get('mae', 0):.4f}")

            # Load APY data at startup to avoid first-request latency.
            try:
                self._load_apy_dataset()
                print("  [SUCCESS] APY dataset loaded for state/crop filters")
            except FileNotFoundError:
                print("  [WARN]️ APY.csv not found; using encoder-based fallback for dropdown values")
            except Exception as dataset_error:
                print(f"  [WARN]️ APY dataset could not be loaded: {dataset_error}")
                print("  [WARN]️ Continuing with encoder-based fallback for dropdown values")
            
            self.is_loaded = True
            print(f"[SUCCESS] Yield Prediction Service ready!")
            
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
            df = self._load_apy_dataset()
            
            # Filter districts by state
            districts = (
                df[df['State'] == state]['District']
                .dropna()
                .replace('', np.nan)
                .dropna()
                .unique()
                .tolist()
            )
            return sorted(districts)
        except Exception as e:
            print(f"[WARN]️  Warning: Could not load districts for state {state}: {e}")
            # Fallback: return all districts
            district_encoder = self.encoders.get('District')
            if district_encoder is not None:
                return sorted(district_encoder.classes_.tolist())
            return []

    def get_crops_by_state(self, state: str) -> List[str]:
        """
        Get crops filtered by selected state from APY dataset.

        Args:
            state: State name to filter crops

        Returns:
            Alphabetically sorted unique crop names
        """
        if not self.is_loaded:
            self.load_model()

        try:
            df = self._load_apy_dataset()
            normalized_state = state.strip().title()

            print(f"🌾 /yield/crops requested state: {state}")

            crops = (
                df[df['State'].str.lower() == normalized_state.lower()]['Crop']
                .dropna()
                .replace('', np.nan)
                .dropna()
                .unique()
                .tolist()
            )
            crops_list = sorted(crops)
            print(f"🌾 /yield/crops found: {len(crops_list)} crops")
            return crops_list
        except Exception as e:
            print(f"[WARN]️  Warning: Could not load crops for state {state}: {e}")
            return []
    
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

    def _canonicalize_encoder_value(self, feature: str, value: str) -> str:
        """Map user input to exact encoder class label using case-insensitive matching."""
        if feature not in self.encoders:
            return value

        cleaned = (value or '').strip()
        if not cleaned:
            return cleaned

        classes = self.encoders[feature].classes_.tolist()
        if cleaned in classes:
            return cleaned

        lookup = {str(item).strip().lower(): item for item in classes}
        return lookup.get(cleaned.lower(), cleaned)
    
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

            # Normalize to canonical encoder labels so case/style mismatches
            # (e.g., "Castor Seed" vs "Castor seed") do not fail validation.
            state = self._canonicalize_encoder_value('State', state)
            district = self._canonicalize_encoder_value('District', district)
            crop = self._canonicalize_encoder_value('Crop', crop)
            season = self._canonicalize_encoder_value('Season', season)

            if area <= 0:
                return {
                    'success': False,
                    'error': "Area must be greater than 0",
                    'predicted_yield': None
                }

            if year < 1900 or year > 2100:
                return {
                    'success': False,
                    'error': "Year must be between 1900 and 2100",
                    'predicted_yield': None
                }
            
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

            print(
                "🌾 Yield encoded values:",
                {
                    'state_encoded': state_encoded,
                    'district_encoded': district_encoded,
                    'crop_encoded': crop_encoded,
                    'season_encoded': season_encoded,
                }
            )
            
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
            print("🌾 Yield feature vector:", features.to_dict(orient='records')[0])
            
            # Make prediction
            raw_predicted_yield = float(self.model.predict(features)[0])

            dataset_median, dataset_p95, fallback_level = self._dataset_median_and_p95(state, district, crop)

            unrealistic = (
                (not np.isfinite(raw_predicted_yield)) or
                (raw_predicted_yield < 0) or
                (dataset_p95 > 0 and raw_predicted_yield > dataset_p95)
            )

            if unrealistic:
                predicted_yield = self._sanitize_predicted_yield(dataset_median)
                source = 'dataset_median'
                adjusted = True
                reason = 'replaced with dataset median due unrealistic model output'
            else:
                predicted_yield = self._sanitize_predicted_yield(raw_predicted_yield)
                source = 'model'
                adjusted = False
                reason = None

            print(
                "🌾 Yield prediction source:",
                {
                    'source': source,
                    'raw_predicted_yield': raw_predicted_yield,
                    'final_predicted_yield': predicted_yield,
                    'fallback_level': fallback_level,
                    'dataset_p95': dataset_p95,
                    'adjusted': adjusted,
                }
            )
            
            # Get confidence (R² score from training)
            confidence = self.metrics.get('r2_score', 0.0)
            
            # Calculate total production estimate
            estimated_production = predicted_yield * area
            
            return {
                'success': True,
                'predicted_yield': round(predicted_yield, 2),
                'raw_predicted_yield': round(raw_predicted_yield, 2),
                'source': source,
                'fallback_level': fallback_level,
                'was_corrected': adjusted,
                'adjusted': adjusted,
                'reason': reason,
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
            print(f"[WARN]️  Warning: Could not load yield model on startup: {e}")
            print(f"   Model will be loaded on first prediction request")
    return _yield_service


# Startup event for FastAPI
async def startup_event():
    """Initialize yield prediction service on application startup"""
    print("🌾 Initializing Yield Prediction Service...")
    try:
        service = get_yield_service()
        if service.is_loaded:
            print("[SUCCESS] Yield Prediction Service initialized successfully")
        else:
            service.load_model()
    except Exception as e:
        print(f"[WARN]️  Yield Prediction Service initialization warning: {e}")
        print(f"   Service will attempt to load model on first request")
