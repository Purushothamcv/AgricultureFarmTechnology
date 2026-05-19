"""
Stress Prediction Service
Simplified model using only farmer-friendly inputs and auto-fetchable data
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os

class StressPredictionService:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.features = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained stress prediction model"""
        try:
            model_dir = Path(__file__).parent / 'model'
            
            model_path = model_dir / 'stress_prediction_model.pkl'
            encoders_path = model_dir / 'stress_label_encoders.pkl'
            features_path = model_dir / 'stress_features.pkl'
            
            if not model_path.exists():
                print(f"[WARN]️ Stress model not found at {model_path}")
                print("Please run: python train_stress_model.py")
                return False
                
            self.model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            self.features = joblib.load(features_path)
            self.model_loaded = True
            
            print("[SUCCESS] Stress prediction model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading stress model: {e}")
            return False
    
    def get_options(self):
        """Get dropdown options for frontend"""
        return {
            'crop_types': [
                'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 
                'Soybean', 'Tomato', 'Potato', 'Onion', 'Chilli'
            ],
            'growth_stages': [
                'Germination', 'Vegetative', 'Flowering', 
                'Fruiting', 'Maturity'
            ]
        }
    
    def predict(self, input_data: dict):
        """
        Predict stress level from input data
        
        Args:
            input_data: Dictionary with all required features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            if not self.load_model():
                return {
                    'success': False,
                    'error': 'Model not loaded. Please train the model first.'
                }
        
        try:
            # Prepare input features
            features_dict = {
                # Manual Farmer Inputs
                'Crop_Type': input_data.get('crop_type', 'Rice'),
                'Crop_Growth_Stage': input_data.get('growth_stage', 'Vegetative'),
                'Soil_Moisture': float(input_data.get('soil_moisture', 50)),
                'Soil_pH': float(input_data.get('soil_ph', 7.0)),
                'Organic_Matter': float(input_data.get('organic_matter', 3.0)),
                'Pest_Damage': float(input_data.get('pest_damage', 0)),
                'Weed_Coverage': float(input_data.get('weed_coverage', 0)),
                
                # Auto-Fetch from APIs
                'Temperature': float(input_data.get('temperature', 25)),
                'Humidity': float(input_data.get('humidity', 60)),
                'Rainfall': float(input_data.get('rainfall', 50)),
                'Wind_Speed': float(input_data.get('wind_speed', 10)),
                'Elevation_Data': float(input_data.get('elevation', 500)),
                'Water_Flow': float(input_data.get('water_flow', 50)),
                'Drainage_Features': float(input_data.get('drainage', 70)),
            }
            
            # Create DataFrame
            df = pd.DataFrame([features_dict])
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unknown categories
                    if df[col].iloc[0] not in encoder.classes_:
                        # Use the most common class (first one)
                        df[col] = encoder.transform([encoder.classes_[0]])
                    else:
                        df[col] = encoder.transform(df[col])
            
            # Ensure correct column order
            df = df[self.features]
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]
            
            # Get confidence for predicted class
            class_index = list(self.model.classes_).index(prediction)
            confidence = probabilities[class_index]
            
            # Generate advice
            advice = self._generate_advice(prediction, features_dict)
            
            # Identify stress factors
            stress_factors = self._identify_stress_factors(features_dict)
            
            return {
                'success': True,
                'stress_level': prediction,
                'confidence': float(confidence),
                'confidence_percentage': f"{confidence * 100:.1f}%",
                'advice': advice,
                'stress_factors': stress_factors,
                'recommendations': self._get_recommendations(prediction, stress_factors)
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _identify_stress_factors(self, data: dict):
        """Identify factors contributing to stress"""
        factors = []
        
        # Temperature stress
        temp = data['Temperature']
        if temp > 38:
            factors.append("Extreme high temperature")
        elif temp > 35:
            factors.append("High temperature")
        elif temp < 18:
            factors.append("Low temperature")
        
        # Soil moisture stress
        moisture = data['Soil_Moisture']
        if moisture < 30:
            factors.append("Severe drought stress")
        elif moisture < 40:
            factors.append("Moderate drought stress")
        elif moisture > 80:
            factors.append("Waterlogging risk")
        
        # Rainfall
        rainfall = data['Rainfall']
        if rainfall < 20:
            factors.append("Insufficient rainfall")
        elif rainfall > 120:
            factors.append("Excessive rainfall")
        
        # Humidity
        humidity = data['Humidity']
        if humidity < 35:
            factors.append("Low humidity stress")
        elif humidity > 85:
            factors.append("High humidity (disease risk)")
        
        # Soil pH
        ph = data['Soil_pH']
        if ph < 6.0:
            factors.append("Acidic soil")
        elif ph > 7.5:
            factors.append("Alkaline soil")
        
        # Organic matter
        if data['Organic_Matter'] < 2.5:
            factors.append("Low organic matter")
        
        # Pest damage
        pest = data['Pest_Damage']
        if pest > 30:
            factors.append("Severe pest damage")
        elif pest > 15:
            factors.append("Moderate pest damage")
        
        # Weed coverage
        weed = data['Weed_Coverage']
        if weed > 25:
            factors.append("High weed competition")
        elif weed > 15:
            factors.append("Moderate weed coverage")
        
        # Wind
        if data['Wind_Speed'] > 25:
            factors.append("High wind stress")
        
        # Drainage
        if data['Drainage_Features'] < 40:
            factors.append("Poor drainage")
        
        if not factors:
            factors.append("Optimal growing conditions")
        
        return factors
    
    def _generate_advice(self, stress_level: str, data: dict):
        """Generate advice based on stress level"""
        if stress_level == 'Low':
            return "Crops are in good health. Continue current management practices and monitor regularly."
        
        elif stress_level == 'Moderate':
            advice_parts = []
            
            if data['Soil_Moisture'] < 40:
                advice_parts.append("Increase irrigation frequency")
            if data['Temperature'] > 35:
                advice_parts.append("Provide shade or mulching")
            if data['Pest_Damage'] > 15:
                advice_parts.append("Apply appropriate pesticides")
            if data['Weed_Coverage'] > 15:
                advice_parts.append("Implement weed control measures")
            if data['Soil_pH'] < 6.0:
                advice_parts.append("Apply lime to raise soil pH")
            elif data['Soil_pH'] > 7.5:
                advice_parts.append("Add organic matter to lower pH")
            
            if advice_parts:
                return "Moderate stress detected. Actions needed: " + ", ".join(advice_parts) + "."
            else:
                return "Moderate stress detected. Monitor closely and adjust management practices."
        
        else:  # High stress
            advice_parts = []
            
            if data['Soil_Moisture'] < 30:
                advice_parts.append("URGENT: Increase irrigation immediately")
            if data['Temperature'] > 38:
                advice_parts.append("Provide immediate cooling measures")
            if data['Pest_Damage'] > 30:
                advice_parts.append("Apply pest control urgently")
            if data['Weed_Coverage'] > 25:
                advice_parts.append("Remove weeds immediately")
            
            if advice_parts:
                return "HIGH STRESS! Immediate action required: " + ", ".join(advice_parts) + "."
            else:
                return "HIGH STRESS! Immediate intervention required. Consider consulting an agricultural expert."
    
    def _get_recommendations(self, stress_level: str, factors: list):
        """Get specific recommendations based on stress factors"""
        recommendations = []
        
        for factor in factors:
            if "temperature" in factor.lower():
                recommendations.append({
                    'factor': 'Temperature',
                    'action': 'Apply mulch, provide shade, or adjust planting schedule'
                })
            elif "drought" in factor.lower() or "rainfall" in factor.lower():
                recommendations.append({
                    'factor': 'Water Stress',
                    'action': 'Increase irrigation, use drip irrigation, apply mulch'
                })
            elif "pest" in factor.lower():
                recommendations.append({
                    'factor': 'Pest Management',
                    'action': 'Apply integrated pest management strategies'
                })
            elif "weed" in factor.lower():
                recommendations.append({
                    'factor': 'Weed Control',
                    'action': 'Manual/mechanical weeding or selective herbicide application'
                })
            elif "ph" in factor.lower():
                recommendations.append({
                    'factor': 'Soil pH',
                    'action': 'Apply lime (acidic) or organic matter (alkaline) to adjust pH'
                })
            elif "organic matter" in factor.lower():
                recommendations.append({
                    'factor': 'Soil Health',
                    'action': 'Add compost or organic fertilizers'
                })
        
        # Remove duplicates
        unique_recs = []
        seen_factors = set()
        for rec in recommendations:
            if rec['factor'] not in seen_factors:
                unique_recs.append(rec)
                seen_factors.add(rec['factor'])
        
        return unique_recs

# Global service instance
stress_service = StressPredictionService()
