"""
Simplified Stress Level Prediction Model Training
Uses only farmer-friendly inputs and auto-fetchable data

Features used:
- Manual Farmer Inputs: Crop_Type, Crop_Growth_Stage, Soil_Moisture, Soil_pH, 
                        Organic_Matter, Pest_Damage, Weed_Coverage
- Auto-Fetch: Temperature, Humidity, Rainfall, Wind_Speed, Elevation_Data, 
              Water_Flow, Drainage_Features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

def create_sample_dataset(n_samples=1000):
    """Create a realistic sample dataset for stress prediction"""
    np.random.seed(42)
    
    # Define realistic ranges
    crop_types = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 'Tomato', 'Potato']
    growth_stages = ['Germination', 'Vegetative', 'Flowering', 'Fruiting', 'Maturity']
    
    data = {
        # Manual Farmer Inputs
        'Crop_Type': np.random.choice(crop_types, n_samples),
        'Crop_Growth_Stage': np.random.choice(growth_stages, n_samples),
        'Soil_Moisture': np.random.uniform(20, 80, n_samples),  # %
        'Soil_pH': np.random.uniform(5.5, 8.0, n_samples),
        'Organic_Matter': np.random.uniform(1.5, 5.0, n_samples),  # %
        'Pest_Damage': np.random.uniform(0, 50, n_samples),  # %
        'Weed_Coverage': np.random.uniform(0, 40, n_samples),  # %
        
        # Auto-Fetch from APIs
        'Temperature': np.random.uniform(15, 42, n_samples),  # °C
        'Humidity': np.random.uniform(30, 95, n_samples),  # %
        'Rainfall': np.random.uniform(0, 150, n_samples),  # mm
        'Wind_Speed': np.random.uniform(0, 35, n_samples),  # km/h
        'Elevation_Data': np.random.uniform(0, 1500, n_samples),  # meters
        'Water_Flow': np.random.uniform(0, 100, n_samples),  # L/min
        'Drainage_Features': np.random.uniform(0, 100, n_samples),  # quality score
    }
    
    df = pd.DataFrame(data)
    
    # Generate stress levels based on conditions
    def calculate_stress(row):
        stress_points = 0
        
        # Temperature stress
        if row['Temperature'] > 38 or row['Temperature'] < 18:
            stress_points += 2
        elif row['Temperature'] > 35 or row['Temperature'] < 20:
            stress_points += 1
            
        # Soil moisture stress
        if row['Soil_Moisture'] < 30:
            stress_points += 2
        elif row['Soil_Moisture'] < 40:
            stress_points += 1
            
        # Drought conditions
        if row['Rainfall'] < 20:
            stress_points += 2
        elif row['Rainfall'] < 40:
            stress_points += 1
            
        # Humidity stress
        if row['Humidity'] < 35 or row['Humidity'] > 85:
            stress_points += 1
            
        # Soil pH stress
        if row['Soil_pH'] < 6.0 or row['Soil_pH'] > 7.5:
            stress_points += 1
            
        # Organic matter deficiency
        if row['Organic_Matter'] < 2.5:
            stress_points += 1
            
        # Pest damage
        if row['Pest_Damage'] > 30:
            stress_points += 2
        elif row['Pest_Damage'] > 15:
            stress_points += 1
            
        # Weed competition
        if row['Weed_Coverage'] > 25:
            stress_points += 1
            
        # Wind stress
        if row['Wind_Speed'] > 25:
            stress_points += 1
            
        # Poor drainage
        if row['Drainage_Features'] < 40:
            stress_points += 1
            
        # Classify stress level
        if stress_points >= 6:
            return 'High'
        elif stress_points >= 3:
            return 'Moderate'
        else:
            return 'Low'
    
    df['Stress_Level'] = df.apply(calculate_stress, axis=1)
    
    return df

def train_stress_model():
    """Train the simplified stress prediction model"""
    print("=" * 60)
    print("STRESS LEVEL PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create sample dataset
    print("\n📊 Creating sample dataset...")
    df = create_sample_dataset(n_samples=1000)
    
    print(f"✓ Dataset created with {len(df)} samples")
    print(f"\nStress Level Distribution:")
    print(df['Stress_Level'].value_counts())
    
    # Separate features and target
    X = df.drop('Stress_Level', axis=1)
    y = df['Stress_Level']
    
    # Encode categorical variables
    print("\n🔄 Encoding categorical variables...")
    label_encoders = {}
    categorical_cols = ['Crop_Type', 'Crop_Growth_Stage']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  ✓ Encoded {col}: {len(le.classes_)} classes")
    
    # Split dataset
    print("\n✂️ Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Training samples: {len(X_train)}")
    print(f"  ✓ Testing samples: {len(X_test)}")
    
    # Train model
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model trained successfully")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{'=' * 60}")
    print("MODEL PERFORMANCE")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE")
    print(f"{'=' * 60}")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Save model and encoders
    print(f"\n💾 Saving model and encoders...")
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'stress_prediction_model.pkl')
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save label encoders
    encoders_path = os.path.join(model_dir, 'stress_label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"  ✓ Encoders saved: {encoders_path}")
    
    # Save feature list
    feature_list = X.columns.tolist()
    features_path = os.path.join(model_dir, 'stress_features.pkl')
    joblib.dump(feature_list, features_path)
    print(f"  ✓ Features saved: {features_path}")
    
    print(f"\n{'=' * 60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("\nFiles saved:")
    print(f"  • {model_path}")
    print(f"  • {encoders_path}")
    print(f"  • {features_path}")
    
    return model, label_encoders, feature_list, accuracy, f1

if __name__ == "__main__":
    train_stress_model()
