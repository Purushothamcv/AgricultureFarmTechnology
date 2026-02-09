"""
Yield Prediction Model Training Script
Uses APY.csv dataset to train a robust yield prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class YieldModelTrainer:
    """Train and evaluate yield prediction model using APY dataset"""
    
    def __init__(self, csv_path="data/APY.csv"):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.encoders = {}
        self.feature_columns = ['State', 'District', 'Crop', 'Crop_Year', 'Season', 'Area']
        self.target_column = 'Yield'
        self.metrics = {}
        
    def load_and_prepare_data(self):
        """Load APY.csv and prepare for training"""
        print("=" * 60)
        print("🌾 YIELD PREDICTION MODEL TRAINING")
        print("=" * 60)
        
        # Load dataset
        print(f"\n📁 Loading dataset from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df):,} rows")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Clean string columns
        string_cols = ['State', 'District', 'Crop', 'Season']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip()
        
        print(f"\n🧹 Data Cleaning:")
        original_count = len(self.df)
        
        # Drop rows with null Yield (as per requirements)
        self.df = self.df.dropna(subset=['Yield'])
        print(f"  - Removed {original_count - len(self.df):,} rows with null Yield")
        
        # Drop rows with zero or negative yield
        self.df = self.df[self.df['Yield'] > 0]
        print(f"  - Kept only positive yields: {len(self.df):,} rows remaining")
        
        # IMPORTANT: Do NOT use Production as feature (data leakage prevention)
        # Since Yield = Production / Area, using Production would leak the target
        print(f"\n⚠️  Data Leakage Prevention:")
        print(f"  - Excluding 'Production' from features")
        print(f"  - Using only: {', '.join(self.feature_columns)}")
        
        # Check for missing values in features
        missing = self.df[self.feature_columns].isnull().sum()
        if missing.any():
            print(f"\n🔍 Missing values in features:")
            print(missing[missing > 0])
            self.df = self.df.dropna(subset=self.feature_columns)
            print(f"  - Dropped rows with missing features: {len(self.df):,} rows remaining")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  - Total records: {len(self.df):,}")
        print(f"  - Unique States: {self.df['State'].nunique()}")
        print(f"  - Unique Districts: {self.df['District'].nunique()}")
        print(f"  - Unique Crops: {self.df['Crop'].nunique()}")
        print(f"  - Unique Seasons: {self.df['Season'].nunique()}")
        print(f"  - Years range: {self.df['Crop_Year'].min()} - {self.df['Crop_Year'].max()}")
        print(f"\n📈 Yield Statistics:")
        print(f"  - Mean: {self.df['Yield'].mean():.2f}")
        print(f"  - Median: {self.df['Yield'].median():.2f}")
        print(f"  - Min: {self.df['Yield'].min():.2f}")
        print(f"  - Max: {self.df['Yield'].max():.2f}")
        print(f"  - Std: {self.df['Yield'].std():.2f}")
        
    def encode_features(self):
        """Encode categorical features using Label Encoding"""
        print(f"\n🔢 Encoding Categorical Features:")
        
        categorical_features = ['State', 'District', 'Crop', 'Season']
        
        for feature in categorical_features:
            print(f"  - Encoding {feature} ({self.df[feature].nunique()} unique values)...")
            encoder = LabelEncoder()
            self.df[f'{feature}_encoded'] = encoder.fit_transform(self.df[feature])
            self.encoders[feature] = encoder
        
        print(f"✅ All categorical features encoded")
        
    def prepare_features_target(self):
        """Prepare feature matrix and target vector"""
        print(f"\n🎯 Preparing Features and Target:")
        
        # Feature columns (encoded + numerical)
        feature_cols = [
            'State_encoded',
            'District_encoded', 
            'Crop_encoded',
            'Crop_Year',
            'Season_encoded',
            'Area'
        ]
        
        X = self.df[feature_cols]
        y = self.df[self.target_column]
        
        print(f"  - Features shape: {X.shape}")
        print(f"  - Target shape: {y.shape}")
        print(f"  - Feature columns: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y, use_timeseries_split=True):
        """Train XGBoost model with proper validation"""
        print(f"\n🤖 Training Model:")
        
        if use_timeseries_split:
            print(f"  - Using TimeSeriesSplit for temporal validation")
            # Sort by year to ensure temporal ordering
            sorted_indices = self.df.sort_values('Crop_Year').index
            X_sorted = X.loc[sorted_indices]
            y_sorted = y.loc[sorted_indices]
            
            # Split: 80% train, 20% test (most recent data as test)
            split_idx = int(len(X_sorted) * 0.8)
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            
            train_years = self.df.loc[X_train.index, 'Crop_Year']
            test_years = self.df.loc[X_test.index, 'Crop_Year']
            print(f"  - Train years: {train_years.min()} - {train_years.max()}")
            print(f"  - Test years: {test_years.min()} - {test_years.max()}")
        else:
            print(f"  - Using random train-test split (80-20)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"  - Training set: {len(X_train):,} samples")
        print(f"  - Test set: {len(X_test):,} samples")
        
        # Try both XGBoost and RandomForest, select best
        print(f"\n🔬 Training XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        print(f"🔬 Training RandomForest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate both models
        print(f"\n📊 Model Comparison:")
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        print(f"\n  XGBoost:")
        print(f"    - R² Score: {xgb_r2:.4f}")
        print(f"    - RMSE: {xgb_rmse:.4f}")
        print(f"    - MAE: {xgb_mae:.4f}")
        
        print(f"\n  RandomForest:")
        print(f"    - R² Score: {rf_r2:.4f}")
        print(f"    - RMSE: {rf_rmse:.4f}")
        print(f"    - MAE: {rf_mae:.4f}")
        
        # Select best model based on R² score
        if xgb_r2 >= rf_r2:
            print(f"\n✅ Selected: XGBoost (better R² score)")
            self.model = xgb_model
            self.metrics = {
                'model_type': 'XGBoost',
                'r2_score': float(xgb_r2),
                'rmse': float(xgb_rmse),
                'mae': float(xgb_mae),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        else:
            print(f"\n✅ Selected: RandomForest (better R² score)")
            self.model = rf_model
            self.metrics = {
                'model_type': 'RandomForest',
                'r2_score': float(rf_r2),
                'rmse': float(rf_rmse),
                'mae': float(rf_mae),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        
        return X_train, X_test, y_train, y_test
    
    def save_model_and_artifacts(self):
        """Save trained model and encoders"""
        print(f"\n💾 Saving Model and Artifacts:")
        
        # Create model directory if not exists
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "yield_prediction_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"  ✅ Model saved to: {model_path}")
        
        # Save encoders
        encoders_path = model_dir / "yield_encoders.pkl"
        joblib.dump(self.encoders, encoders_path)
        print(f"  ✅ Encoders saved to: {encoders_path}")
        
        # Save metrics
        metrics_path = model_dir / "yield_model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  ✅ Metrics saved to: {metrics_path}")
        
        # Save feature info
        feature_info = {
            'feature_columns': self.feature_columns,
            'encoded_features': list(self.encoders.keys()),
            'target_column': self.target_column,
            'categorical_mappings': {
                feature: {
                    'classes': encoder.classes_.tolist(),
                    'n_classes': len(encoder.classes_)
                }
                for feature, encoder in self.encoders.items()
            }
        }
        feature_info_path = model_dir / "yield_feature_info.json"
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  ✅ Feature info saved to: {feature_info_path}")
        
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        try:
            self.load_and_prepare_data()
            self.encode_features()
            X, y = self.prepare_features_target()
            X_train, X_test, y_train, y_test = self.train_model(X, y)
            self.save_model_and_artifacts()
            
            print(f"\n" + "=" * 60)
            print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"=" * 60)
            print(f"\n📈 Final Model Performance:")
            print(f"  - Model Type: {self.metrics['model_type']}")
            print(f"  - R² Score: {self.metrics['r2_score']:.4f}")
            print(f"  - RMSE: {self.metrics['rmse']:.4f}")
            print(f"  - MAE: {self.metrics['mae']:.4f}")
            print(f"\n✅ Model is ready for production use!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main training function"""
    trainer = YieldModelTrainer(csv_path="data/APY.csv")
    success = trainer.run_training_pipeline()
    
    if success:
        print(f"\n🚀 You can now use the model in your API!")
        print(f"   Model file: model/yield_prediction_model.pkl")
        print(f"   Encoders file: model/yield_encoders.pkl")
    else:
        print(f"\n⚠️  Training failed. Please check the errors above.")


if __name__ == "__main__":
    main()
