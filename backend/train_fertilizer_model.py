"""
Fertilizer Recommendation Model Training Script
Uses fertilizer_recommendation.csv dataset to train a classification model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FertilizerModelTrainer:
    """Train and evaluate fertilizer recommendation model"""
    
    def __init__(self, csv_path="data/fertilizer_recommendation.csv"):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.encoders = {}
        self.label_encoder = None
        self.feature_columns = []
        self.target_column = 'Recommended_Fertilizer'
        self.metrics = {}
        
    def load_and_prepare_data(self):
        """Load fertilizer dataset and prepare for training"""
        print("=" * 60)
        print("🌱 FERTILIZER RECOMMENDATION MODEL TRAINING")
        print("=" * 60)
        
        # Load dataset
        print(f"\n📁 Loading dataset from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df):,} rows")
        
        print(f"\n📊 Dataset Info:")
        print(f"  - Columns: {len(self.df.columns)}")
        print(f"  - Features: {list(self.df.columns[:10])}...")
        
        # Check for missing target
        original_count = len(self.df)
        self.df = self.df.dropna(subset=[self.target_column])
        dropped = original_count - len(self.df)
        if dropped > 0:
            print(f"🧹 Dropped {dropped} rows with missing fertilizer labels")
        
        # Get unique fertilizers
        fertilizers = self.df[self.target_column].unique()
        print(f"\n🥬 Unique Fertilizers in Dataset: {len(fertilizers)}")
        print(f"   {sorted(fertilizers)}")
        
        # Define features (exclude target and identifiers)
        exclude_cols = [
            self.target_column,
            'Fertilizer_Used_Last_Season',  # Past info, not predictor
            'Yield_Last_Season'  # Past info, not predictor
        ]
        
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        print(f"\n🎯 Using {len(self.feature_columns)} features:")
        print(f"   {self.feature_columns}")
        
    def encode_features(self):
        """Encode categorical features"""
        print(f"\n🔢 Encoding Features:")
        
        # Identify categorical columns
        categorical_cols = self.df[self.feature_columns].select_dtypes(include=['object']).columns.tolist()
        numerical_cols = [col for col in self.feature_columns if col not in categorical_cols]
        
        print(f"  - Categorical features ({len(categorical_cols)}): {categorical_cols}")
        print(f"  - Numerical features ({len(numerical_cols)}): {numerical_cols}")
        
        # Encode categorical features
        for col in categorical_cols:
            print(f"    Encoding {col} ({self.df[col].nunique()} unique values)...")
            encoder = LabelEncoder()
            self.df[f'{col}_encoded'] = encoder.fit_transform(self.df[col].astype(str))
            self.encoders[col] = encoder
        
        # Update feature columns to use encoded versions
        self.feature_columns = [
            f'{col}_encoded' if col in categorical_cols else col
            for col in self.feature_columns
        ]
        
        # Encode target
        print(f"  - Encoding target: {self.target_column}")
        self.label_encoder = LabelEncoder()
        self.df['target_encoded'] = self.label_encoder.fit_transform(self.df[self.target_column])
        
        print(f"✅ All features encoded")
        
    def prepare_features_target(self):
        """Prepare feature matrix and target vector"""
        print(f"\n🎯 Preparing Features and Target:")
        
        X = self.df[self.feature_columns]
        y = self.df['target_encoded']
        
        print(f"  - Features shape: {X.shape}")
        print(f"  - Target shape: {y.shape}")
        print(f"  - Feature columns: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train Random Forest classification model"""
        print(f"\n🤖 Training Model:")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  - Training set: {len(X_train):,} samples")
        print(f"  - Test set: {len(X_test):,} samples")
        
        # Train Random Forest
        print(f"\n🔬 Training RandomForestClassifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print(f"\n📊 Model Evaluation:")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  - F1-Score (weighted): {f1:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Show per-class performance
        for fert in sorted(self.label_encoder.classes_):
            if fert in report:
                print(f"  {fert:20s} - Precision: {report[fert]['precision']:.2f}, "
                      f"Recall: {report[fert]['recall']:.2f}, "
                      f"F1: {report[fert]['f1-score']:.2f}")
        
        self.metrics = {
            'model_type': 'RandomForestClassifier',
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_)
        }
        
        return X_train, X_test, y_train, y_test
    
    def save_model_and_artifacts(self):
        """Save trained model, encoders, and metadata"""
        print(f"\n💾 Saving Model and Artifacts:")
        
        # Create model directory
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "fertilizer_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"  ✅ Model saved to: {model_path}")
        
        # Save feature encoders
        encoders_path = model_dir / "fertilizer_encoders.pkl"
        joblib.dump(self.encoders, encoders_path)
        print(f"  ✅ Feature encoders saved: {encoders_path}")
        
        # Save label encoder
        label_encoder_path = model_dir / "fertilizer_label_encoder.pkl"
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"  ✅ Label encoder saved: {label_encoder_path}")
        
        # Save metrics
        metrics_path = model_dir / "fertilizer_model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  ✅ Metrics saved: {metrics_path}")
        
        # Save feature info
        feature_info = {
            'feature_columns': self.feature_columns,
            'original_features': list(self.encoders.keys()),
            'numerical_features': [col for col in self.feature_columns if not col.endswith('_encoded')],
            'target_column': self.target_column,
            'fertilizer_classes': self.label_encoder.classes_.tolist()
        }
        feature_info_path = model_dir / "fertilizer_feature_info.json"
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  ✅ Feature info saved: {feature_info_path}")
        
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
            print(f"  - Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy'] * 100:.2f}%)")
            print(f"  - F1-Score: {self.metrics['f1_score']:.4f}")
            print(f"  - Classes: {self.metrics['n_classes']} fertilizer types")
            print(f"\n✅ Model is ready for production use!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main training function"""
    trainer = FertilizerModelTrainer(csv_path="data/fertilizer_recommendation.csv")
    success = trainer.run_training_pipeline()
    
    if success:
        print(f"\n🚀 You can now use the model in your API!")
        print(f"   Model file: model/fertilizer_model.pkl")
        print(f"   Encoders: model/fertilizer_encoders.pkl")
    else:
        print(f"\n⚠️  Training failed. Please check the errors above.")


if __name__ == "__main__":
    main()
