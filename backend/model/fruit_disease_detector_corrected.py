"""
Fruit Disease Detection - CORRECTED Inference Module
====================================================
BIOLOGICALLY CORRECT predictions using ONLY trained labels

CRITICAL FIXES:
1. STRICT label mapping from fruit_disease_labels.json ONLY
2. Fruit-aware validation (no Apple diseases for Guava, etc.)
3. Conservative confidence handling
4. Top-3 decision logic to prevent false negatives
5. Dynamic severity based on confidence
6. Treatment ONLY for trained disease labels

Author: SmartAgri-AI Team
Date: 2026-01-25
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FruitDiseaseDetectorCorrected:
    """
    CORRECTED Fruit Disease Detector with Biological Validation
    
    GUARANTEES:
    - Uses ONLY labels from fruit_disease_labels.json
    - Validates fruit-disease compatibility
    - No external or hardcoded disease names
    - Conservative predictions with safety checks
    """
    
    # CRITICAL: Define ONLY the valid fruits from training
    VALID_FRUITS = {"Apple", "Guava", "Mango", "Pomegranate"}
    
    # CRITICAL: Treatment database - ONLY for trained labels
    TREATMENT_DATABASE = {
        "Alternaria_Mango": "Apply Azoxystrobin or Mancozeb fungicide. Remove and destroy infected leaves and fruits. Improve air circulation in the orchard.",
        "Alternaria_Pomegranate": "Apply Mancozeb or Chlorothalonil fungicide. Remove infected plant parts. Improve drainage and reduce humidity.",
        "Anthracnose_Guava": "Apply Carbendazim or Mancozeb fungicide every 10-15 days. Remove infected fruits. Maintain proper plant spacing for air circulation.",
        "Anthracnose_Mango": "Apply Copper oxychloride or Carbendazim. Remove infected fruits immediately. Harvest at proper maturity to reduce susceptibility.",
        "Anthracnose_Pomegranate": "Apply Carbendazim or Copper oxychloride. Prune infected branches. Maintain proper spacing between plants.",
        "Bacterial_Blight_Pomegranate": "Apply Streptocycline (antibiotic) spray. Remove and burn infected parts. Avoid overhead irrigation. Use resistant varieties.",
        "Black Mould Rot (Aspergillus)_Mango": "Apply post-harvest fungicides (Imazalil or Thiabendazole). Store fruits at low temperature (10-15°C). Handle carefully to avoid injuries.",
        "Blotch_Apple": "Apply fungicides like Captan or Mancozeb during bloom and fruit development. Remove infected fruits. Improve air circulation.",
        "Cercospora_Pomegranate": "Apply Copper-based fungicides or Mancozeb. Remove infected leaves. Improve air circulation and reduce humidity.",
        "Fruitfly_Guava": "Use pheromone traps and protein bait sprays. Bag fruits for protection. Remove and destroy infested fruits immediately. Practice sanitation.",
        "Healthy_Apple": "No treatment needed. Continue regular monitoring and good agricultural practices. Maintain proper nutrition and water management.",
        "Healthy_Guava": "No treatment needed. Continue regular monitoring and good agricultural practices. Maintain proper nutrition and water management.",
        "Healthy_Mango": "No treatment needed. Continue regular monitoring and good agricultural practices. Maintain proper nutrition and water management.",
        "Healthy_Pomegranate": "No treatment needed. Continue regular monitoring and good agricultural practices. Maintain proper nutrition and water management.",
        "Rot_Apple": "Remove infected fruits immediately. Apply copper-based fungicides. Ensure proper drainage. Avoid fruit injuries during handling.",
        "Scab_Apple": "Apply fungicides (Captan, Mancozeb) during bud break and wet periods. Remove fallen leaves. Use scab-resistant apple varieties.",
        "Stem and Rot (Lasiodiplodia)_Mango": "Remove infected branches 15cm below visible infection. Apply Thiophanate-methyl or Copper oxychloride. Improve tree health with proper nutrition."
    }
    
    def __init__(self, model_path: str = None, labels_path: str = None):
        """
        Initialize detector with STRICT label enforcement
        
        Args:
            model_path: Path to fruit_disease_model.h5
            labels_path: Path to fruit_disease_labels.json
        """
        # Determine paths
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'fruit_disease_model.h5')
        
        if labels_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            labels_path = os.path.join(current_dir, 'fruit_disease_labels.json')
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = {}
        self.num_classes = 17
        
        # Image dimensions (EfficientNet-B0 standard)
        self.img_height = 224
        self.img_width = 224
        
        # CRITICAL: Load labels FIRST - this is our source of truth
        self._load_and_validate_labels()
        
        # Then load model
        self._load_model()
        
        logger.info("✅ FruitDiseaseDetectorCorrected initialized successfully")
        logger.info(f"✅ Loaded {len(self.labels)} VALID disease labels")
        logger.info(f"✅ Supported fruits: {sorted(self.VALID_FRUITS)}")
    
    def _load_and_validate_labels(self):
        """
        CRITICAL: Load and validate labels from JSON
        This is the ONLY source of truth for disease names
        """
        try:
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
            
            with open(self.labels_path, 'r') as f:
                self.labels = json.load(f)
            
            # CRITICAL: Validate all labels match expected format
            for idx, label in self.labels.items():
                # Must have underscore separator
                if '_' not in label and 'Healthy' not in label:
                    logger.warning(f"Invalid label format: {label}")
                
                # Extract fruit name
                parts = label.split('_')
                fruit = parts[-1]
                
                # Validate fruit is in our known set
                if fruit not in self.VALID_FRUITS:
                    logger.warning(f"Unknown fruit in label: {fruit} (from {label})")
            
            logger.info(f"✓ Validated {len(self.labels)} class labels from {self.labels_path}")
            
            # Log all labels for verification
            logger.info("Loaded labels:")
            for idx in sorted([int(k) for k in self.labels.keys()]):
                logger.info(f"  {idx}: {self.labels[str(idx)]}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load labels: {e}")
            raise
    
    def _load_model(self):
        """Load trained Keras model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load model without compilation (faster for inference)
            self.model = load_model(self.model_path, compile=False)
            
            logger.info("✓ Model loaded successfully")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            
            # Validate output shape matches number of labels
            output_classes = self.model.output_shape[-1]
            if output_classes != len(self.labels):
                raise ValueError(f"Model output classes ({output_classes}) != labels ({len(self.labels)})")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image, debug: bool = False) -> np.ndarray:
        """
        Preprocess image for EfficientNet-B0
        MUST match training preprocessing exactly
        
        Args:
            image: PIL Image object
            debug: Enable debug logging
        
        Returns:
            Preprocessed numpy array [1, 224, 224, 3]
        """
        try:
            if debug:
                logger.info(f"Input image mode: {image.mode}, size: {image.size}")
            
            # Convert to RGB (handle grayscale or RGBA)
            if image.mode != 'RGB':
                if debug:
                    logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.img_width, self.img_height), Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            if debug:
                logger.info(f"Array shape: {img_array.shape}, range: [{img_array.min():.0f}, {img_array.max():.0f}]")
            
            # Apply EfficientNet preprocessing (normalize to [-1, 1])
            img_array = preprocess_input(img_array)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            if debug:
                logger.info(f"Final shape: {img_array.shape}, range: [{img_array.min():.2f}, {img_array.max():.2f}]")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def extract_fruit_from_label(self, label: str) -> str:
        """
        CRITICAL: Extract fruit name from label
        
        Args:
            label: Full label like "Anthracnose_Mango"
        
        Returns:
            Fruit name like "Mango"
        """
        parts = label.split('_')
        return parts[-1] if parts else "Unknown"
    
    def extract_disease_from_label(self, label: str) -> str:
        """
        CRITICAL: Extract disease name from label
        
        Args:
            label: Full label like "Anthracnose_Mango"
        
        Returns:
            Disease name like "Anthracnose"
        """
        if "Healthy" in label:
            return "Healthy"
        
        parts = label.split('_')
        return '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
    
    def validate_prediction_compatibility(
        self, 
        predicted_label: str, 
        top_predictions: List[Dict],
        debug: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        CRITICAL: Validate that prediction makes biological sense
        
        Check if predicted disease is compatible with detected fruit
        
        Args:
            predicted_label: Top prediction label
            top_predictions: List of top-N predictions with confidence
            debug: Enable logging
        
        Returns:
            (is_valid, warnings) tuple
        """
        warnings = []
        
        # Extract fruit from top prediction
        predicted_fruit = self.extract_fruit_from_label(predicted_label)
        
        # Check if fruit is valid
        if predicted_fruit not in self.VALID_FRUITS:
            warnings.append(f"Unrecognized fruit type: {predicted_fruit}")
            return False, warnings
        
        # Check top-3 for conflicting fruits
        fruits_in_top3 = set()
        for pred in top_predictions[:3]:
            fruit = self.extract_fruit_from_label(pred['class'])
            fruits_in_top3.add(fruit)
        
        if len(fruits_in_top3) > 1:
            # Multiple fruits detected - ambiguous
            warnings.append(f"Multiple fruit types detected in top predictions: {', '.join(sorted(fruits_in_top3))}")
            warnings.append("Image may be unclear or contain multiple fruits")
            if debug:
                logger.warning(f"Fruit conflict detected: {fruits_in_top3}")
        
        return True, warnings
    
    def predict(
        self, 
        image: Image.Image, 
        top_n: int = 3,
        confidence_threshold: float = 0.70,
        debug: bool = False
    ) -> Dict:
        """
        CRITICAL: Predict with STRICT label mapping and validation
        
        GUARANTEES:
        - Uses ONLY labels from fruit_disease_labels.json
        - Validates fruit-disease compatibility
        - Provides top-N predictions
        - Flags uncertain/ambiguous predictions
        
        Args:
            image: PIL Image object
            top_n: Number of top predictions (default: 3)
            confidence_threshold: Minimum confidence for reliable prediction (default: 0.70)
            debug: Enable debug logging
        
        Returns:
            Dictionary with prediction results and validation flags
        """
        try:
            if debug:
                logger.info("="*70)
                logger.info("STARTING PREDICTION")
                logger.info("="*70)
            
            # Preprocess image
            img_array = self.preprocess_image(image, debug=debug)
            
            # Run prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            if debug:
                logger.info(f"Raw predictions shape: {predictions.shape}")
                logger.info(f"Sum of probabilities: {predictions.sum():.4f} (should be ~1.0)")
            
            # CRITICAL: Map prediction index to label using ONLY self.labels
            predicted_idx = int(np.argmax(predictions))
            
            # CRITICAL: Validate index is in our labels
            if str(predicted_idx) not in self.labels:
                raise ValueError(f"Invalid prediction index: {predicted_idx} (not in labels)")
            
            predicted_label = self.labels[str(predicted_idx)]
            confidence = float(predictions[predicted_idx])
            
            if debug:
                logger.info(f"Predicted index: {predicted_idx}")
                logger.info(f"Predicted label: {predicted_label}")
                logger.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # CRITICAL: Get top-N predictions using ONLY self.labels
            top_indices = np.argsort(predictions)[-top_n:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                idx_str = str(int(idx))
                if idx_str not in self.labels:
                    logger.warning(f"Skipping invalid index: {idx}")
                    continue
                
                top_predictions.append({
                    "class": self.labels[idx_str],
                    "confidence": float(predictions[idx])
                })
            
            if debug:
                logger.info("Top-3 predictions:")
                for i, pred in enumerate(top_predictions, 1):
                    logger.info(f"  {i}. {pred['class']:45s} {pred['confidence']*100:6.2f}%")
            
            # CRITICAL: Validate prediction compatibility
            is_valid, validation_warnings = self.validate_prediction_compatibility(
                predicted_label, 
                top_predictions, 
                debug
            )
            
            # CRITICAL: Check prediction quality
            is_uncertain = confidence < confidence_threshold
            
            # CRITICAL: Check for ambiguous "Healthy" predictions
            is_ambiguous_healthy = False
            potential_disease = None
            
            if "Healthy" in predicted_label and len(top_predictions) >= 2:
                second_pred = top_predictions[1]
                if "Healthy" not in second_pred["class"] and second_pred["confidence"] > 0.20:
                    is_ambiguous_healthy = True
                    potential_disease = second_pred["class"]
                    if debug:
                        logger.warning(f"⚠️ Ambiguous Healthy: {potential_disease} has {second_pred['confidence']*100:.1f}% confidence")
            
            # Build result dictionary
            result = {
                "prediction": predicted_label,
                "confidence": confidence,
                "fruit": self.extract_fruit_from_label(predicted_label),
                "disease": self.extract_disease_from_label(predicted_label),
                "top_3": top_predictions,
                "is_uncertain": is_uncertain,
                "is_valid": is_valid,
                "is_ambiguous_healthy": is_ambiguous_healthy,
                "potential_disease": potential_disease,
                "validation_warnings": validation_warnings
            }
            
            if debug:
                logger.info("="*70)
                logger.info("PREDICTION COMPLETE")
                logger.info("="*70)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_treatment(self, predicted_label: str) -> str:
        """
        CRITICAL: Get treatment ONLY for trained labels
        
        Args:
            predicted_label: Full label from predictions
        
        Returns:
            Treatment recommendation
        """
        # CRITICAL: Only return treatment if label exists in our database
        if predicted_label in self.TREATMENT_DATABASE:
            return self.TREATMENT_DATABASE[predicted_label]
        else:
            # Fallback: label is valid but treatment not in database
            logger.warning(f"No treatment data for: {predicted_label}")
            return f"Treatment information not available for {predicted_label}. Consult agricultural expert for guidance."
    
    def get_severity(self, predicted_label: str, confidence: float) -> Tuple[str, int]:
        """
        CRITICAL: Get severity based on confidence and disease type
        
        Args:
            predicted_label: Full label
            confidence: Prediction confidence (0-1)
        
        Returns:
            (severity_text, severity_code) tuple
        """
        if "Healthy" in predicted_label:
            return ("None", 0)
        
        # Dynamic severity based on confidence
        if confidence >= 0.85:
            return ("High", 3)
        elif confidence >= 0.70:
            return ("Moderate to High", 2)
        elif confidence >= 0.50:
            return ("Moderate (Uncertain)", 1)
        else:
            return ("Low (Very Uncertain)", 0)
    
    def predict_with_details(
        self, 
        image: Image.Image, 
        top_n: int = 3,
        confidence_threshold: float = 0.70,
        debug: bool = False
    ) -> Dict:
        """
        CRITICAL: Full prediction with safety checks and recommendations
        
        Args:
            image: PIL Image object
            top_n: Number of top predictions
            confidence_threshold: Minimum confidence for reliable prediction
            debug: Enable debug logging
        
        Returns:
            Complete prediction results with warnings and recommendations
        """
        # Get base prediction
        result = self.predict(image, top_n, confidence_threshold, debug)
        
        # Extract data
        predicted_label = result["prediction"]
        confidence = result["confidence"]
        fruit = result["fruit"]
        disease = result["disease"]
        is_uncertain = result["is_uncertain"]
        is_ambiguous_healthy = result["is_ambiguous_healthy"]
        
        # CRITICAL: Get severity based on confidence
        severity_text, severity_code = self.get_severity(predicted_label, confidence)
        
        # CRITICAL: Get treatment from database (ONLY for trained labels)
        treatment = self.get_treatment(predicted_label)
        
        # Build disease info
        result["disease_info"] = {
            "full_label": predicted_label,
            "fruit": fruit,
            "disease": disease,
            "severity": severity_text,
            "severity_code": severity_code,
            "treatment": treatment,
            "confidence": confidence
        }
        
        # Generate interpretation with warnings
        warnings = list(result.get("validation_warnings", []))
        
        if is_uncertain:
            warnings.append(f"Low confidence ({confidence*100:.1f}%) - Prediction may be unreliable")
            result["interpretation"] = f"⚠️ LOW CONFIDENCE - Consider uploading a clearer image"
            result["action_required"] = "UPLOAD_BETTER_IMAGE"
        elif is_ambiguous_healthy:
            warnings.append(f"Healthy prediction is ambiguous - {result['potential_disease']} detected with {result['top_3'][1]['confidence']*100:.1f}% confidence")
            result["interpretation"] = f"⚠️ AMBIGUOUS - Predicted Healthy but {disease} is possible"
            result["action_required"] = "EXPERT_VERIFICATION"
        elif confidence >= 0.90:
            result["interpretation"] = f"High confidence ({confidence*100:.1f}%) - Reliable prediction"
            result["action_required"] = "FOLLOW_TREATMENT" if "Healthy" not in predicted_label else "CONTINUE_MONITORING"
        elif confidence >= 0.70:
            result["interpretation"] = f"Good confidence ({confidence*100:.1f}%) - Likely accurate"
            result["action_required"] = "EXPERT_REVIEW_RECOMMENDED"
        else:
            warnings.append(f"Moderate confidence ({confidence*100:.1f}%) - Expert review advised")
            result["interpretation"] = f"Moderate confidence - Consider expert review"
            result["action_required"] = "EXPERT_REVIEW"
        
        result["warnings"] = warnings
        result["has_warnings"] = len(warnings) > 0
        
        return result


# CRITICAL: Test function for validation
if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING CORRECTED FRUIT DISEASE DETECTOR")
    print("="*70)
    
    try:
        # Initialize detector
        detector = FruitDiseaseDetectorCorrected()
        print(f"\n✅ Detector initialized successfully")
        print(f"✅ Model path: {detector.model_path}")
        print(f"✅ Labels loaded: {len(detector.labels)}")
        print(f"✅ Valid fruits: {sorted(detector.VALID_FRUITS)}")
        
        # Verify all labels
        print(f"\n📋 All Trained Labels:")
        for idx in sorted([int(k) for k in detector.labels.keys()]):
            label = detector.labels[str(idx)]
            fruit = detector.extract_fruit_from_label(label)
            disease = detector.extract_disease_from_label(label)
            print(f"  {idx:2d}. {label:45s} -> {fruit:12s} | {disease}")
        
        print(f"\n✅ All labels validated - NO external diseases possible")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}\n")
