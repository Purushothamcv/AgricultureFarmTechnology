"""
QUICK FRUIT DISEASE MODEL TRAINING
===================================
Fast training script to get a working model immediately (20-30 minutes)

Strategy: Single-phase training with frozen EfficientNet backbone
- Target: 85-90% accuracy in 20-25 epochs
- No fine-tuning (faster, more stable)
- Production-ready in <30 minutes
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

# ========================================
# CONFIGURATION
# ========================================
class Config:
    # Paths
    DATASET_PATH = r"C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend\data\archive"
    MODEL_SAVE_PATH = r"C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend\model\fruit_disease_model.h5"
    LABELS_PATH = r"C:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\backend\model\fruit_disease_labels.json"
    
    # Model parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 25  # Quick training
    LEARNING_RATE = 1e-3
    
    # Data split
    VALIDATION_SPLIT = 0.2


def create_data_generators():
    """Create training and validation generators with augmentation"""
    
    print("\n" + "="*70)
    print("CREATING DATA GENERATORS")
    print("="*70)
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Validation (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        Config.DATASET_PATH,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        Config.DATASET_PATH,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save class labels
    labels_map = {v: k for k, v in train_generator.class_indices.items()}
    with open(Config.LABELS_PATH, 'w') as f:
        json.dump(labels_map, f, indent=4)
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {validation_generator.samples}")
    print(f"✓ Number of classes: {len(train_generator.class_indices)}")
    print(f"✓ Labels saved to: {Config.LABELS_PATH}")
    
    # Compute class weights for imbalance
    class_counts = np.bincount(train_generator.classes)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    print("\n✓ Class weights computed for imbalanced dataset:")
    for cls_id, weight in sorted(class_weights_dict.items())[:5]:
        cls_name = labels_map[cls_id]
        print(f"  - {cls_name}: {weight:.2f} (samples: {class_counts[cls_id]})")
    
    return train_generator, validation_generator, class_weights_dict, labels_map


def build_model():
    """Build EfficientNet-B0 model with frozen backbone (fast training)"""
    
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    # Load EfficientNet-B0 with ImageNet weights
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*Config.IMG_SIZE, 3)
    )
    
    # FREEZE backbone for fast training
    base_model.trainable = False
    print("✓ EfficientNet-B0 backbone: FROZEN (feature extraction only)")
    
    # Build classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(17, activation='softmax')
    ], name='FruitDiseaseDetector')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print summary
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"✓ Architecture: EfficientNet-B0 + Classification Head")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Frozen parameters: {total_params - trainable_params:,}")
    print(f"✓ Learning rate: {Config.LEARNING_RATE}")
    
    return model


def create_callbacks():
    """Create training callbacks"""
    
    return [
        # Save best model
        ModelCheckpoint(
            filepath=Config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print(" "*15 + "QUICK FRUIT DISEASE MODEL TRAINING")
    print(" "*20 + "EfficientNet-B0 (Frozen Backbone)")
    print("="*70)
    
    start_time = datetime.now()
    
    # System info
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Verify dataset
    if not os.path.exists(Config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {Config.DATASET_PATH}")
    
    # Create data generators
    train_gen, val_gen, class_weights, labels = create_data_generators()
    
    # Build model
    model = build_model()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"✓ Target epochs: {Config.EPOCHS}")
    print(f"✓ Batch size: {Config.BATCH_SIZE}")
    print(f"✓ Class weights: ENABLED")
    print("="*70 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Load best model
    print("\n✓ Loading best model...")
    model = keras.models.load_model(Config.MODEL_SAVE_PATH)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"✓ Validation Loss: {val_loss:.4f}")
    print(f"✓ Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Training complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED!")
    print("="*70)
    print(f"✓ Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✓ Labels saved to: {Config.LABELS_PATH}")
    print(f"✓ Training duration: {duration}")
    print(f"✓ Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"✓ Total classes: 17")
    print("\n✓ Model is ready for production deployment!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
