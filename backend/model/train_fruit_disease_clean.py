"""
Fruit Disease Detection - Clean Training Script (RESTART FROM SCRATCH)
=======================================================================
Framework: TensorFlow/Keras
Architecture: EfficientNet-B0 (Transfer Learning)
Dataset: Multi-class fruit disease classification (17 classes)
Training Strategy: Two-phase (Feature Extraction + Fine-Tuning)

CRITICAL: This script completely restarts training - NO checkpoint resumption
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# CONFIGURATION
# ========================================
class Config:
    # Paths
    DATASET_PATH = os.path.join('..', 'data', 'archive')
    MODEL_OUTPUT = 'fruit_disease_model.h5'
    LABELS_OUTPUT = 'fruit_disease_labels.json'
    HISTORY_OUTPUT = 'training_history.json'
    
    # Model Architecture
    INPUT_SHAPE = (224, 224, 3)  # EfficientNet-B0 standard input
    NUM_CLASSES = 17
    
    # Training - Phase 1: Feature Extraction
    PHASE1_EPOCHS = 20
    PHASE1_LR = 1e-3
    PHASE1_BATCH_SIZE = 32
    
    # Training - Phase 2: Fine-Tuning
    PHASE2_EPOCHS = 30
    PHASE2_LR = 1e-5
    PHASE2_BATCH_SIZE = 32
    UNFREEZE_LAYERS = 30  # Unfreeze top 30 layers
    
    # Data Split
    VALIDATION_SPLIT = 0.2
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    # Regularization
    DROPOUT_RATE = 0.45
    DENSE_UNITS = 256

# Initialize configuration
config = Config()

# ========================================
# SETUP & ENVIRONMENT
# ========================================
def setup_environment():
    """Configure TensorFlow and GPU settings"""
    print("=" * 70)
    print("FRUIT DISEASE CNN TRAINING - CLEAN RESTART")
    print("=" * 70)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"✗ GPU Error: {e}")
    else:
        print("⚠ Running on CPU (training will be slower)")
    
    print("=" * 70)
    print()

# ========================================
# DATA PIPELINE
# ========================================
def create_data_generators():
    """
    Create training and validation data generators with EfficientNet preprocessing
    and strong data augmentation
    """
    print("📊 CREATING DATA PIPELINE")
    print("-" * 70)
    
    # Training Data Augmentation (STRONG)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Validation Data (NO AUGMENTATION, only preprocessing)
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Training Generator
    train_generator = train_datagen.flow_from_directory(
        config.DATASET_PATH,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.PHASE1_BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation Generator
    val_generator = val_datagen.flow_from_directory(
        config.DATASET_PATH,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.PHASE1_BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Number of classes: {len(train_generator.class_indices)}")
    print(f"✓ Batch size: {config.PHASE1_BATCH_SIZE}")
    print()
    
    # Save class labels
    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    
    with open(config.LABELS_OUTPUT, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"✓ Class labels saved to: {config.LABELS_OUTPUT}")
    print()
    
    # Display class distribution
    print("📈 CLASS DISTRIBUTION:")
    print("-" * 70)
    for class_name, class_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        print(f"  [{class_idx:2d}] {class_name}")
    print()
    
    return train_generator, val_generator, class_indices

# ========================================
# CLASS IMBALANCE HANDLING
# ========================================
def compute_class_weights(train_generator):
    """
    Compute class weights to handle class imbalance
    """
    print("⚖️  COMPUTING CLASS WEIGHTS (Class Imbalance Handling)")
    print("-" * 70)
    
    # Get class labels for all training samples
    y_train = train_generator.classes
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Display weights
    print("Class weights computed:")
    for class_idx, weight in class_weight_dict.items():
        print(f"  Class {class_idx:2d}: {weight:.4f}")
    print()
    
    return class_weight_dict

# ========================================
# MODEL ARCHITECTURE
# ========================================
def build_model(num_classes):
    """
    Build EfficientNet-B0 model with custom classification head
    
    Phase 1: All EfficientNet layers frozen
    Phase 2: Top layers unfrozen for fine-tuning
    """
    print("🏗️  BUILDING MODEL ARCHITECTURE")
    print("-" * 70)
    
    # Load EfficientNet-B0 pretrained on ImageNet
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=config.INPUT_SHAPE,
        pooling=None  # We'll add our own pooling
    )
    
    # Freeze all layers for Phase 1
    base_model.trainable = False
    
    print(f"✓ Base model: EfficientNet-B0")
    print(f"✓ Pretrained weights: ImageNet")
    print(f"✓ Input shape: {config.INPUT_SHAPE}")
    print(f"✓ Base model layers: {len(base_model.layers)}")
    print(f"✓ Initial state: ALL LAYERS FROZEN")
    print()
    
    # Build custom classification head
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    x = base_model(inputs, training=False)  # training=False for batch norm
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_features')(x)
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs, outputs, name='FruitDiseaseModel')
    
    print("✓ Custom classification head added:")
    print(f"  - GlobalAveragePooling2D")
    print(f"  - Dense({config.DENSE_UNITS}, activation='relu')")
    print(f"  - Dropout({config.DROPOUT_RATE})")
    print(f"  - Dense({num_classes}, activation='softmax')")
    print()
    
    return model, base_model

# ========================================
# TRAINING - PHASE 1: FEATURE EXTRACTION
# ========================================
def train_phase1(model, train_gen, val_gen, class_weights):
    """
    Phase 1: Feature Extraction
    - All EfficientNet layers frozen
    - Train only classification head
    - Higher learning rate (1e-3)
    """
    print("=" * 70)
    print("🚀 PHASE 1: FEATURE EXTRACTION")
    print("=" * 70)
    print("Strategy: Train classification head with frozen base model")
    print(f"Epochs: {config.PHASE1_EPOCHS}")
    print(f"Learning Rate: {config.PHASE1_LR}")
    print(f"Batch Size: {config.PHASE1_BATCH_SIZE}")
    print()
    
    # Compile model for Phase 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    )
    
    print("✓ Model compiled with:")
    print(f"  - Optimizer: Adam (lr={config.PHASE1_LR})")
    print(f"  - Loss: categorical_crossentropy")
    print(f"  - Metrics: accuracy, precision, recall, top-3 accuracy")
    print()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'phase1_best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured:")
    print(f"  - ModelCheckpoint (save best model based on val_accuracy)")
    print(f"  - EarlyStopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"  - ReduceLROnPlateau (patience={config.REDUCE_LR_PATIENCE})")
    print()
    
    print("🎯 Starting Phase 1 Training...")
    print("-" * 70)
    
    # Train
    history_phase1 = model.fit(
        train_gen,
        epochs=config.PHASE1_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print()
    print("✓ Phase 1 Training Complete!")
    print("=" * 70)
    print()
    
    return history_phase1

# ========================================
# TRAINING - PHASE 2: FINE-TUNING
# ========================================
def train_phase2(model, base_model, train_gen, val_gen, class_weights):
    """
    Phase 2: Fine-Tuning
    - Unfreeze top N layers of EfficientNet
    - Train with lower learning rate (1e-5)
    - Improve accuracy through fine-tuning
    """
    print("=" * 70)
    print("🔥 PHASE 2: FINE-TUNING")
    print("=" * 70)
    print(f"Strategy: Unfreeze top {config.UNFREEZE_LAYERS} layers of EfficientNet")
    print(f"Epochs: {config.PHASE2_EPOCHS}")
    print(f"Learning Rate: {config.PHASE2_LR}")
    print(f"Batch Size: {config.PHASE2_BATCH_SIZE}")
    print()
    
    # Unfreeze top layers
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = total_layers - config.UNFREEZE_LAYERS
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    frozen_count = len(base_model.layers) - trainable_count
    
    print(f"✓ Total layers in base model: {total_layers}")
    print(f"✓ Frozen layers: {frozen_count}")
    print(f"✓ Trainable layers: {trainable_count}")
    print()
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    )
    
    print("✓ Model recompiled with:")
    print(f"  - Optimizer: Adam (lr={config.PHASE2_LR})")
    print(f"  - Loss: categorical_crossentropy")
    print(f"  - Metrics: accuracy, precision, recall, top-3 accuracy")
    print()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            config.MODEL_OUTPUT,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured:")
    print(f"  - ModelCheckpoint (save to: {config.MODEL_OUTPUT})")
    print(f"  - EarlyStopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"  - ReduceLROnPlateau (patience={config.REDUCE_LR_PATIENCE})")
    print()
    
    print("🎯 Starting Phase 2 Training...")
    print("-" * 70)
    
    # Train
    history_phase2 = model.fit(
        train_gen,
        epochs=config.PHASE2_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print()
    print("✓ Phase 2 Training Complete!")
    print("=" * 70)
    print()
    
    return history_phase2

# ========================================
# EVALUATION & VISUALIZATION
# ========================================
def plot_training_history(history_phase1, history_phase2):
    """Plot training metrics for both phases"""
    print("📊 Generating training visualizations...")
    
    # Combine histories
    phase1_epochs = len(history_phase1.history['accuracy'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Two-Phase Training', fontsize=16, fontweight='bold')
    
    # Prepare data
    metrics = {
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Phase 1
        train_metric_1 = history_phase1.history.get(metric_key, [])
        val_metric_1 = history_phase1.history.get(f'val_{metric_key}', [])
        
        # Phase 2
        train_metric_2 = history_phase2.history.get(metric_key, [])
        val_metric_2 = history_phase2.history.get(f'val_{metric_key}', [])
        
        # Combine
        train_metric = train_metric_1 + train_metric_2
        val_metric = val_metric_1 + val_metric_2
        
        epochs = range(1, len(train_metric) + 1)
        
        # Plot
        ax.plot(epochs, train_metric, 'b-', label=f'Training {metric_name}', linewidth=2)
        ax.plot(epochs, val_metric, 'r-', label=f'Validation {metric_name}', linewidth=2)
        ax.axvline(x=phase1_epochs, color='green', linestyle='--', label='Phase 1 → Phase 2', linewidth=2)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training plots saved to: training_history.png")
    print()

def evaluate_model(model, val_gen):
    """Final evaluation on validation set"""
    print("=" * 70)
    print("📊 FINAL MODEL EVALUATION")
    print("=" * 70)
    
    results = model.evaluate(val_gen, verbose=0)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'top3_accuracy']
    
    print("Validation Set Performance:")
    print("-" * 70)
    for metric_name, value in zip(metrics, results):
        print(f"  {metric_name.capitalize():20s}: {value:.4f}")
    print()
    
    return dict(zip(metrics, results))

# ========================================
# MAIN TRAINING PIPELINE
# ========================================
def main():
    """Main training pipeline - RESTART FROM SCRATCH"""
    
    # Setup
    setup_environment()
    
    # Data Pipeline
    train_gen, val_gen, class_indices = create_data_generators()
    
    # Class Weights
    class_weights = compute_class_weights(train_gen)
    
    # Build Model
    model, base_model = build_model(config.NUM_CLASSES)
    
    print("📋 MODEL SUMMARY")
    print("=" * 70)
    model.summary()
    print()
    
    # Phase 1: Feature Extraction
    history_phase1 = train_phase1(model, train_gen, val_gen, class_weights)
    
    # Phase 2: Fine-Tuning
    history_phase2 = train_phase2(model, base_model, train_gen, val_gen, class_weights)
    
    # Visualization
    plot_training_history(history_phase1, history_phase2)
    
    # Final Evaluation
    final_metrics = evaluate_model(model, val_gen)
    
    # Save Training History
    combined_history = {
        'phase1': {k: [float(v) for v in vals] for k, vals in history_phase1.history.items()},
        'phase2': {k: [float(v) for v in vals] for k, vals in history_phase2.history.items()},
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'config': {
            'architecture': 'EfficientNet-B0',
            'input_shape': list(config.INPUT_SHAPE),
            'num_classes': config.NUM_CLASSES,
            'phase1_epochs': config.PHASE1_EPOCHS,
            'phase2_epochs': config.PHASE2_EPOCHS,
            'phase1_lr': config.PHASE1_LR,
            'phase2_lr': config.PHASE2_LR,
            'batch_size': config.PHASE1_BATCH_SIZE,
            'dropout_rate': config.DROPOUT_RATE,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(config.HISTORY_OUTPUT, 'w') as f:
        json.dump(combined_history, f, indent=2)
    
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Model saved to: {config.MODEL_OUTPUT}")
    print(f"✓ Labels saved to: {config.LABELS_OUTPUT}")
    print(f"✓ History saved to: {config.HISTORY_OUTPUT}")
    print(f"✓ Plot saved to: training_history.png")
    print()
    print("📦 DEPLOYMENT READY:")
    print(f"  - Load model: keras.models.load_model('{config.MODEL_OUTPUT}')")
    print(f"  - No custom objects required")
    print(f"  - FastAPI compatible")
    print("=" * 70)

if __name__ == '__main__':
    main()
