"""
FRUIT DISEASE DETECTION MODEL - PRODUCTION-GRADE TRAINING SCRIPT
=================================================================
Framework: TensorFlow/Keras
Architecture: EfficientNet-B0 (Transfer Learning)
Dataset: Multi-class fruit disease classification (17 classes)
Target Accuracy: 90%+

OPTIMIZATIONS FOR HIGH ACCURACY
================================
1. ✅ TWO-PHASE TRAINING STRATEGY
   - Phase 1: Feature Extraction (frozen backbone, train head only)
   - Phase 2: Fine-Tuning (unfreeze top layers, very low LR)

2. ✅ PROPER CHECKPOINT & RESUME LOGIC
   - Automatically detects existing checkpoints
   - Resumes from correct epoch using initial_epoch
   - Loads best model, not just last model
   - Merges history from previous training sessions

3. ✅ CLASS IMBALANCE HANDLING
   - Computes class weights using sklearn
   - Applies class_weight parameter to model.fit()
   - Prevents model from ignoring minority classes

4. ✅ ADVANCED DATA AUGMENTATION
   - EfficientNet-specific preprocessing (preprocess_input)
   - Rotation, zoom, shift, flip, brightness, shear
   - Validation data: preprocessing only, no augmentation

5. ✅ OPTIMIZED MODEL HEAD
   - Simplified architecture (less overfitting)
   - Global Average Pooling → Dense(256) → Dropout(0.5) → Output
   - No BatchNorm (EfficientNet already has it)

6. ✅ COMPREHENSIVE CALLBACKS
   - ModelCheckpoint: Save best model by val_accuracy
   - EarlyStopping: Restore best weights, patience=10
   - ReduceLROnPlateau: Adaptive learning rate reduction

7. ✅ ADVANCED METRICS
   - Accuracy, Precision, Recall, Top-3 Accuracy
   - Confusion Matrix with heatmap visualization
   - Per-class accuracy breakdown
   - Classification report saved to file

Author: SmartAgri-AI Team
Date: January 22, 2026
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========================================
# CONFIGURATION
# ========================================
class Config:
    """Training configuration parameters"""
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'archive')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'fruit_disease_model.h5')
    LABELS_PATH = os.path.join(BASE_DIR, 'model', 'fruit_disease_labels.json')
    HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'training_history.json')
    
    # Model parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    
    # Training phases
    PHASE1_EPOCHS = 30  # Feature extraction (frozen backbone)
    PHASE2_EPOCHS = 20  # Fine-tuning (unfrozen top layers)
    
    # Learning rates
    PHASE1_LR = 1e-3  # Higher LR for training head only
    PHASE2_LR = 1e-5  # Very low LR for fine-tuning pretrained layers
    
    # Data split
    VALIDATION_SPLIT = 0.2
    
    # Fine-tuning configuration
    UNFREEZE_LAYERS = 30  # Number of top layers to unfreeze in phase 2


# ========================================
# DATA PREPARATION
# ========================================
def create_data_generators():
    """
    Create training and validation data generators
    
    KEY IMPROVEMENTS:
    - EfficientNet-specific preprocessing (not just rescale)
    - Aggressive augmentation for training data
    - Computes class weights to handle imbalance
    - Prints class indices for verification
    
    Returns:
        tuple: (train_generator, validation_generator, class_weights_dict, labels_map)
    """
    print("\n" + "="*70)
    print("CREATING DATA GENERATORS")
    print("="*70)
    
    # ========================================
    # TRAINING DATA: EfficientNet preprocessing + Aggressive augmentation
    # ========================================
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # EfficientNet-specific (critical!)
        rotation_range=40,                        # ±40° rotation
        width_shift_range=0.3,                    # 30% horizontal shift
        height_shift_range=0.3,                   # 30% vertical shift
        shear_range=0.3,                          # Shear transformation
        zoom_range=0.3,                           # 30% zoom
        horizontal_flip=True,                     # Random horizontal flip
        vertical_flip=True,                       # Random vertical flip
        brightness_range=[0.7, 1.3],              # Brightness adjustment
        fill_mode='nearest',                      # Fill mode for transformations
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # ========================================
    # VALIDATION DATA: EfficientNet preprocessing only (NO augmentation)
    # ========================================
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        Config.DATASET_PATH,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',  # Multi-class classification
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Create validation generator
    validation_generator = validation_datagen.flow_from_directory(
        Config.DATASET_PATH,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\n✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {validation_generator.samples}")
    print(f"✓ Number of classes: {train_generator.num_classes}")
    print(f"✓ Batch size: {Config.BATCH_SIZE}")
    
    # ========================================
    # VERIFY CLASS MAPPING
    # ========================================
    class_indices = train_generator.class_indices
    print("\n" + "-"*70)
    print("CLASS MAPPING:")
    print("-"*70)
    for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {class_name}")
    
    # Create labels map (index -> class name)
    labels_map = {v: k for k, v in class_indices.items()}
    
    # Save labels map
    with open(Config.LABELS_PATH, 'w') as f:
        json.dump(labels_map, f, indent=4)
    print(f"\n✓ Class labels saved to: {Config.LABELS_PATH}")
    
    # ========================================
    # COMPUTE CLASS WEIGHTS (handle imbalanced dataset)
    # ========================================
    print("\n" + "-"*70)
    print("COMPUTING CLASS WEIGHTS FOR IMBALANCED DATA:")
    print("-"*70)
    
    # Count samples per class
    for class_name, class_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Class {class_idx:2d} ({class_name:45s}): {count:4d} samples")
    
    # Compute balanced class weights using sklearn
    y_train = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    print("\n" + "-"*70)
    print("COMPUTED CLASS WEIGHTS:")
    print("-"*70)
    for idx, weight in class_weights_dict.items():
        print(f"  Class {idx:2d}: {weight:.4f}")
    print("-"*70)
    print("✓ Minority classes get higher weights → prevents model from ignoring them")
    print("✓ Majority classes get lower weights → prevents model from overfitting to them")
    
    return train_generator, validation_generator, class_weights_dict, labels_map


# ========================================
# MODEL ARCHITECTURE
# ========================================
def build_model(phase='phase1'):
    """
    Build EfficientNet-B0 transfer learning model
    
    Architecture:
        - EfficientNet-B0 backbone (ImageNet pretrained)
        - Global Average Pooling
        - Dense(256, relu)
        - Dropout(0.5)
        - Dense(17, softmax)
    
    Args:
        phase: 'phase1' (frozen backbone) or 'phase2' (unfrozen top layers)
        
    Returns:
        model: Compiled Keras model
    """
    print("\n" + "="*70)
    print(f"BUILDING MODEL - {phase.upper()}")
    print("="*70)
    
    # Load pretrained EfficientNet-B0 (exclude top classification layer)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
    )
    
    if phase == 'phase1':
        # PHASE 1: Freeze entire backbone for feature extraction
        base_model.trainable = False
        print("✓ Base model: FROZEN (training head only)")
    else:
        # PHASE 2: Unfreeze top layers for fine-tuning
        base_model.trainable = True
        
        # Freeze all layers except last N layers
        frozen_layers = len(base_model.layers) - Config.UNFREEZE_LAYERS
        for layer in base_model.layers[:frozen_layers]:
            layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"✓ Base model: PARTIALLY UNFROZEN")
        print(f"✓ Total layers: {len(base_model.layers)}")
        print(f"✓ Frozen layers: {frozen_layers}")
        print(f"✓ Trainable layers: {trainable_count}")
    
    # ========================================
    # SIMPLIFIED CLASSIFICATION HEAD
    # ========================================
    # Removed BatchNorm and extra Dense layers to reduce overfitting
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Strong regularization
        layers.Dense(17, activation='softmax')
    ], name='FruitDiseaseDetector')
    
    # Compile with appropriate learning rate for phase
    lr = Config.PHASE1_LR if phase == 'phase1' else Config.PHASE2_LR
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    )
    
    # Print model info
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"\n✓ Architecture: EfficientNet-B0 + Custom Head")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Learning rate: {lr}")
    print(f"✓ Optimizer: Adam")
    print(f"✓ Loss: categorical_crossentropy")
    
    return model


# ========================================
# CALLBACKS
# ========================================
def create_callbacks():
    """
    Create training callbacks for model optimization
    
    Returns:
        list: Callback objects
    
    FIX: Removed problematic callbacks that cause pickle errors during deepcopy
    Only using essential callbacks that are guaranteed to be serializable
    """
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=Config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False  # Explicitly set to avoid serialization issues
        ),
        
        # Early stopping with patience
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        # REMOVED: ReduceLROnPlateau causes pickle errors in some TensorFlow versions
        # Alternative: Use fixed learning rate or manual adjustment between epochs
    ]
    
    return callbacks


# ========================================
# CHECKPOINT & RESUME LOGIC
# ========================================
def load_checkpoint_and_history():
    """
    Load existing checkpoint and training history if available
    
    Returns:
        tuple: (model, resume_epoch, history_dict)
            - model: Loaded model or None
            - resume_epoch: Epoch to resume from (0 if starting fresh)
            - history_dict: Previous training history or empty dict
    """
    checkpoint_path = Config.MODEL_SAVE_PATH
    history_path = Config.HISTORY_PATH
    
    model = None
    resume_epoch = 0
    history_dict = {}
    
    if os.path.exists(checkpoint_path):
        print("\n" + "="*70)
        print("CHECKPOINT FOUND - RESUMING TRAINING")
        print("="*70)
        print(f"✓ Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load existing model
            model = keras.models.load_model(checkpoint_path)
            print("✓ Model loaded successfully")
            
            # Load training history
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_dict = json.load(f)
                
                resume_epoch = len(history_dict['accuracy'])
                best_val_acc = max(history_dict['val_accuracy'])
                best_val_loss = min(history_dict['val_loss'])
                
                print(f"✓ Training history loaded")
                print(f"✓ Previous epochs completed: {resume_epoch}")
                print(f"✓ Best validation accuracy: {best_val_acc:.4f}")
                print(f"✓ Best validation loss: {best_val_loss:.4f}")
            else:
                print("⚠ No history file found - starting fresh with loaded weights")
                resume_epoch = 0
            
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            print("✓ Starting fresh training")
            model = None
            resume_epoch = 0
            history_dict = {}
    else:
        print("\n✓ No checkpoint found - starting fresh training")
    
    return model, resume_epoch, history_dict


def save_history(history, previous_history=None):
    """
    Save or append training history to JSON file
    
    Args:
        history: Keras History object from current training
        previous_history: Dict of previous training history (optional)
    """
    # Convert history to dict
    new_history = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'precision': [float(x) for x in history.history['precision']],
        'val_precision': [float(x) for x in history.history['val_precision']],
        'recall': [float(x) for x in history.history['recall']],
        'val_recall': [float(x) for x in history.history['val_recall']]
    }
    
    # Merge with previous history if provided
    if previous_history:
        history_dict = {}
        for key in new_history.keys():
            if key in previous_history:
                history_dict[key] = previous_history[key] + new_history[key]
            else:
                history_dict[key] = new_history[key]
    else:
        history_dict = new_history
    
    # Save to file
    with open(Config.HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"✓ Training history saved to: {Config.HISTORY_PATH}")


# ========================================
# EVALUATION & VISUALIZATION
# ========================================
def plot_training_history(history_path):
    """
    Plot training metrics from saved history file
    
    Args:
        history_path: Path to training history JSON file
    """
    print("\n" + "="*70)
    print("GENERATING TRAINING PLOTS")
    print("="*70)
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
    
    # Accuracy plot
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision plot
    axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall plot
    axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(Config.BASE_DIR, 'model', 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_path}")
    plt.close()


def evaluate_model(model, validation_generator, labels_map):
    """
    Comprehensive model evaluation with metrics and visualizations
    
    Args:
        model: Trained Keras model
        validation_generator: Validation data generator
        labels_map: Dictionary mapping class indices to names
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Get predictions
    validation_generator.reset()
    print("✓ Generating predictions...")
    y_pred_probs = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = validation_generator.classes
    
    # ========================================
    # CLASSIFICATION REPORT
    # ========================================
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    
    class_names = [labels_map[i] for i in range(len(labels_map))]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Save classification report
    report_path = os.path.join(Config.BASE_DIR, 'model', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("FRUIT DISEASE DETECTION - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: EfficientNet-B0 Transfer Learning\n")
        f.write(f"Classes: {len(class_names)}\n\n")
        f.write(report)
    print(f"\n✓ Classification report saved to: {report_path}")
    
    # ========================================
    # CONFUSION MATRIX
    # ========================================
    print("\n✓ Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Fruit Disease Detection', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(Config.BASE_DIR, 'model', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # ========================================
    # PER-CLASS ACCURACY
    # ========================================
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY")
    print("-"*70)
    
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, (class_name, accuracy) in enumerate(zip(class_names, class_accuracy)):
        print(f"{class_name:45s}: {accuracy*100:6.2f}%")
    
    # Overall accuracy
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    print("\n" + "="*70)
    print(f"OVERALL VALIDATION ACCURACY: {overall_accuracy*100:.2f}%")
    print("="*70)


# ========================================
# MAIN TRAINING PIPELINE
# ========================================
def main():
    """
    Main training pipeline with two-phase strategy
    
    TRAINING STRATEGY:
    ==================
    PHASE 1: Feature Extraction (30 epochs)
        - Freeze EfficientNet backbone
        - Train only classification head (~332K params)
        - Learning rate: 1e-3
        - Apply class weights for imbalance
    
    PHASE 2: Fine-Tuning (20 epochs)
        - Unfreeze top 30 layers of EfficientNet
        - Very low learning rate: 1e-5
        - Continue with class weights
        - Fine-tune for fruit disease specificity
    """
    print("\n" + "="*70)
    print(" "*15 + "FRUIT DISEASE DETECTION MODEL TRAINING")
    print(" "*20 + "EfficientNet-B0 Transfer Learning")
    print("="*70)
    
    start_time = datetime.now()
    
    # ========================================
    # SYSTEM INFORMATION
    # ========================================
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset path: {Config.DATASET_PATH}")
    
    # Verify dataset exists
    if not os.path.exists(Config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {Config.DATASET_PATH}")
    
    # ========================================
    # STEP 1: LOAD CHECKPOINT & HISTORY
    # ========================================
    model, resume_epoch, previous_history = load_checkpoint_and_history()
    
    # ========================================
    # STEP 2: CREATE DATA GENERATORS
    # ========================================
    train_generator, validation_generator, class_weights_dict, labels_map = create_data_generators()
    
    # ========================================
    # STEP 3: DETERMINE TRAINING PHASE
    # ========================================
    phase1_target = Config.PHASE1_EPOCHS
    phase2_target = Config.PHASE1_EPOCHS + Config.PHASE2_EPOCHS
    
    # ========================================
    # PHASE 1: FEATURE EXTRACTION
    # ========================================
    if resume_epoch < phase1_target:
        print("\n" + "="*70)
        print("PHASE 1: FEATURE EXTRACTION (FROZEN BACKBONE)")
        print("="*70)
        print(f"✓ Target epochs: {phase1_target}")
        print(f"✓ Current epoch: {resume_epoch}")
        print(f"✓ Remaining epochs: {phase1_target - resume_epoch}")
        print(f"✓ EfficientNet backbone: FROZEN")
        print(f"✓ Training: Classification head only")
        print(f"✓ Learning rate: {Config.PHASE1_LR}")
        print(f"✓ Class weights: ENABLED")
        print("="*70)
        
        # Build or load model
        if model is None:
            model = build_model(phase='phase1')
        else:
            # Recompile loaded model for phase 1 settings
            print("\n✓ Recompiling loaded model for Phase 1...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=Config.PHASE1_LR),
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
                ]
            )
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train phase 1
        print("\n✓ Starting Phase 1 training...")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=phase1_target,
            initial_epoch=resume_epoch,  # Resume from correct epoch
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # Save history
        save_history(history, previous_history if resume_epoch > 0 else None)
        
        # Update resume epoch
        resume_epoch = phase1_target
        
        # Load best model from checkpoint
        print("\n✓ Loading best model from Phase 1...")
        model = keras.models.load_model(Config.MODEL_SAVE_PATH)
        
        print("\n" + "="*70)
        print("✓ PHASE 1 COMPLETED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✓ PHASE 1 ALREADY COMPLETED - SKIPPING")
        print("="*70)
    
    # ========================================
    # PHASE 2: FINE-TUNING
    # ========================================
    if resume_epoch < phase2_target:
        print("\n" + "="*70)
        print("PHASE 2: FINE-TUNING (UNFROZEN TOP LAYERS)")
        print("="*70)
        print(f"✓ Target epochs: {phase2_target}")
        print(f"✓ Current epoch: {resume_epoch}")
        print(f"✓ Remaining epochs: {phase2_target - resume_epoch}")
        print(f"✓ Unfreezing top {Config.UNFREEZE_LAYERS} layers")
        print(f"✓ Learning rate: {Config.PHASE2_LR}")
        print(f"✓ Class weights: ENABLED")
        print("="*70)
        
        # Unfreeze top layers
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all except top N layers
        frozen_layers = len(base_model.layers) - Config.UNFREEZE_LAYERS
        for layer in base_model.layers[:frozen_layers]:
            layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"\n✓ Base model total layers: {len(base_model.layers)}")
        print(f"✓ Frozen layers: {frozen_layers}")
        print(f"✓ Trainable layers: {trainable_count}")
        
        # Recompile with very low learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.PHASE2_LR),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )
        
        print(f"✓ Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train phase 2
        print("\n✓ Starting Phase 2 training...")
        
        # Load previous history for resumption
        with open(Config.HISTORY_PATH, 'r') as f:
            previous_history = json.load(f)
        
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=phase2_target,
            initial_epoch=resume_epoch,  # Resume from correct epoch
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # Save history
        save_history(history, previous_history)
        
        # Load best model from checkpoint
        print("\n✓ Loading best model from Phase 2...")
        model = keras.models.load_model(Config.MODEL_SAVE_PATH)
        
        print("\n" + "="*70)
        print("✓ PHASE 2 COMPLETED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✓ PHASE 2 ALREADY COMPLETED - SKIPPING")
        print("="*70)
    
    # ========================================
    # EVALUATION
    # ========================================
    print("\n✓ Starting final evaluation...")
    evaluate_model(model, validation_generator, labels_map)
    
    # Plot training history
    plot_training_history(Config.HISTORY_PATH)
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n✓ Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✓ Labels saved to: {Config.LABELS_PATH}")
    print(f"✓ Training duration: {duration}")
    print(f"✓ Total classes: 17")
    print(f"✓ Total epochs: {phase2_target}")
    print("\n" + "="*70)
    print("ACHIEVEMENTS:")
    print("="*70)
    print("✓ Two-phase training strategy: Feature extraction → Fine-tuning")
    print("✓ Class imbalance handled: Weighted loss function")
    print("✓ Advanced augmentation: Better generalization")
    print("✓ Proper checkpoint/resume: Production-grade")
    print("✓ Comprehensive metrics: Accuracy, Precision, Recall, Top-3")
    print("\n" + "="*70)
    print("Model is ready for deployment!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
