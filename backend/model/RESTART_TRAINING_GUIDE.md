# 🚀 Fruit Disease CNN Training - RESTART GUIDE

## Overview
This guide provides instructions for **completely restarting** the fruit disease CNN training from scratch. No checkpoints, no old optimizer states - a clean, fresh start.

---

## 📋 Prerequisites

### 1. Dataset Structure
```
backend/data/archive/
├── Alternaria_Mango/
├── Alternaria_Pomegranate/
├── Anthracnose_Guava/
├── Anthracnose_Mango/
├── Anthracnose_Pomegranate/
├── Bacterial_Blight_Pomegranate/
├── Black Mould Rot (Aspergillus)_Mango/
├── Blotch_Apple/
├── Cercospora_Pomegranate/
├── Fruitfly_Guava/
├── Healthy_Apple/
├── Healthy_Guava/
├── Healthy_Mango/
├── Healthy_Pomegranate/
├── Rot_Apple/
├── Scab_Apple/
└── Stem and Rot (Lasiodiplodia)_Mango/
```
**Total: 17 classes** (ImageFolder structure)

### 2. Required Dependencies
```bash
pip install tensorflow>=2.13.0
pip install keras>=2.13.0
pip install pillow>=9.5.0
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install numpy
```

---

## 🎯 Quick Start (3 Steps)

### Step 1: Navigate to Model Directory
```bash
cd backend/model
```

### Step 2: Run Training Script
```bash
python train_fruit_disease_clean.py
```

### Step 3: Wait for Completion
- **Phase 1**: ~20 epochs (feature extraction)
- **Phase 2**: ~30 epochs (fine-tuning)
- **Total Time**: 1-3 hours (depends on GPU/CPU)

---

## 🏗️ Architecture Details

### Base Model: EfficientNet-B0
- **Pretrained on**: ImageNet
- **Input size**: 224×224×3
- **Parameters**: ~5.3M
- **Why EfficientNet?**
  - State-of-the-art accuracy
  - Fast inference (~10-30ms)
  - Perfect for production deployment
  - Better than ResNet/VGG/MobileNet

### Custom Classification Head
```python
GlobalAveragePooling2D
↓
Dense(256, activation='relu')
↓
Dropout(0.45)
↓
Dense(17, activation='softmax')
```

---

## 📊 Two-Phase Training Strategy

### **Phase 1: Feature Extraction** (Epochs 1-20)
| Parameter | Value |
|-----------|-------|
| Base Model | **FROZEN** (all layers) |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Focus | Train classification head |

**What happens:**
- EfficientNet layers are frozen (weights unchanged)
- Only the custom classification head learns
- Higher learning rate for faster convergence
- Establishes good feature representations

### **Phase 2: Fine-Tuning** (Epochs 21-50)
| Parameter | Value |
|-----------|-------|
| Base Model | **PARTIALLY UNFROZEN** (top 30 layers) |
| Learning Rate | 1e-5 |
| Optimizer | Adam |
| Focus | Adapt features to fruit diseases |

**What happens:**
- Top 30 layers of EfficientNet become trainable
- Lower learning rate to prevent catastrophic forgetting
- Model adapts ImageNet features to fruit disease domain
- Achieves higher accuracy

---

## 🎨 Data Augmentation

### Training Augmentations (STRONG)
- **Rotation**: ±40°
- **Width/Height Shift**: 30%
- **Zoom**: 30%
- **Horizontal Flip**: Yes
- **Vertical Flip**: Yes
- **Brightness**: [0.7, 1.3]
- **Shear**: 20%

### Validation Augmentations
- **None** (only EfficientNet preprocessing)

---

## ⚖️ Class Imbalance Handling

The script automatically:
1. Computes class weights using `sklearn`
2. Applies weights during training
3. Prevents bias toward majority classes

**Example:**
- Class with 79 samples → higher weight
- Class with 1450 samples → lower weight

---

## 📦 Output Files

After training completes, you'll have:

| File | Description |
|------|-------------|
| `fruit_disease_model.h5` | **Final trained model** (deploy this) |
| `fruit_disease_labels.json` | Class index → label mapping |
| `training_history.json` | Complete training metrics |
| `training_history.png` | Visualization of training |
| `phase1_best_model.h5` | Best model from Phase 1 |

---

## 🔧 Configuration Options

Edit the `Config` class in `train_fruit_disease_clean.py`:

```python
class Config:
    # Paths
    DATASET_PATH = '../data/archive'
    MODEL_OUTPUT = 'fruit_disease_model.h5'
    
    # Training - Phase 1
    PHASE1_EPOCHS = 20
    PHASE1_LR = 1e-3
    PHASE1_BATCH_SIZE = 32
    
    # Training - Phase 2
    PHASE2_EPOCHS = 30
    PHASE2_LR = 1e-5
    UNFREEZE_LAYERS = 30
    
    # Regularization
    DROPOUT_RATE = 0.45
    DENSE_UNITS = 256
```

---

## 📈 Expected Performance

### Target Metrics
- **Validation Accuracy**: > 92%
- **Top-3 Accuracy**: > 98%
- **Precision**: > 90%
- **Recall**: > 90%

### Training Progress
```
Epoch 1-5:   Rapid improvement (60-75% accuracy)
Epoch 6-15:  Steady growth (75-88% accuracy)
Epoch 16-20: Plateau Phase 1 (88-91% accuracy)
[SWITCH TO PHASE 2]
Epoch 21-30: Fine-tuning (91-94% accuracy)
Epoch 31-50: Refinement (94-95%+ accuracy)
```

---

## 🚀 FastAPI Integration

### Load Model in FastAPI
```python
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model('model/fruit_disease_model.h5')

with open('model/fruit_disease_labels.json', 'r') as f:
    labels = json.load(f)

# Prediction function
def predict_fruit_disease(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [(labels[str(idx)], float(predictions[0][idx])) for idx in top3_idx]
    
    return {
        'predicted_class': labels[str(predicted_class)],
        'confidence': float(confidence),
        'top3_predictions': top3
    }
```

---

## 🛠️ Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size
```python
PHASE1_BATCH_SIZE = 16  # Instead of 32
PHASE2_BATCH_SIZE = 16
```

### Issue: Training Too Slow
**Solutions**:
1. Reduce epochs: `PHASE1_EPOCHS = 10`, `PHASE2_EPOCHS = 15`
2. Increase batch size (if you have GPU): `BATCH_SIZE = 64`
3. Use GPU: Install `tensorflow-gpu`

### Issue: Low Accuracy (<85%)
**Solutions**:
1. Increase Phase 2 epochs: `PHASE2_EPOCHS = 40`
2. Unfreeze more layers: `UNFREEZE_LAYERS = 50`
3. Check dataset quality (corrupted images?)

### Issue: Overfitting (train >> val accuracy)
**Solutions**:
1. Increase dropout: `DROPOUT_RATE = 0.6`
2. Stronger augmentation (already strong in script)
3. Reduce model complexity: `DENSE_UNITS = 128`

### Issue: Model File Corrupted
**Solution**: Use best model from Phase 1
```python
model = tf.keras.models.load_model('phase1_best_model.h5')
```

---

## ✅ Validation Checklist

Before considering training complete:

- [ ] Training completed without errors
- [ ] `fruit_disease_model.h5` file exists
- [ ] `fruit_disease_labels.json` file exists
- [ ] Validation accuracy > 90%
- [ ] No NaN values in training metrics
- [ ] Model loads successfully with `keras.models.load_model()`
- [ ] Test prediction works correctly
- [ ] Training plots show convergence

---

## 🎓 Design Choices Explained

### Why EfficientNet-B0?
- **Accuracy**: Outperforms ResNet50/VGG16
- **Efficiency**: 5.3M params vs 25M (ResNet50)
- **Speed**: Fast inference for production
- **Pretrained**: Leverages ImageNet knowledge

### Why Two-Phase Training?
- **Phase 1**: Quickly learns disease patterns without disturbing pretrained features
- **Phase 2**: Adapts generic features to specific fruit disease domain
- **Result**: Higher accuracy than single-phase training

### Why Strong Augmentation?
- **17 classes**: Diverse diseases require robust features
- **Class imbalance**: Augmentation helps minority classes
- **Generalization**: Model works on unseen images
- **Overfitting prevention**: Regularization through data

### Why Class Weights?
- **Dataset**: Imbalanced (79-1450 samples per class)
- **Without weights**: Model ignores rare diseases
- **With weights**: All diseases learned equally
- **Result**: Fair predictions across all classes

---

## 📞 Need Help?

### Common Questions

**Q: How long does training take?**
A: 1-3 hours depending on hardware (GPU: 1h, CPU: 3h)

**Q: Can I stop training and resume?**
A: No, this script restarts from scratch. Don't interrupt.

**Q: Can I use a different dataset?**
A: Yes, but ensure ImageFolder structure and update `DATASET_PATH`

**Q: How do I know if training is working?**
A: Watch validation accuracy - should reach >90% by end

**Q: Can I deploy this on mobile?**
A: Yes, convert to TFLite for mobile deployment

---

## 🎯 Next Steps After Training

1. **Test Model**: Run predictions on test images
2. **Deploy to FastAPI**: Integrate with backend
3. **Performance Testing**: Test inference speed
4. **Frontend Integration**: Connect to React frontend
5. **Production Monitoring**: Track prediction accuracy

---

## 📚 Additional Resources

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Class Imbalance Handling](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

---

**🎉 Ready to train? Run the script and watch your model learn!**

```bash
python train_fruit_disease_clean.py
```
