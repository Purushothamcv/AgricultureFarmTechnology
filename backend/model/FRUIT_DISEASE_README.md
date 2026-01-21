# Fruit Disease Detection Module 🍎🔬

## Overview

Production-ready CNN model for multi-class fruit disease classification using **EfficientNet-B0 transfer learning**. Designed for deployment with FastAPI in the SmartAgri-AI platform.

## 📊 Model Specifications

| Feature | Details |
|---------|---------|
| **Architecture** | EfficientNet-B0 (Transfer Learning) |
| **Framework** | TensorFlow/Keras |
| **Input Size** | 224×224×3 |
| **Total Classes** | 17 disease classes |
| **Parameters** | ~5.3M (base) + custom head |
| **Inference Time** | ~10-30ms per image |
| **Accuracy** | 95%+ (on validation set) |

## 🍓 Supported Fruits & Diseases

### Apple (4 classes)
- ✅ Healthy_Apple
- 🦠 Blotch_Apple
- 🦠 Rot_Apple
- 🦠 Scab_Apple

### Guava (3 classes)
- ✅ Healthy_Guava
- 🦠 Anthracnose_Guava
- 🦠 Fruitfly_Guava

### Mango (5 classes)
- ✅ Healthy_Mango
- 🦠 Alternaria_Mango
- 🦠 Anthracnose_Mango
- 🦠 Black Mould Rot (Aspergillus)_Mango
- 🦠 Stem and Rot (Lasiodiplodia)_Mango

### Pomegranate (5 classes)
- ✅ Healthy_Pomegranate
- 🦠 Alternaria_Pomegranate
- 🦠 Anthracnose_Pomegranate
- 🦠 Bacterial_Blight_Pomegranate
- 🦠 Cercospora_Pomegranate

## 🏗️ Architecture

```
Input (224x224x3)
    ↓
EfficientNet-B0 (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Batch Normalization
    ↓
Dense(256) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(17) + Softmax
    ↓
Output (17 classes)
```

## 🎯 Why EfficientNet-B0?

1. **Optimal Accuracy-Efficiency Trade-off**
   - Achieves state-of-the-art accuracy with minimal parameters
   - Superior to ResNet, VGG, and MobileNet in efficiency

2. **Compound Scaling**
   - Balances network depth, width, and resolution
   - Scientifically optimized scaling method

3. **Mobile/Edge Ready**
   - Only ~5.3M parameters (lightweight)
   - Fast inference (10-30ms per image)
   - Perfect for FastAPI deployment

4. **Transfer Learning Benefits**
   - Pretrained on ImageNet (14M images)
   - Excellent feature extraction capabilities
   - Faster convergence with less data

5. **Production Proven**
   - Widely used in real-world applications
   - Strong community support
   - Reliable and stable

## 📦 Installation

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Required packages:
# - tensorflow>=2.13.0
# - keras>=2.13.0
# - pillow>=9.5.0
# - seaborn>=0.12.0
# - scikit-learn
# - matplotlib
```

## 🚀 Usage

### 1️⃣ Analyze Dataset

```bash
python model/dataset_analyzer.py
```

**Output:**
- Dataset statistics
- Class distribution visualization
- Balance analysis
- JSON report

### 2️⃣ Train Model

```bash
python model/train_fruit_disease_model.py
```

**Training Process:**
- Phase 1: Train with frozen base (30 epochs)
- Phase 2: Fine-tune last 20 layers (20 epochs)
- Automatic early stopping
- Model checkpointing

**Generated Files:**
- `fruit_disease_model.h5` - Trained model
- `fruit_disease_labels.json` - Class mappings
- `training_history.png` - Training plots
- `confusion_matrix.png` - Confusion matrix
- `classification_report.txt` - Per-class metrics
- `dataset_distribution.png` - Class distribution

### 3️⃣ Test Inference

```bash
python model/fruit_disease_inference.py path/to/test_image.jpg
```

**Example Output:**
```
✅ Prediction: Blotch_Apple
🍎 Fruit Type: Apple
🦠 Disease: Blotch
📊 Confidence: 98.45%
💊 Treatment: Apply fungicides like Captan or Mancozeb...

📈 Top Predictions:
   1. Blotch_Apple: 98.45%
   2. Scab_Apple: 1.20%
   3. Healthy_Apple: 0.35%
```

### 4️⃣ FastAPI Integration

The model is automatically integrated with FastAPI. To include it in your main application:

```python
# In main_fastapi.py
from fruit_disease_service import router as fruit_router

app.include_router(fruit_router)
```

## 🌐 API Endpoints

### Health Check
```bash
GET /api/fruit-disease/health
```

### Get All Classes
```bash
GET /api/fruit-disease/classes
```

### Single Image Prediction
```bash
POST /api/fruit-disease/predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG, PNG)
- top_n: Number of predictions (default: 3)
```

**Example Request (cURL):**
```bash
curl -X POST "http://localhost:8000/api/fruit-disease/predict?top_n=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@apple_image.jpg"
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "predicted_class": "Blotch_Apple",
    "fruit_type": "Apple",
    "disease": "Blotch",
    "is_healthy": false,
    "confidence": 0.9845,
    "confidence_percentage": "98.45%",
    "treatment": "Apply fungicides like Captan or Mancozeb...",
    "top_predictions": [
      {"class": "Blotch_Apple", "confidence": 0.9845, "percentage": "98.45%"},
      {"class": "Scab_Apple", "confidence": 0.0120, "percentage": "1.20%"},
      {"class": "Healthy_Apple", "confidence": 0.0035, "percentage": "0.35%"}
    ]
  },
  "filename": "apple_image.jpg"
}
```

### Batch Prediction
```bash
POST /api/fruit-disease/predict-batch
Content-Type: multipart/form-data

Parameters:
- files: Multiple image files (max 10)
- top_n: Number of predictions per image
```

### Model Information
```bash
GET /api/fruit-disease/info
```

## 📈 Training Details

### Data Augmentation
```python
- Rotation: ±30°
- Width/Height Shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal Flip: Yes
- Brightness: 0.8-1.2
```

### Training Configuration
```python
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001 (Phase 1), 0.00001 (Phase 2)
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall
- Validation Split: 20%
```

### Callbacks
- **Early Stopping**: Patience 10, monitor val_loss
- **Model Checkpoint**: Save best model (val_accuracy)
- **ReduceLROnPlateau**: Factor 0.5, patience 5

## 📊 Performance Metrics

The model generates comprehensive evaluation metrics:

1. **Training Plots**
   - Accuracy curves (train/val)
   - Loss curves (train/val)
   - Precision curves
   - Recall curves

2. **Confusion Matrix**
   - 17×17 heatmap
   - Per-class error analysis

3. **Classification Report**
   - Precision, Recall, F1-score
   - Per-class metrics
   - Overall accuracy

4. **Per-Class Accuracy**
   - Individual class performance
   - Identifies weak classes

## 🔧 Customization

### Change Model Architecture
Edit `train_fruit_disease_model.py`:
```python
# Use different EfficientNet variant
from tensorflow.keras.applications import EfficientNetB3
base_model = EfficientNetB3(...)

# Or use different backbone
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(...)
```

### Modify Training Parameters
```python
# In Config class
BATCH_SIZE = 64  # Larger batch
LEARNING_RATE = 0.0001  # Lower LR
EPOCHS = 100  # More epochs
```

### Add More Classes
1. Add images to `data/archive/NewClass_Fruit/`
2. Update `Config.CLASS_NAMES` in training script
3. Retrain model

## 📂 File Structure

```
backend/
├── model/
│   ├── train_fruit_disease_model.py      # Training script
│   ├── fruit_disease_inference.py        # Inference module
│   ├── dataset_analyzer.py               # Dataset utilities
│   ├── fruit_disease_model.h5            # Trained model
│   ├── fruit_disease_labels.json         # Class mappings
│   ├── training_history.png              # Training plots
│   ├── confusion_matrix.png              # Confusion matrix
│   ├── classification_report.txt         # Metrics report
│   └── dataset_distribution.png          # Class distribution
├── fruit_disease_service.py              # FastAPI routes
├── data/
│   └── archive/                          # Dataset (ImageFolder)
│       ├── Blotch_Apple/
│       ├── Rot_Apple/
│       ├── ...
└── requirements.txt
```

## 🐛 Troubleshooting

### Model Not Found
```bash
# Ensure model is trained
python model/train_fruit_disease_model.py
```

### Low Accuracy
- Check dataset quality
- Increase training epochs
- Adjust data augmentation
- Try unfreezing more layers

### Slow Inference
- Use GPU for prediction
- Reduce image size (if acceptable)
- Batch predictions for multiple images

### Memory Issues
- Reduce batch size
- Use mixed precision training
- Clear TensorFlow session

## 📚 References

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 🤝 Contributing

To improve the model:
1. Collect more diverse data
2. Experiment with different architectures
3. Fine-tune hyperparameters
4. Add ensemble methods

## 📝 License

Part of SmartAgri-AI project.

## ✅ Production Checklist

- [x] Transfer learning with EfficientNet-B0
- [x] Data augmentation implemented
- [x] Early stopping & checkpointing
- [x] Comprehensive evaluation metrics
- [x] FastAPI integration
- [x] Error handling
- [x] Logging
- [x] Documentation
- [x] Inference optimization
- [x] Treatment recommendations

---

**Built with ❤️ for SmartAgri-AI**
