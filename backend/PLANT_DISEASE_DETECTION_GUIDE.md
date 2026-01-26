# Plant Leaf Disease Detection - Implementation Guide

## 🎯 Overview

A production-ready plant leaf disease detection system integrated into the SmartAgri FastAPI backend. The system uses a pre-trained CNN model (`plant_disease_prediction_model.h5`) and dynamically extracts disease classes from the dataset folder structure.

---

## 📁 Project Structure

```
backend/
├── main_fastapi.py                    # Main FastAPI application
├── plant_disease_service.py           # Plant disease detection service (NEW)
├── test_plant_disease.py              # Test script (NEW)
├── model/
│   └── plant_disease_prediction_model.h5    # Trained model
└── data/
    └── plant-village dataset/
        └── plantvillage dataset/
            └── color/
                ├── Apple___Apple_scab/
                ├── Tomato___Late_blight/
                └── ... (38 disease classes)
```

---

## 🚀 Quick Start

### 1. Start the Server

```bash
cd backend
uvicorn main_fastapi:app --reload --port 8000
```

### 2. Verify Service Health

```bash
curl http://localhost:8000/predict/plant-disease/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_classes": 38,
  "image_size": [224, 224],
  "sample_classes": ["Apple___Apple_scab", "Apple___Black_rot", ...]
}
```

### 3. Test Prediction

```bash
cd backend
python test_plant_disease.py
```

---

## 🔌 API Endpoints

### 1. Predict Plant Disease

**POST** `/predict/plant-disease`

Upload an image and get disease prediction.

#### Request

- **Content-Type:** `multipart/form-data`
- **Body:** `file` (image file - JPG, PNG)

#### cURL Example

```bash
curl -X POST http://localhost:8000/predict/plant-disease \
  -F "file=@path/to/leaf_image.jpg"
```

#### Python Example

```python
import requests

url = "http://localhost:8000/predict/plant-disease"
files = {'file': open('leaf_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

#### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/predict/plant-disease', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

#### Response

```json
{
  "plant": "Tomato",
  "prediction": "Tomato___Late_blight",
  "confidence": 0.91,
  "top_3": [
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.91
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.06
    },
    {
      "class": "Tomato___healthy",
      "confidence": 0.03
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `plant` | string | Plant name (extracted from class) |
| `prediction` | string | Full disease class name |
| `confidence` | float | Confidence score (0.0 - 1.0) |
| `top_3` | array | Top 3 predictions with confidence scores |

### 2. Health Check

**GET** `/predict/plant-disease/health`

Check service status and configuration.

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_classes": 38,
  "image_size": [224, 224],
  "sample_classes": ["Apple___Apple_scab", "Apple___Black_rot", ...]
}
```

---

## 🏗️ Architecture

### Component Overview

```
┌─────────────────┐
│   Frontend      │
│  (React/Vue)    │
└────────┬────────┘
         │ HTTP POST /predict/plant-disease
         │ (multipart/form-data)
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│  ┌───────────────────────────────────┐  │
│  │  plant_disease_service.py         │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │ 1. Validate image format    │  │  │
│  │  │ 2. Preprocess image         │  │  │
│  │  │ 3. Run model prediction     │  │  │
│  │  │ 4. Map indices to classes   │  │  │
│  │  │ 5. Format response          │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ├──► Model: plant_disease_prediction_model.h5
         └──► Dataset: data/plant-village dataset/
```

### Service Initialization Flow

```
Application Startup
    │
    ├─► 1. Extract class names from dataset folders
    │       └─► Sort alphabetically (matches training order)
    │
    ├─► 2. Create index → class name mapping
    │       └─► {0: "Apple___Apple_scab", 1: "Apple___Black_rot", ...}
    │
    └─► 3. Load trained model
            └─► Validate: model classes == dataset classes
```

### Prediction Pipeline

```
Image Upload
    │
    ├─► 1. Read image file
    ├─► 2. Convert to PIL Image
    ├─► 3. Preprocess:
    │       ├─► Convert to RGB
    │       ├─► Resize to 224x224
    │       ├─► Normalize to [0, 1]
    │       └─► Add batch dimension
    │
    ├─► 4. Model prediction
    │       └─► Output: [0.02, 0.91, 0.01, ..., 0.04]
    │
    ├─► 5. Get top K predictions
    │       └─► Sort by confidence
    │
    └─► 6. Map indices to class names
            └─► Return formatted response
```

---

## 🧪 Testing

### Automated Testing

```bash
cd backend
python test_plant_disease.py
```

**Features:**
- ✅ Health check verification
- ✅ Multi-image prediction testing
- ✅ Error handling validation
- ✅ Comprehensive test summary

### Interactive Testing

```bash
python test_plant_disease.py -i
```

Enter image paths interactively to test predictions.

### Manual Testing with cURL

```bash
# Find a test image
cd "data/plant-village dataset/plantvillage dataset/color/Tomato___Late_blight"

# Get first image file
$image = (Get-ChildItem -Filter *.jpg | Select-Object -First 1).Name

# Test prediction
curl -X POST http://localhost:8000/predict/plant-disease `
  -F "file=@$image"
```

---

## 🔧 Configuration

### Model Configuration

Located in [plant_disease_service.py](plant_disease_service.py):

```python
MODEL_PATH = "model/plant_disease_prediction_model.h5"
DATASET_PATH = "data/plant-village dataset/plantvillage dataset/color"
IMAGE_SIZE = (224, 224)
TOP_K_PREDICTIONS = 3
```

### Adjusting Image Size

If your model uses a different input size:

```python
# Change this in plant_disease_service.py
IMAGE_SIZE = (299, 299)  # For InceptionV3
# or
IMAGE_SIZE = (331, 331)  # For NASNet
```

### Changing Top-K Predictions

```python
TOP_K_PREDICTIONS = 5  # Return top 5 predictions
```

---

## 📊 Supported Disease Classes

The system automatically detects all classes from the dataset folder structure. Currently supports **38 disease classes**:

### Apple Diseases
- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust
- Apple___healthy

### Tomato Diseases
- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Septoria_leaf_spot
- Tomato___Spider_mites Two-spotted_spider_mite
- Tomato___Target_Spot
- Tomato___Tomato_Yellow_Leaf_Curl_Virus

### Potato Diseases
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

### Other Plants
- Blueberry___healthy
- Cherry_(including_sour)___healthy
- Cherry_(including_sour)___Powdery_mildew
- Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
- Corn_(maize)___Common_rust_
- Corn_(maize)___healthy
- Corn_(maize)___Northern_Leaf_Blight
- Grape___Black_rot
- Grape___Esca_(Black_Measles)
- Grape___healthy
- Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
- Orange___Haunglongbing_(Citrus_greening)
- Peach___Bacterial_spot
- Peach___healthy
- Pepper,_bell___Bacterial_spot
- Pepper,_bell___healthy
- Raspberry___healthy
- Soybean___healthy
- Squash___Powdery_mildew
- Strawberry___healthy
- Strawberry___Leaf_scorch

> **Note:** Class names are automatically extracted from dataset folder names. No hardcoding required!

---

## 🐛 Troubleshooting

### Issue: "Model file not found"

**Solution:**
```bash
# Verify model exists
ls backend/model/plant_disease_prediction_model.h5

# If missing, check for alternative names:
ls backend/model/plant_disease*
```

### Issue: "Dataset directory not found"

**Solution:**
```bash
# Verify dataset exists
ls "backend/data/plant-village dataset/plantvillage dataset/color"

# Update path in plant_disease_service.py if different
```

### Issue: "Model classes don't match dataset"

**Symptoms:** Warning during startup
```
⚠️  Model output classes (38) != Dataset classes (35)
```

**Solution:**
1. Verify dataset folder structure
2. Ensure all training classes are present
3. Check for duplicate/missing folders

### Issue: "Low prediction confidence"

**Possible Causes:**
1. Image preprocessing mismatch
2. Poor image quality
3. Out-of-distribution samples

**Solution:**
1. Verify preprocessing matches training
2. Test with known good images from dataset
3. Check image is clear and focused on leaf

### Issue: "CORS error from frontend"

**Solution:** Add frontend domain to CORS origins in [main_fastapi.py](main_fastapi.py):

```python
allowed_origins = [
    "http://localhost:3000",
    "https://your-frontend-domain.com",
]
```

---

## 🔒 Error Handling

The API returns appropriate HTTP status codes and error messages:

### 400 Bad Request
- Invalid file type (non-image)
- Corrupt image file

```json
{
  "detail": "Invalid file type: text/plain. Please upload an image file (JPG, PNG)."
}
```

### 500 Internal Server Error
- Model prediction failure
- Preprocessing error

```json
{
  "detail": "Prediction failed: ..."
}
```

### 503 Service Unavailable
- Model not loaded
- Service initialization failed

```json
{
  "detail": "Plant disease model not initialized. Please restart the server."
}
```

---

## 📈 Performance Considerations

### Model Loading
- ✅ Model loaded ONCE at startup
- ✅ No reloading on each request
- ✅ Minimal memory footprint

### Image Processing
- ✅ Efficient PIL Image operations
- ✅ Single batch prediction
- ✅ No disk I/O during prediction

### Typical Response Times
- Health check: ~10ms
- Prediction: ~100-500ms (depending on model size)

---

## 🔄 Frontend Integration

### React Example

```jsx
import { useState } from 'react';

function PlantDiseaseDetector() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict/plant-disease', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      
      {loading && <p>Analyzing image...</p>}
      
      {result && (
        <div>
          <h3>Plant: {result.plant}</h3>
          <h4>Disease: {result.prediction}</h4>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          
          <h4>Top Predictions:</h4>
          <ul>
            {result.top_3.map((pred, idx) => (
              <li key={idx}>
                {pred.class}: {(pred.confidence * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default PlantDiseaseDetector;
```

### Vue.js Example

```vue
<template>
  <div>
    <input type="file" accept="image/*" @change="handleImageUpload" />
    
    <div v-if="loading">Analyzing image...</div>
    
    <div v-if="result">
      <h3>Plant: {{ result.plant }}</h3>
      <h4>Disease: {{ result.prediction }}</h4>
      <p>Confidence: {{ (result.confidence * 100).toFixed(2) }}%</p>
      
      <h4>Top Predictions:</h4>
      <ul>
        <li v-for="(pred, idx) in result.top_3" :key="idx">
          {{ pred.class }}: {{ (pred.confidence * 100).toFixed(2) }}%
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      result: null,
      loading: false
    };
  },
  methods: {
    async handleImageUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      this.loading = true;
      try {
        const response = await fetch('http://localhost:8000/predict/plant-disease', {
          method: 'POST',
          body: formData
        });
        
        this.result = await response.json();
      } catch (error) {
        console.error('Prediction failed:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>
```

---

## ✅ Implementation Checklist

- [x] Model loading at startup
- [x] Dynamic class extraction from dataset
- [x] Image preprocessing pipeline
- [x] FastAPI endpoint creation
- [x] Response formatting
- [x] CORS configuration
- [x] Error handling
- [x] Health check endpoint
- [x] Comprehensive testing script
- [x] Documentation

---

## 📝 Key Features

✅ **No Hardcoded Classes** - Dynamically extracted from dataset folders  
✅ **Production Ready** - Efficient, error-handled, well-documented  
✅ **Frontend Ready** - CORS enabled, clean JSON responses  
✅ **Easy Testing** - Comprehensive test script included  
✅ **Maintainable** - Clean code structure, well-commented  
✅ **Scalable** - Single model load, efficient predictions  

---

## 🎓 Next Steps

1. **Deploy to Production**
   - Use production ASGI server (Gunicorn + Uvicorn)
   - Configure proper CORS for production domains
   - Set up logging and monitoring

2. **Enhance Frontend**
   - Add image preview before upload
   - Show confidence meters
   - Display treatment recommendations

3. **Optimize Performance**
   - Implement request queuing
   - Add response caching
   - Use GPU acceleration if available

4. **Add Features**
   - Multiple image batch processing
   - Historical prediction tracking
   - Disease information database

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Run test script: `python test_plant_disease.py`
3. Check logs during server startup
4. Verify model and dataset paths

---

**Implementation Date:** January 2026  
**Status:** ✅ Complete and Production Ready
