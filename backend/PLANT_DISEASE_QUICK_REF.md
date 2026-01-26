# Plant Disease Detection - Quick Reference

## 🚀 Start Server

```bash
cd backend
uvicorn main_fastapi:app --reload --port 8000
```

## ✅ Health Check

```bash
curl http://localhost:8000/predict/plant-disease/health
```

## 🧪 Run Tests

```bash
cd backend
python test_plant_disease.py
```

## 📤 Test Prediction (cURL)

```bash
curl -X POST http://localhost:8000/predict/plant-disease \
  -F "file=@path/to/image.jpg"
```

## 🔍 Test Prediction (PowerShell)

```powershell
cd backend
$image = "data\plant-village dataset\plantvillage dataset\color\Tomato___Late_blight\00e75bb9-d6e6-4ed0-af15-08a46fa586f1___GHLB2 Leaf 301.JPG"
curl.exe -X POST http://localhost:8000/predict/plant-disease -F "file=@$image"
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/plant-disease` | Upload image for disease prediction |
| GET | `/predict/plant-disease/health` | Check service health |

## 📝 Response Format

```json
{
  "plant": "Tomato",
  "prediction": "Tomato___Late_blight",
  "confidence": 0.91,
  "top_3": [
    {"class": "Tomato___Late_blight", "confidence": 0.91},
    {"class": "Tomato___Early_blight", "confidence": 0.06},
    {"class": "Tomato___healthy", "confidence": 0.03}
  ]
}
```

## 🔧 Configuration Files

- **Service:** `backend/plant_disease_service.py`
- **Main App:** `backend/main_fastapi.py`
- **Test Script:** `backend/test_plant_disease.py`
- **Model:** `backend/model/plant_disease_prediction_model.h5`
- **Dataset:** `backend/data/plant-village dataset/plantvillage dataset/color/`

## 🐛 Common Issues

### Model not found
```bash
# Check if model exists
ls backend/model/plant_disease_prediction_model.h5
```

### Dataset not found
```bash
# Check if dataset exists
ls "backend/data/plant-village dataset/plantvillage dataset/color"
```

### Port already in use
```bash
# Use different port
uvicorn main_fastapi:app --reload --port 8001
```

## 📚 Full Documentation

See [PLANT_DISEASE_DETECTION_GUIDE.md](PLANT_DISEASE_DETECTION_GUIDE.md) for complete documentation.
