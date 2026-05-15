# API Testing Guide - Fruit Disease Detection

## Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. Get Supported Fruits

**Endpoint**: `GET /api/fruit-disease-detection/supported-fruits`

**Description**: Get list of all supported fruits

**cURL Example**:
```bash
curl -X GET "http://localhost:8000/api/fruit-disease-detection/supported-fruits"
```

**JavaScript/Fetch Example**:
```javascript
const response = await fetch('http://localhost:8000/api/fruit-disease-detection/supported-fruits');
const data = await response.json();
console.log(data);
```

**Response (Success)**:
```json
{
  "success": true,
  "data": {
    "fruits": ["Apple", "Mango", "Pomegranate", "Guava"],
    "supported_fruits_details": {
      "Apple": ["Blotch", "Healthy", "Rot", "Scab"],
      "Mango": ["Alternaria", "Anthracnose", "Black Mould Rot (Aspergillus)", "Healthy", "Stem and Rot (Lasiodiplodia)"],
      "Pomegranate": ["Alternaria", "Anthracnose", "Bacterial Blight", "Cercospora", "Healthy"],
      "Guava": ["Anthracnose", "Fruitfly", "Healthy"]
    },
    "total_fruits": 4
  }
}
```

---

### 2. Predict Disease with Fruit Selection

**Endpoint**: `POST /api/fruit-disease-detection/predict-with-selection`

**Description**: Predict fruit disease with fruit type validation

**Parameters**:
- `fruit_type` (string, required): Selected fruit type - "Apple", "Mango", "Pomegranate", or "Guava"
- `file` (file, required): Image file (JPEG, PNG) - max 10MB
- `confidence_threshold` (float, optional): Minimum confidence threshold - default 0.50
- `debug` (boolean, optional): Enable debug logging - default false

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/fruit-disease-detection/predict-with-selection" \
  -F "fruit_type=Mango" \
  -F "file=@path/to/mango.jpg" \
  -F "confidence_threshold=0.50" \
  -F "debug=false"
```

**Python Example (Requests)**:
```python
import requests

url = "http://localhost:8000/api/fruit-disease-detection/predict-with-selection"

files = {'file': open('mango.jpg', 'rb')}
data = {
    'fruit_type': 'Mango',
    'confidence_threshold': 0.50,
    'debug': False
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**JavaScript Example (Fetch)**:
```javascript
const formData = new FormData();
formData.append('fruit_type', 'Mango');
formData.append('file', fileInput.files[0]);
formData.append('confidence_threshold', 0.50);
formData.append('debug', false);

const response = await fetch('http://localhost:8000/api/fruit-disease-detection/predict-with-selection', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data);
```

**Response (Success)**:
```json
{
  "success": true,
  "data": {
    "selected_fruit": "Mango",
    "prediction": "Anthracnose_Mango",
    "confidence": 0.9234,
    "disease_info": {
      "fruit": "Mango",
      "disease": "Anthracnose",
      "severity": "High",
      "treatment": "Apply copper-based fungicide every 10-14 days during monsoon"
    },
    "interpretation": "High confidence detection of Anthracnose",
    "warnings": [],
    "has_warnings": false,
    "action_required": "FOLLOW_TREATMENT",
    "top_3": [
      {
        "class": "Anthracnose_Mango",
        "confidence": 0.9234
      },
      {
        "class": "Healthy_Mango",
        "confidence": 0.0512
      },
      {
        "class": "Stem and Rot (Lasiodiplodia)_Mango",
        "confidence": 0.0254
      }
    ]
  },
  "filename": "mango.jpg"
}
```

**Response (Error - Unsupported Fruit)**:
```json
{
  "success": false,
  "error": "This fruit is currently not supported or not available in the trained model. Supported fruits: Apple, Mango, Pomegranate, Guava",
  "data": {
    "selected_fruit": "Orange",
    "supported_fruits": ["Apple", "Mango", "Pomegranate", "Guava"]
  }
}
```

**Response (Error - No Fruit Selected)**:
```json
{
  "success": false,
  "error": "Please select a fruit type from the dropdown",
  "data": null
}
```

**Response (Error - No Image Uploaded)**:
```json
{
  "success": false,
  "error": "Invalid file type. Please upload an image (JPEG, PNG, etc.)",
  "data": null
}
```

**Response (Error - Invalid Image)**:
```json
{
  "success": false,
  "error": "Unable to detect disease. Please upload a valid fruit image.",
  "data": null
}
```

**Response (Error - Fruit Mismatch)**:
```json
{
  "success": false,
  "error": "Unable to detect disease. Please upload a valid fruit image.",
  "data": {
    "selected_fruit": "Mango",
    "detected_fruit": "Apple",
    "confidence": 0.45
  }
}
```

**Response (Error - File Too Large)**:
```json
{
  "success": false,
  "error": "File too large. Maximum size is 10MB",
  "data": null
}
```

---

### 3. Health Check

**Endpoint**: `GET /api/fruit-disease-detection/health`

**Description**: Check service health and availability

**cURL Example**:
```bash
curl -X GET "http://localhost:8000/api/fruit-disease-detection/health"
```

**JavaScript Example**:
```javascript
const response = await fetch('http://localhost:8000/api/fruit-disease-detection/health');
const data = await response.json();
console.log(data);
```

**Response (Healthy)**:
```json
{
  "status": "healthy",
  "service": "Fruit Disease Detection (Selection)",
  "detector_ready": true,
  "supported_fruits": ["Apple", "Mango", "Pomegranate", "Guava"],
  "total_fruits": 4
}
```

**Response (Unhealthy)**:
```json
{
  "status": "unhealthy",
  "service": "Fruit Disease Detection (Selection)",
  "error": "Detector not initialized",
  "supported_fruits": ["Apple", "Mango", "Pomegranate", "Guava"]
}
```

---

## Error Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Disease prediction successful |
| 400 | Bad Request | Missing fruit_type, invalid file, unsupported fruit |
| 503 | Service Unavailable | Detector not initialized |
| 500 | Server Error | Unexpected error during processing |

---

## Testing Checklist

### Basic Functionality
- [ ] Test GET `/api/fruit-disease-detection/supported-fruits`
- [ ] Test POST `/api/fruit-disease-detection/predict-with-selection` with valid inputs
- [ ] Test GET `/api/fruit-disease-detection/health`

### Error Cases
- [ ] Missing fruit_type parameter → Should return "Please select a fruit type"
- [ ] Missing file parameter → Should return "Invalid file type"
- [ ] Unsupported fruit (e.g., "Orange") → Should return "not supported" error
- [ ] Invalid image file → Should return "Unable to detect disease"
- [ ] File > 10MB → Should return "File too large"
- [ ] Non-image file → Should return "Invalid file type"
- [ ] Predict wrong fruit → Should return "Unable to detect disease"

### Edge Cases
- [ ] Test with low confidence threshold (0.1)
- [ ] Test with high confidence threshold (0.9)
- [ ] Test with very small images
- [ ] Test with very large images (but < 10MB)
- [ ] Test with different image formats (JPG, PNG)

---

## Quick Test Script (Python)

```python
import requests
import os

# Configuration
BASE_URL = "http://localhost:8000/api/fruit-disease-detection"
FRUIT_TYPE = "Mango"
IMAGE_PATH = "path/to/mango.jpg"

def test_supported_fruits():
    """Test: Get supported fruits"""
    print("Testing: Get supported fruits...")
    response = requests.get(f"{BASE_URL}/supported-fruits")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict(fruit_type, image_path):
    """Test: Predict disease"""
    print(f"Testing: Predict disease for {fruit_type}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}\n")
        return
    
    files = {'file': open(image_path, 'rb')}
    data = {
        'fruit_type': fruit_type,
        'confidence_threshold': 0.50
    }
    
    response = requests.post(f"{BASE_URL}/predict-with-selection", files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_health():
    """Test: Health check"""
    print("Testing: Health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    test_health()
    test_supported_fruits()
    test_predict(FRUIT_TYPE, IMAGE_PATH)
```

---

## Postman Collection

Import this into Postman:

```json
{
  "info": {
    "name": "Fruit Disease Detection API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Get Supported Fruits",
      "request": {
        "method": "GET",
        "url": {
          "raw": "http://localhost:8000/api/fruit-disease-detection/supported-fruits",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8000",
          "path": ["api", "fruit-disease-detection", "supported-fruits"]
        }
      }
    },
    {
      "name": "Predict Disease",
      "request": {
        "method": "POST",
        "url": {
          "raw": "http://localhost:8000/api/fruit-disease-detection/predict-with-selection",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8000",
          "path": ["api", "fruit-disease-detection", "predict-with-selection"]
        },
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "fruit_type",
              "value": "Mango",
              "type": "text"
            },
            {
              "key": "file",
              "type": "file",
              "src": "path/to/image.jpg"
            },
            {
              "key": "confidence_threshold",
              "value": "0.50",
              "type": "text"
            }
          ]
        }
      }
    },
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": {
          "raw": "http://localhost:8000/api/fruit-disease-detection/health",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8000",
          "path": ["api", "fruit-disease-detection", "health"]
        }
      }
    }
  ]
}
```

---

## Debugging Tips

1. **Enable Debug Mode**:
   ```bash
   curl -X POST "http://localhost:8000/api/fruit-disease-detection/predict-with-selection" \
     -F "fruit_type=Mango" \
     -F "file=@mango.jpg" \
     -F "debug=true"
   ```

2. **Check Backend Logs**: Look for detailed error messages in backend console

3. **Verify Image Format**: Ensure image is valid JPEG or PNG
   ```bash
   file mango.jpg  # Should show: JPEG image data
   ```

4. **Check File Size**:
   ```bash
   ls -lh mango.jpg  # Should be less than 10MB
   ```

5. **Test Health First**: Always check health before predicting
   ```bash
   curl http://localhost:8000/api/fruit-disease-detection/health
   ```

---

**Last Updated**: 2026-01-25
**API Version**: 1.0
