# SmartAgri-AI Minimal GitHub Package

This repo should contain only the files needed to clone, install, and run the current FastAPI + React app.

## 1) Keep in GitHub

### Root files
- `.gitignore`
- `.env.example`
- `docker-compose.yml`
- `readme.md`

### Backend runtime files
- `backend/main_fastapi.py`
- `backend/auth.py`
- `backend/database.py`
- `backend/weather_location.py`
- `backend/chatbot_service.py`
- `backend/crop_service.py`
- `backend/crop_models.py`
- `backend/api_crop_prediction.py`
- `backend/api_fertilizer.py`
- `backend/fertilizer_prediction_service.py`
- `backend/api_stress.py`
- `backend/stress_prediction_service.py`
- `backend/api_fruit_disease_production.py`
- `backend/fruit_disease_api_v2.py`
- `backend/fruit_disease_service.py`
- `backend/plant_disease_service.py`
- `backend/yield_prediction_service.py`
- `backend/model_manager.py`
- `backend/logging_config.py`
- `backend/requirements.txt`
- `backend/.env.example`

### Backend runtime data / model artifacts
Keep only the files the code actually loads at runtime:
- `backend/model/crop_model.pkl`
- `backend/model/fertilizer_model.pkl`
- `backend/model/fertilizer_encoders.pkl`
- `backend/model/fertilizer_label_encoder.pkl`
- `backend/model/fertilizer_feature_info.json`
- `backend/model/fertilizer_model_metrics.json`
- `backend/model/yield_prediction_model.pkl`
- `backend/model/yield_encoders.pkl`
- `backend/model/yield_feature_info.json`
- `backend/model/yield_model_metrics.json`
- `backend/model/stress_prediction_model.pkl`
- `backend/model/stress_label_encoders.pkl`
- `backend/model/stress_features.pkl`
- `backend/model/fruit_disease_model.h5`
- `backend/model/fruit_disease_labels.json`
- `backend/model/plant_disease_prediction_model.h5`

### Frontend runtime files
- `frontend/index.html`
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/vite.config.js`
- `frontend/tailwind.config.js`
- `frontend/postcss.config.js`
- `frontend/eslint.config.js`
- `frontend/.env.example`
- `frontend/src/**`
- `frontend/public/auth-test.html` if you want the auth test page

## 2) Delete before pushing

These should stay out of the repo to keep it small:
- `.env`
- `backend/.env`
- `frontend/.env`
- `frontend/.env.local`
- `frontend/.env.production`
- `node_modules/`
- `frontend/node_modules/`
- `dist/`
- `frontend/dist/`
- `build/`
- `venv/`, `.venv/`, `env/`, `ENV/`
- `__pycache__/`, `backend/__pycache__/`, `frontend/__pycache__/`
- `*.pyc`, `*.pyo`, `*.pyd`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.cache/`
- `backend/data/archive/`
- `backend/data/*.csv`
- `backend/data/*.tsv`
- `backend/data/*.xlsx`
- `backend/evaluation_graphs/`
- `backend/model/*history*`
- `backend/model/*report*`
- `backend/model/*visualization*`
- `backend/model/*.png`
- `backend/model/*.log`
- `backend/model/*.tmp`
- `backend/model/*.temp`
- training-only scripts, backups, and one-off utilities that are not imported by `main_fastapi.py`

## 3) Final optimized folder structure

```text
SmartAgri-AI/
├── backend/
│   ├── main_fastapi.py
│   ├── auth.py
│   ├── database.py
│   ├── chatbot_service.py
│   ├── crop_service.py
│   ├── crop_models.py
│   ├── api_crop_prediction.py
│   ├── api_fertilizer.py
│   ├── fertilizer_prediction_service.py
│   ├── api_stress.py
│   ├── stress_prediction_service.py
│   ├── api_fruit_disease_production.py
│   ├── fruit_disease_api_v2.py
│   ├── fruit_disease_service.py
│   ├── plant_disease_service.py
│   ├── yield_prediction_service.py
│   ├── weather_location.py
│   ├── model_manager.py
│   ├── logging_config.py
│   ├── requirements.txt
│   ├── .env.example
│   └── model/
│       ├── crop_model.pkl
│       ├── fertilizer_model.pkl
│       ├── fertilizer_encoders.pkl
│       ├── fertilizer_label_encoder.pkl
│       ├── fertilizer_feature_info.json
│       ├── fertilizer_model_metrics.json
│       ├── yield_prediction_model.pkl
│       ├── yield_encoders.pkl
│       ├── yield_feature_info.json
│       ├── yield_model_metrics.json
│       ├── stress_prediction_model.pkl
│       ├── stress_label_encoders.pkl
│       ├── stress_features.pkl
│       ├── fruit_disease_model.h5
│       ├── fruit_disease_labels.json
│       └── plant_disease_prediction_model.h5
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── eslint.config.js
│   ├── .env.example
│   ├── public/
│   └── src/
└── docker-compose.yml
```

## 4) Environment variables to store in `.env`

### Backend `.env`
- `MONGODB_URL`
- `DATABASE_NAME`
- `USERS_DATABASE`
- `CHATBOT_DATABASE`
- `LEGACY_DATABASE`
- `GROQ_API_KEY`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_EXPIRATION_MINUTES`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `OPENWEATHER_API_KEY`
- `NEWSAPI_KEY`
- `PORT`
- `HOST`
- `CORS_ORIGINS`
- `ENVIRONMENT`
- `LOW_MEMORY_MODE`

### Frontend `.env`
- `VITE_API_BASE_URL` or `VITE_BACKEND_URL`
- `VITE_GOOGLE_CLIENT_ID`

## 5) Clone and run after pushing

```powershell
git clone https://github.com/Purushothamcv/AgricultureFarmTechnology.git
cd AgricultureFarmTechnology

copy backend\.env.example backend\.env
copy frontend\.env.example frontend\.env

cd backend
pip install -r requirements.txt
python -m uvicorn main_fastapi:app --host 127.0.0.1 --port 8000

cd ..\frontend
npm install
npm run dev
```

If you run from the repo root, use:

```powershell
pip install -r backend\requirements.txt
cd frontend
npm install
```

## 6) Notes

- Keep trained ML model files only if the corresponding endpoint is used at runtime.
- Do not commit `.env`, `node_modules`, `venv`, cache folders, or generated build output.
- If you want the repo even smaller, remove training scripts and legacy backups that are not imported by `main_fastapi.py`.

5. **Run the FastAPI server**
   ```bash
   uvicorn main_fastapi:app --reload
   ```
   Backend will run at: `http://127.0.0.1:8000`

6. **Access API Documentation**
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   Frontend will run at: `http://localhost:3000`

4. **Build for production**
   ```bash
   npm run build
   ```

---

## 🔌 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login

### Weather & Location
- `GET /api/weather?lat={lat}&lon={lon}` - Get weather data
- `GET /api/location-data?lat={lat}&lon={lon}` - Get location-based data

### Crop Management
- `POST /api/crop/recommend` - Get crop recommendations
- `POST /predict/manual` - Manual crop prediction (8 parameters)
- `POST /predict/location` - Location-based crop prediction

### Yield & Fertilizer
- `POST /api/yield/predict` - Predict crop yield
- `POST /api/fertilizer/recommend` - Fertilizer recommendations

### Disease Detection
- `POST /api/disease/leaf` - Leaf disease detection
- `POST /api/disease/fruit` - Fruit disease detection
- `POST /api/potato/predict` - Potato disease prediction (LSTM)

### Plant Management
- `POST /api/stress/predict` - Plant stress prediction
- `POST /api/spray/recommend` - Best spray time recommendation

### Chatbot
- `POST /api/chat` - Chat with AI assistant

---

## 💡 Usage Guide

### 1. **Getting Started**
   - Register a new account or login
   - Navigate to Dashboard
   - Select your location (search, click map, or use GPS)

### 2. **Get Weather Data**
   - Click "Use My Location" for instant GPS positioning
   - Or search for your city in the search bar
   - Click "Get Weather & Recommendations"
   - Weather data is stored for auto-fill in other modules

### 3. **Crop Recommendation**
   - Choose between Manual or Map-based mode
   - **Manual**: Enter soil nutrients and environmental data
   - **Map**: Select location and let the system auto-fetch data
   - Get instant crop recommendations

### 4. **Disease Detection**
   - Upload clear image of plant leaf or fruit
   - System analyzes image using CNN model
   - Receive disease diagnosis and treatment recommendations

### 5. **Best Spray Time**
   - Weather data auto-fills from Dashboard
   - Or click "Use Current Weather" for live data
   - Or manually enter forecast data
   - Get safe spray time recommendations with risk analysis

### 6. **Yield Prediction**
   - Select crop type and enter area
   - Weather auto-fills or enter manually
   - Get estimated yield based on conditions

---

## 🎨 Features in Detail

### Machine Learning Models

#### **Crop Recommendation Model**
- **Algorithm**: Random Forest Classifier
- **Features (8)**:
  1. Nitrogen (N) - 0-140 kg/ha
  2. Phosphorus (P) - 5-145 kg/ha
  3. Potassium (K) - 5-205 kg/ha
  4. Temperature - 0-50°C
  5. Humidity - 0-100%
  6. pH - 3.5-9.5
  7. Rainfall - 0-300mm
  8. Ozone - 0-100 ppb
- **Training Data**: 2,200+ samples across 22 crop types
- **Accuracy**: ~95% on test data

#### **Regional Soil Profiles**
Pre-configured for Indian regions:
- North India: High ozone (35 ppb), moderate nutrients
- South India: Balanced nutrients, moderate ozone (30 ppb)
- East India: Lower ozone (28 ppb), rich soil
- West India: High potassium, moderate ozone (32 ppb)
- Central India: Balanced profile (30 ppb ozone)

### Weather Integration

**Open-Meteo API Features**:
- Temperature (current & forecast)
- Relative humidity
- Precipitation/rainfall
- Wind speed & direction
- Atmospheric pressure
- No API key required
- Global coverage
- Hourly & daily forecasts

### Database Schema

**MongoDB Collections**:

1. **users**
   ```javascript
   {
     _id: ObjectId,
     username: String,
     email: String (unique),
     password: String (bcrypt hashed),
     created_at: DateTime
   }
   ```

2. **plant_disease_predictions**
   ```javascript
   {
     _id: ObjectId,
     user_id: String,
     image_path: String,
     disease_type: String,
     confidence: Number,
     timestamp: DateTime
   }
   ```

3. **weather_logs**
   ```javascript
   {
     _id: ObjectId,
     lat: Number,
     lon: Number,
     temperature: Number,
     humidity: Number,
     rainfall: Number,
     wind_speed: Number,
     fetched_at: DateTime
   }
   ```

---

## 🔒 Security Features

- **Password Hashing**: Bcrypt with 12 rounds
- **Protected Routes**: Authentication required for sensitive operations
- **CORS Configuration**: Restricted origins (localhost:3000, 3001, 3002)
- **Input Validation**: Pydantic models for request validation
- **SQL Injection Prevention**: MongoDB query parameterization
- **Environment Variables**: Sensitive data in .env files (not committed)

---

## 🌐 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Your Name** - *Initial work*

---

## 🙏 Acknowledgments

- Open-Meteo API for free weather data
- OpenStreetMap & Nominatim for mapping services
- Leaflet.js community for excellent mapping library
- Scikit-learn team for ML tools
- FastAPI & React communities

---

## 📧 Contact

For questions or support, please open an issue or contact:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🔮 Future Enhancements

- [ ] Real-time satellite imagery integration
- [ ] Soil testing kit integration via IoT
- [ ] Multi-language support (Hindi, Tamil, Telugu, etc.)
- [ ] Mobile application (React Native)
- [ ] Advanced analytics dashboard with charts
- [ ] Community forum for farmers
- [ ] Market price prediction
- [ ] Crop insurance recommendations
- [ ] SMS/WhatsApp notifications
- [ ] Drone imagery analysis
- [ ] Blockchain for supply chain tracking

---

**⭐ If you find this project helpful, please give it a star!**

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- Open-Meteo Weather API
- Streamlit-Folium (interactive maps)

## 📁 Project Structure
```
smart_agriculture_ozone/
├── app.py
├── utils.py
├── data/
│   ├── sample_data.py
│   └── real_data_explained.md
├── model/
│   └── model_training.py
├── requirements.txt
└── README.md
```

## 🚀 Run the App
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python model/model_training.py
```

3. Launch the Streamlit app:
```bash
streamlit run app.py
```

## 🧠 Model
- `LinearRegression` is trained on synthetic data generated using domain-based relationships.
- Easily replaceable with real-world datasets.

## 📌 Data Sources
See `data/real_data_explained.md` for guidance on integrating real datasets.

## 📬 Contact
For help or improvements, reach out or open a GitHub issue.

---
Developed with ❤️ to support smart agriculture & sustainable farming.
"# smart_agri" 
