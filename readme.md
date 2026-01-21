# 🌾 SmartAgri-AI: Intelligent Agricultural Management System

## 📋 Overview

A full-stack AI-powered agricultural platform built with FastAPI and React. Features ML-based crop recommendations, disease detection, yield prediction, and real-time weather integration with interactive maps. Helps farmers make data-driven decisions for optimal farming practices.

**Tech Stack**: Python | FastAPI | React | MongoDB | Scikit-learn | TensorFlow | Leaflet.js

---

## ✨ Key Features

### 🌍 **Interactive Location-Based Dashboard**
- **Interactive Leaflet.js Maps**: Click-to-select location on map or search by city name
- **Geolocation Support**: "Use My Location" button for instant GPS-based positioning
- **Real-time Weather Integration**: Automatic weather data fetching from Open-Meteo API
- **Smart Search**: Nominatim API integration for global location search
- **Weather Auto-fill**: Weather data automatically propagates across all modules

### 🌱 **Crop Recommendation System**
- **Dual Input Modes**:
  - **Manual Entry**: Input soil nutrients (N, P, K), pH, and environmental data
  - **Map-Based Selection**: Select location on map for automatic weather/soil data retrieval
- **8-Feature ML Model**: Random Forest Classifier trained on soil and climate parameters
- **Regional Soil Data**: Pre-configured soil profiles for different Indian regions
- **Instant Predictions**: Get crop recommendations optimized for your conditions
- **Parameters**: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall, Ozone

### 🥔 **Potato Disease Prediction (LSTM)**
- **Time-Series Analysis**: LSTM neural network for disease outbreak prediction
- **Historical Data Processing**: Analyzes patterns in environmental conditions
- **Disease Risk Assessment**: Early warning system for potential disease outbreaks
- **Field-Specific Predictions**: Multi-field support with individual tracking

### 🍃 **Leaf Disease Detection**
- **Image-Based Diagnosis**: Upload plant leaf images for disease identification
- **Deep Learning Classification**: CNN model for accurate disease detection
- **Instant Results**: Real-time analysis with disease type and confidence score
- **Treatment Recommendations**: Actionable advice for disease management

### 🍎 **Fruit Disease Detection**
- **Multi-Fruit Support**: Detects diseases across various fruit crops
- **Visual Analysis**: Image upload and processing for fruit disease identification
- **Disease Database**: Comprehensive collection of fruit disease signatures
- **Prevention Guidelines**: Suggestions for disease prevention and control

### 📊 **Yield Prediction**
- **Crop-Specific Models**: Predictive analytics for different crop types
- **Environmental Factors**: Considers temperature, rainfall, humidity, soil moisture
- **Area-Based Calculations**: Estimates total yield based on cultivated area
- **Ozone Impact Analysis**: Accounts for atmospheric ozone effects on crop growth

### 💧 **Fertilizer Recommendation**
- **Nutrient Analysis**: NPK (Nitrogen, Phosphorus, Potassium) requirement calculation
- **Soil-Based Recommendations**: Tailored fertilizer suggestions based on soil type
- **Crop-Specific Guidance**: Different recommendations for different crops
- **Quantity Estimation**: Precise fertilizer amount calculations

### ⚠️ **Plant Stress Prediction**
- **Multi-Factor Analysis**: Evaluates stress from temperature, water, nutrients
- **Stress Level Classification**: Categorizes stress as Low, Moderate, High, or Critical
- **Early Warning System**: Alerts farmers to potential crop stress conditions
- **Preventive Measures**: Recommendations to mitigate identified stress factors

### 🌤️ **Best Spray Time Calculator**
- **Weather-Based Analysis**: Determines optimal pesticide/herbicide application timing
- **Three Input Methods**:
  1. Use Current Weather (GPS-based)
  2. Auto-fill from Dashboard weather data
  3. Manual weather forecast entry
- **Safety Assessment**: Evaluates wind speed, temperature, humidity, rainfall
- **Time Slot Recommendations**: Suggests best times (early morning, evening, etc.)
- **Risk Factors Display**: Highlights favorable and unfavorable conditions

### 🤖 **AI Agricultural Chatbot**
- **Natural Language Processing**: Conversational interface for agricultural queries
- **Context-Aware Responses**: Provides relevant information about modules and farming
- **Module Integration**: Explains features and guides users through the platform
- **24/7 Assistance**: Always available for farming-related questions

### 🔐 **User Authentication System**
- **MongoDB Integration**: Secure user data storage in "FinalProject" database
- **Bcrypt Password Hashing**: Industry-standard password encryption (12 rounds)
- **Session Management**: Persistent login with localStorage
- **Protected Routes**: Role-based access control for authenticated users
- **Registration & Login**: Complete authentication workflow

---

## 🛠️ Technology Stack

### **Backend**
- **FastAPI**: Modern, fast Python web framework with automatic API documentation
- **Python 3.10**: Core programming language
- **MongoDB**: NoSQL database for user management and logging (Motor async driver)
- **Scikit-learn 1.7.2**: Machine Learning models (Random Forest, Decision Trees)
- **TensorFlow/Keras**: Deep learning for LSTM and CNN models
- **Joblib**: Model serialization and loading
- **Passlib**: Password hashing with bcrypt
- **Requests**: HTTP client for external API calls
- **Uvicorn**: ASGI server with hot-reload support

### **Frontend**
- **React 18**: Modern UI library with hooks
- **React Router v6**: Client-side routing with protected routes
- **Vite**: Lightning-fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Lucide React**: Beautiful icon library
- **Axios**: Promise-based HTTP client
- **Leaflet.js 1.9.4**: Interactive map library
- **React-Leaflet**: React components for Leaflet integration

### **External APIs**
- **Open-Meteo API**: Free weather data (no API key required)
  - Temperature, humidity, precipitation
  - Wind speed and atmospheric conditions
- **Nominatim API (OpenStreetMap)**: Geocoding and location search
  - Forward geocoding (city → coordinates)
  - Reverse geocoding (coordinates → address)

### **Machine Learning Models**
- **Crop Recommendation**: Random Forest Classifier (8 features)
- **Disease Detection**: Convolutional Neural Networks (CNN)
- **Yield Prediction**: Regression models
- **Potato Disease LSTM**: Time-series prediction with LSTM networks
- **Stress Prediction**: Multi-factor classification models

---

## 📁 Project Structure

```
SmartAgri-AI/
│
├── frontend/                      # React Frontend Application
│   ├── src/
│   │   ├── components/           # Reusable UI Components
│   │   │   ├── Navbar.jsx
│   │   │   ├── WeatherCard.jsx
│   │   │   ├── InputField.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── ChatMessage.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   └── ProtectedRoute.jsx
│   │   │
│   │   ├── pages/                # Page Components
│   │   │   ├── Dashboard.jsx     # Main dashboard with map
│   │   │   ├── Login.jsx
│   │   │   ├── Register.jsx
│   │   │   ├── CropRecommendation.jsx
│   │   │   ├── YieldPrediction.jsx
│   │   │   ├── FertilizerRecommendation.jsx
│   │   │   ├── StressPrediction.jsx
│   │   │   ├── BestSprayTime.jsx
│   │   │   ├── LeafDisease.jsx
│   │   │   ├── FruitDisease.jsx
│   │   │   └── Chatbot.jsx
│   │   │
│   │   ├── services/             # API Integration
│   │   │   ├── api.js            # Axios instance
│   │   │   └── services.js       # Service functions
│   │   │
│   │   ├── context/              # React Context
│   │   │   └── AuthContext.jsx   # Authentication state
│   │   │
│   │   ├── App.jsx               # Root component with routes
│   │   ├── main.jsx              # Entry point
│   │   └── index.css             # Global styles
│   │
│   ├── public/                   # Static assets
│   ├── package.json              # Dependencies
│   ├── vite.config.js            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── postcss.config.js         # PostCSS config
│
├── actual/                        # Potato Disease Module
│   ├── model/
│   │   ├── potato_disease_lstm_model.h5
│   │   └── LSTM_timeSeries.py
│   ├── data/
│   │   ├── combined_potato_disease_data.csv
│   │   └── fields.json
│   ├── app.py                    # Flask app for potato disease
│   ├── main.py
│   └── train.py
│
├── model/                         # ML Model Training Scripts
│   ├── model_training.py         # Crop recommendation model
│   ├── train_crop_model.py
│   ├── fertilizer_model_training.py
│   ├── best_time_model_training.py
│   └── predict_stress_level.py
│
├── data/                          # Training Datasets
│   ├── crop.csv                  # Crop training data
│   ├── fertilizer.csv            # Fertilizer data
│   └── sample_data.py
│
├── static/                        # Frontend Static Files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── app.js
│   │   ├── dashboard.js
│   │   └── auto_update.js
│   └── leaflet.js
│
├── templates/                     # HTML Templates
│   ├── index.html
│   ├── weather.html
│   └── crop_recommendation.html
│
├── main_fastapi.py               # Main FastAPI Application
├── crop_service.py               # Crop prediction service
├── crop_models.py                # Pydantic models
├── utils.py                      # Utility functions
├── weather_fastapi.py            # Weather service
├── requirements.txt              # Python dependencies
├── crop_model.pkl                # Trained ML model
└── README.md                     # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 16+** and **npm**
- **MongoDB** (local or MongoDB Atlas)
- **Git**

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SmartAgri-AI.git
   cd SmartAgri-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start MongoDB**
   ```bash
   # Local MongoDB
   mongod --dbpath /path/to/data/db
   
   # Or use MongoDB Atlas (cloud)
   # Update connection string in main_fastapi.py
   ```

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
