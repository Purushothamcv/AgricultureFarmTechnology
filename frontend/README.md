# Smart Agri Frontend

A modern, professional React-based frontend for the Smart Agriculture Decision Support System.

## 🌟 Features

- **Authentication System**: Secure login and registration
- **Dashboard**: Interactive map with weather integration
- **Crop Recommendation**: AI-powered crop suggestions based on soil and climate
- **Yield Prediction**: Predict crop yields based on multiple factors
- **Fertilizer Recommendation**: Optimize fertilizer usage
- **Stress Prediction**: Monitor and predict crop stress levels
- **Best Spray Time**: Weather-based spraying recommendations
- **Disease Detection**: 
  - Fruit disease classification
  - Plant leaf disease detection
- **AI Chatbot**: Agriculture assistant for farming queries

## 🛠️ Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **React Router v6** - Routing
- **Tailwind CSS** - Styling
- **Axios** - API calls
- **Leaflet/React-Leaflet** - Interactive maps
- **Lucide React** - Icons
- **Context API** - State management

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Navbar.jsx
│   │   ├── ProtectedRoute.jsx
│   │   ├── InputField.jsx
│   │   ├── WeatherCard.jsx
│   │   ├── ResultCard.jsx
│   │   ├── ImageUploader.jsx
│   │   ├── LoadingSpinner.jsx
│   │   └── ChatMessage.jsx
│   ├── pages/               # Application pages
│   │   ├── Login.jsx
│   │   ├── Register.jsx
│   │   ├── Dashboard.jsx
│   │   ├── CropRecommendation.jsx
│   │   ├── YieldPrediction.jsx
│   │   ├── FertilizerRecommendation.jsx
│   │   ├── StressPrediction.jsx
│   │   ├── BestSprayTime.jsx
│   │   ├── FruitDisease.jsx
│   │   ├── LeafDisease.jsx
│   │   └── Chatbot.jsx
│   ├── context/             # React Context
│   │   └── AuthContext.jsx
│   ├── services/            # API services
│   │   ├── api.js
│   │   └── services.js
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── public/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the frontend root:
```env
VITE_API_BASE_URL=http://localhost:8000/api
```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The build files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 🎨 Design System

### Colors

- **Primary Green**: Agriculture-themed green palette
  - `primary-50` to `primary-900`
- **Neutral Grays**: For text and backgrounds
- **Semantic Colors**: Success (green), Warning (yellow), Error (red), Info (blue)

### Components

All components follow a consistent design pattern:
- Card-based layouts
- Responsive grid systems
- Consistent spacing (Tailwind)
- Accessible form controls
- Loading states
- Error handling

## 🔐 Authentication

The app uses JWT-based authentication:
- Login/Register pages
- Protected routes
- Auth context for global state
- Token stored in localStorage
- Auto-redirect on 401 errors

**Demo Credentials:**
- Email: demo@smartagri.com
- Password: demo123

## 🗺️ Routes

| Route | Description | Protected |
|-------|-------------|-----------|
| `/login` | Login page | No |
| `/register` | Registration page | No |
| `/dashboard` | Main dashboard | Yes |
| `/crop-recommendation` | Crop recommendation | Yes |
| `/yield-prediction` | Yield prediction | Yes |
| `/fertilizer` | Fertilizer guide | Yes |
| `/stress-prediction` | Stress monitoring | Yes |
| `/spray-time` | Spray timing | Yes |
| `/fruit-disease` | Fruit disease detection | Yes |
| `/leaf-disease` | Leaf disease detection | Yes |
| `/chatbot` | AI assistant | Yes |

## 📡 API Integration

The frontend communicates with the backend through RESTful APIs:

### Base URL
Configure in `.env` file: `VITE_API_BASE_URL`

### API Services

All API calls are centralized in `src/services/`:
- `authService` - Authentication
- `weatherService` - Weather data
- `cropService` - Crop operations
- `fertilizerService` - Fertilizer recommendations
- `diseaseService` - Disease detection
- `chatbotService` - AI chatbot

### Request/Response Interceptors

- Auto-attaches JWT token to requests
- Handles 401 (unauthorized) responses
- Centralizes error handling

## 🎯 Features Overview

### 1. Dashboard
- Interactive map (Leaflet)
- Click to select location
- Auto-fill lat/long
- Weather card display
- Module navigation

### 2. Crop Recommendation
- Input: N, P, K, temperature, humidity, pH, rainfall, ozone
- Output: Recommended crop

### 3. Yield Prediction
- Input: Crop type, area, soil, weather
- Output: Predicted yield

### 4. Fertilizer Recommendation
- Input: Soil nutrients, crop, conditions
- Output: Fertilizer type and dosage

### 5. Stress Prediction
- Input: Environmental parameters
- Output: Stress level (Low/Medium/High)

### 6. Best Spray Time
- Input: Weather conditions
- Output: Safe/unsafe spraying window

### 7. Disease Detection
- Image upload
- AI-powered classification
- Disease identification with confidence

### 8. AI Chatbot
- Chat interface
- Agriculture-focused assistant
- Quick question prompts

## 📱 Responsive Design

The application is fully responsive:
- Desktop: Full layout with sidebar navigation
- Tablet: Adapted grid layouts
- Mobile: Hamburger menu, stacked layouts

## 🧪 Demo Mode

When backend is unavailable, the app provides demo/fallback data for testing:
- Weather: Sample data
- Predictions: Random realistic results
- Disease detection: Sample classifications
- Chatbot: Predefined responses

## 🔧 Customization

### Change Theme Colors

Edit `tailwind.config.js`:
```js
theme: {
  extend: {
    colors: {
      primary: {
        // Your color palette
      }
    }
  }
}
```

### Add New Module

1. Create page component in `src/pages/`
2. Add route in `App.jsx`
3. Add navigation link in `Navbar.jsx`
4. Add service in `src/services/services.js`

## 🐛 Troubleshooting

### Map not loading
- Check Leaflet CSS is imported
- Verify internet connection (uses OpenStreetMap tiles)

### API errors
- Check backend is running
- Verify `VITE_API_BASE_URL` in `.env`
- Check browser console for details

### Build errors
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`

## 📄 License

This project is part of the Smart Agri system.

## 👥 Contributing

1. Follow the existing code style
2. Use meaningful component and variable names
3. Add comments for complex logic
4. Test responsive design
5. Ensure accessibility

## 📞 Support

For issues or questions:
- Check documentation
- Review code comments
- Contact development team

---

**Built with ❤️ for Smart Agriculture**
