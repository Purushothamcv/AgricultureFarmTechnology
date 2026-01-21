# 🌾 Smart Agri - Complete Frontend Application

> **Professional, Production-Ready React Frontend for Agricultural Decision Support System**

[![React](https://img.shields.io/badge/React-18.2.0-blue?logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.0.8-646CFF?logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.3.6-38B2AC?logo=tailwind-css)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Quick Start (30 seconds)

```powershell
# Navigate to frontend
cd frontend

# Install & Run
npm install
npm run dev

# Open http://localhost:3000
# Login: demo@smartagri.com / demo123
```

---

## ✨ What's Inside

### 🔐 Authentication
- ✅ Secure login & registration
- ✅ JWT-based authentication
- ✅ Protected routes
- ✅ Auto session management

### 🗺️ Dashboard
- ✅ Interactive Leaflet map
- ✅ Location-based weather
- ✅ Real-time data display
- ✅ Quick module access

### 🌾 Agricultural Modules (8)
1. **Crop Recommendation** - AI-powered crop selection
2. **Yield Prediction** - Accurate harvest forecasting
3. **Fertilizer Guide** - Optimal nutrient recommendations
4. **Stress Monitoring** - Plant health assessment
5. **Spray Timing** - Weather-based spraying schedule
6. **Fruit Disease** - Image-based disease detection
7. **Leaf Disease** - Plant disease identification
8. **AI Chatbot** - Intelligent agriculture assistant

---

## 📸 Screenshots

### Dashboard
```
┌──────────────────────────────────────────────────┐
│  Smart Agri        Dashboard  Modules  Logout    │
├──────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │                 │  │  Current Weather      │  │
│  │   Interactive   │  │  🌡️ 28°C  💧 65%    │  │
│  │      Map        │  │  🌧️ 2mm   💨 12km/h │  │
│  │                 │  │                      │  │
│  └─────────────────┘  └──────────────────────┘  │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │Crop │ │Yield│ │Fert │ │Chat │              │
│  └─────┘ └─────┘ └─────┘ └─────┘              │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Framework** | React 18.2 |
| **Build Tool** | Vite 5.0 |
| **Routing** | React Router v6 |
| **Styling** | Tailwind CSS 3.3 |
| **State** | Context API |
| **HTTP Client** | Axios |
| **Maps** | Leaflet + React-Leaflet |
| **Icons** | Lucide React |
| **Dev Server** | Vite Dev Server |

---

## 📦 Installation

### Prerequisites
```
Node.js >= 16.0.0
npm >= 7.0.0
```

### Steps

1. **Clone/Navigate to project**
```powershell
cd "c:\Users\purus\OneDrive\New folder\Desktop\ml projects\SmartAgri-AI\frontend"
```

2. **Install dependencies**
```powershell
npm install
```

3. **Configure environment**
```powershell
# Copy example env file
Copy-Item .env.example .env

# Edit .env if needed
# VITE_API_BASE_URL=http://localhost:8000/api
```

4. **Start development server**
```powershell
npm run dev
```

5. **Open browser**
```
http://localhost:3000
```

---

## 🚀 Available Commands

```powershell
# Development
npm run dev          # Start dev server (localhost:3000)

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint

# Quick Start
.\start.ps1         # Automated setup & start
```

---

## 📁 Project Structure

```
frontend/
│
├── src/
│   ├── components/          # ⚛️ Reusable Components (8)
│   │   ├── ChatMessage.jsx
│   │   ├── ImageUploader.jsx
│   │   ├── InputField.jsx
│   │   ├── LoadingSpinner.jsx
│   │   ├── Navbar.jsx
│   │   ├── ProtectedRoute.jsx
│   │   ├── ResultCard.jsx
│   │   └── WeatherCard.jsx
│   │
│   ├── pages/              # 📄 Application Pages (10)
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
│   │
│   ├── context/            # 🔄 State Management
│   │   └── AuthContext.jsx
│   │
│   ├── services/           # 🌐 API Layer
│   │   ├── api.js
│   │   └── services.js
│   │
│   ├── App.jsx             # Main application
│   ├── main.jsx            # Entry point
│   └── index.css           # Global styles
│
├── public/                 # Static assets
├── .env.example           # Environment template
├── package.json           # Dependencies
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind theme
├── README.md             # This file
├── SETUP_GUIDE.md        # Detailed setup
└── PROJECT_SUMMARY.md    # Complete summary
```

---

## 🎨 Design System

### Color Theme
```css
Primary: #22c55e  /* Agriculture Green */
Success: #10b981  /* Green */
Warning: #f59e0b  /* Amber */
Error:   #ef4444  /* Red */
Info:    #3b82f6  /* Blue */
```

### Components
- 📦 **Cards** - Rounded, shadowed containers
- 🔘 **Buttons** - Primary (green) & Secondary (gray)
- 📝 **Inputs** - Bordered with focus states
- 🎨 **Icons** - Consistent 24px/20px Lucide icons

---

## 🔌 API Integration

### Environment Variable
```env
VITE_API_BASE_URL=http://localhost:8000/api
```

### Expected Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/auth/login` | User login |
| POST | `/auth/register` | User registration |
| GET | `/weather?lat=X&lon=Y` | Weather data |
| POST | `/crop/recommend` | Crop recommendation |
| POST | `/crop/predict-yield` | Yield prediction |
| POST | `/fertilizer/recommend` | Fertilizer advice |
| POST | `/crop/predict-stress` | Stress analysis |
| POST | `/crop/best-spray-time` | Spray timing |
| POST | `/disease/fruit` | Fruit disease detection |
| POST | `/disease/leaf` | Leaf disease detection |
| POST | `/chatbot/message` | Chat responses |

### Demo Mode
✅ App works without backend (uses fallback data)

---

## 📱 Features

### ✅ Authentication
- Secure login/register
- JWT token management
- Protected routes
- Auto-logout on token expiry

### ✅ Dashboard
- Interactive map (click to select location)
- Real-time weather display
- Quick navigation to modules

### ✅ Crop Recommendation
- Input: N, P, K, temp, humidity, pH, rainfall, ozone
- Output: Recommended crop

### ✅ Yield Prediction
- Input: Crop, area, soil, weather
- Output: Predicted yield with confidence

### ✅ Fertilizer Guide
- Input: Soil nutrients, crop type
- Output: Fertilizer recommendation

### ✅ Stress Monitoring
- Input: Environmental parameters
- Output: Stress level (Low/Medium/High)

### ✅ Spray Timing
- Input: Weather conditions
- Output: Safe/unsafe assessment

### ✅ Disease Detection
- Upload plant/fruit images
- AI-powered classification
- Treatment recommendations

### ✅ AI Chatbot
- WhatsApp-style interface
- Agriculture-focused responses
- Quick question prompts

---

## 📲 Responsive Design

| Device | Layout |
|--------|--------|
| 📱 **Mobile** (< 640px) | Stacked, hamburger menu |
| 📱 **Tablet** (640-1024px) | 2-column grids |
| 💻 **Desktop** (> 1024px) | Full navigation, multi-column |

---

## 🧪 Testing

### Demo Credentials
```
Email: demo@smartagri.com
Password: demo123
```

### Test Checklist
- [ ] Login/Register flow
- [ ] Map interaction
- [ ] Weather fetch
- [ ] All 8 modules work
- [ ] Image upload
- [ ] Chatbot messaging
- [ ] Responsive on mobile
- [ ] Error handling

---

## 🏗️ Building for Production

### Build
```powershell
npm run build
```

### Output
- Directory: `dist/`
- Optimized & minified
- Ready for deployment

### Deploy to:
- **Netlify** - Drop `dist/` folder
- **Vercel** - Connect GitHub repo
- **AWS S3** - Upload `dist/` + CloudFront
- **Nginx** - Serve `dist/` folder

---

## 🔧 Customization

### Change Theme Color
Edit `tailwind.config.js`:
```js
primary: {
  600: '#YOUR_COLOR'  // Main brand color
}
```

### Add New Page
1. Create component in `src/pages/YourPage.jsx`
2. Add route in `App.jsx`
3. Add navigation in `Navbar.jsx`

### Change API URL
Edit `.env`:
```env
VITE_API_BASE_URL=https://your-api.com/api
```

---

## 🐛 Troubleshooting

### "npm not found"
**Install Node.js:** https://nodejs.org/

### Port 3000 in use
**Change port** in `vite.config.js`:
```js
server: { port: 3001 }
```

### Map not loading
**Check:**
- Internet connection
- Leaflet CSS in `index.html`
- Browser console errors

### API errors
**Verify:**
- Backend is running
- `.env` file exists
- API URL is correct
- CORS is enabled on backend

---

## 📚 Documentation

- 📖 **README.md** - This file (overview)
- 📋 **SETUP_GUIDE.md** - Detailed setup instructions
- 📊 **PROJECT_SUMMARY.md** - Complete project summary

---

## ✅ Features Checklist

- ✅ Authentication (Login/Register)
- ✅ Protected Routes
- ✅ Dashboard with Map
- ✅ Weather Integration
- ✅ 8 Agricultural Modules
- ✅ Image Upload & Preview
- ✅ AI Chatbot Interface
- ✅ Responsive Design
- ✅ Loading States
- ✅ Error Handling
- ✅ Demo Mode
- ✅ Production Ready

---

## 🎓 Code Quality

### ✅ Best Practices
- Component-based architecture
- Reusable components
- Separation of concerns
- DRY principle
- Consistent naming
- Proper error handling
- Environment variables
- API abstraction

### ✅ Performance
- Vite for fast builds
- Lazy loading ready
- Optimized assets
- Minimal re-renders

---

## 🚀 Next Steps

1. ✅ **Run the app** - Test all features
2. ✅ **Connect backend** - Update API endpoints
3. ✅ **Customize** - Adjust theme/content
4. ✅ **Test** - Verify all modules
5. ✅ **Deploy** - Go to production!

---

## 📄 License

MIT License - Feel free to use for your projects!

---

## 🙏 Acknowledgments

- **React Team** - Amazing framework
- **Tailwind CSS** - Beautiful styling
- **Leaflet** - Interactive maps
- **Lucide** - Beautiful icons

---

## 🎉 Summary

You have a **complete, production-ready React frontend** featuring:

- 🔐 Secure authentication system
- 🗺️ Interactive map dashboard
- 🌾 8 agriculture modules
- 📸 Image upload & processing
- 💬 AI chatbot interface
- 📱 Fully responsive design
- ⚡ Fast & optimized
- 📚 Complete documentation

**Ready to deploy and impress! 🚀**

---

<div align="center">

**Built with ❤️ for Smart Agriculture**

[Documentation](./SETUP_GUIDE.md) • [Features](./PROJECT_SUMMARY.md) • [Report Issues](https://github.com)

</div>
