# 📍 Map Auto-Fill Feature Guide

## What Data is Auto-Filled from the Map?

When you select a location on the map, the system automatically fetches and fills the following fields:

### 🌤️ **Weather Data** (Blue Indicators)
These fields get real-time weather data from OpenWeatherMap API:

| Field | Source | Visual Indicator |
|-------|--------|------------------|
| **Temperature** (°C) | Current weather at location | 🌤️ Blue background |
| **Humidity** (%) | Current weather at location | 🌤️ Blue background |
| **Rainfall** (mm) | Recent rainfall data | 🌤️ Blue background |

### 🌱 **Soil Characteristics** (Green Indicators)
These fields get real soil data from ISRIC SoilGrids API:

| Field | Source | Visual Indicator |
|-------|--------|------------------|
| **Soil Type** | Texture classification from clay/sand/silt | 📍 Green background |
| **Soil pH** | pH measurement from SoilGrids | 📍 Green background |
| **Soil Moisture** | Estimated from weather + elevation | 📍 Green background |
| **Organic Carbon** (%) | Soil organic carbon content | 📍 Green background |
| **Electrical Conductivity** | Estimated from CEC data | 📍 Green background |

### 📍 **Location Data** (Purple Indicator)
| Field | Source | Visual Indicator |
|-------|--------|------------------|
| **Region** | Geocoded from coordinates | 📍 Purple background |
| **State** | OpenStreetMap geocoding | Display only |
| **District** | OpenStreetMap geocoding | Display only |
| **Elevation** | Open-Elevation API | Display in soil section |

---

## 🎯 How to Use

### Method 1: Use Your Current Location
```
1. Click "Use My Location" button
2. Allow browser location access
3. Wait 5-10 seconds for data to load
4. ✅ Fields auto-filled with weather & soil data
```

### Method 2: Select from Map
```
1. Click "Select from Map" button
2. Click anywhere on the interactive map
3. Click "Confirm & Fetch Soil Data"
4. Wait 5-10 seconds for APIs to respond
5. ✅ Fields auto-filled with weather & soil data
```

---

## 🔄 Manual Override

**You can ALWAYS manually edit any auto-filled value!**

- Auto-filled fields show colored backgrounds and icons
- Simply type new values to override
- Useful if you have more accurate local data
- System uses your manual values for prediction

---

## 📊 What Gets Auto-Filled in Each Module

### Fertilizer Recommendation
✅ Temperature, Humidity, Rainfall
✅ Soil Type, pH, Moisture, Organic Carbon, EC
✅ Region

### Stress Prediction
✅ Temperature, Humidity, Rainfall, Wind Speed
✅ Soil pH, Soil Moisture, Organic Matter
✅ Elevation

---

## 🌐 Data Sources

| API | What It Provides | Reliability |
|-----|------------------|-------------|
| **ISRIC SoilGrids** | Global soil properties at 250m resolution | High (scientific data) |
| **OpenWeatherMap** | Real-time weather conditions | High (updated hourly) |
| **Open-Elevation** | Terrain elevation data | High (SRTM data) |
| **OpenStreetMap** | Location names and boundaries | High (community maintained) |

---

## ⏱️ Performance

- **Data Fetch Time**: 5-15 seconds
- **Caching**: 24 hours (same location = instant)
- **Fallback**: Manual entry if APIs fail
- **Timeout Handling**: 10 seconds per API

---

## 🎨 Visual Indicators

### Field Colors:
- **Blue Background** 🌤️ = Weather data from map
- **Green Background** 📍 = Soil data from map
- **Purple Background** 📍 = Location data from map
- **White Background** = Manual entry needed

### Status Badges:
- ✅ **"Weather Data Fetched"** - Real-time weather loaded
- ✅ **"Soil Data Fetched"** - Scientific soil data loaded
- ⚠️ **"Unable to fetch"** - Please enter manually

---

## 🚫 What is NOT Auto-Filled

The following fields require farmer knowledge and cannot be auto-detected:

### Crop Information
- Crop Type
- Growth Stage
- Season

### NPK Levels (Requires Soil Testing)
- Nitrogen Level
- Phosphorus Level
- Potassium Level

### Agricultural History
- Irrigation Type
- Previous Crop
- Pest Damage
- Weed Coverage

**These require local knowledge or soil lab testing!**

---

## 💡 Tips for Best Results

1. **Zoom In**: Get closer to your exact field on the map for better accuracy
2. **Compare Data**: If you have recent soil test results, compare with auto-filled values
3. **Weather Timing**: Weather data is current - adjust if conditions changed
4. **Trust Soil Data**: SoilGrids data is scientifically validated for agriculture
5. **Manual Override**: Use local knowledge to improve any estimates

---

## 🔧 Troubleshooting

### "Unable to fetch soil data"
- **Cause**: Location might be in ocean/water body, or APIs temporarily down
- **Solution**: Enter soil values manually or try nearby location

### "Weather data unavailable"
- **Cause**: Weather API rate limit or network issue
- **Solution**: Use manual weather values or try again in a minute

### Location not accurate
- **Cause**: Browser location approximation
- **Solution**: Use "Select from Map" method and zoom in to exact field

---

## 📝 Example Workflow

```
1. 🗺️ Open Fertilizer Recommendation page
2. 📍 Click "Use My Location" or "Select from Map"
3. ⏳ Wait for auto-fill (5-10 seconds)
4. ✅ Review blue (weather) and green (soil) fields
5. ✏️ Adjust any values if you have better local data
6. 📝 Fill in crop details and NPK levels manually
7. 🚀 Click "Get Recommendation"
```

---

## 🎉 Benefits

- ⚡ **Faster**: No need to manually enter 8+ fields
- 🎯 **Accurate**: Real scientific soil data
- 🌍 **Global**: Works anywhere in the world
- 🔄 **Always Manual Override**: You're always in control
- 📱 **Farmer-Friendly**: Visual indicators show what's auto-filled

---

_Last Updated: February 23, 2026_
