import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import LoadingSpinner from '../components/LoadingSpinner';
import axios from 'axios';
import { AlertTriangle, MapPin, Sprout, Map } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================
// ENVIRONMENTAL DATA FETCHING HELPERS
// ============================================

/**
 * Fetch weather data from Open-Meteo API
 * Returns: temperature, humidity, rainfall, wind_speed
 */
async function fetchWeatherDataFromOpenMeteo(lat, lon) {
  try {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true&hourly=relativehumidity_2m,precipitation,windspeed_10m&timezone=auto`
    );
    
    if (!response.ok) throw new Error('Weather API failed');
    
    const data = await response.json();
    
    return {
      temperature: data.current_weather.temperature || 25,
      wind_speed: data.current_weather.windspeed || 10,
      humidity: data.hourly?.relativehumidity_2m?.[0] || 60,
      rainfall: data.hourly?.precipitation?.[0] || 50
    };
  } catch (error) {
    console.error('❌ Error fetching weather from Open-Meteo:', error);
    return {
      temperature: 25,
      wind_speed: 10,
      humidity: 60,
      rainfall: 50
    };
  }
}

/**
 * Fetch elevation data from Open-Elevation API
 * Returns: elevation in meters
 */
async function fetchElevationData(lat, lon) {
  try {
    const response = await fetch(
      `https://api.open-elevation.com/api/v1/lookup?locations=${lat},${lon}`
    );
    
    if (!response.ok) throw new Error('Elevation API failed');
    
    const data = await response.json();
    return data.results?.[0]?.elevation || 500;
  } catch (error) {
    console.error('❌ Error fetching elevation from Open-Elevation:', error);
    return 500; // Default fallback
  }
}

/**
 * Fetch all environmental data for a location
 */
async function fetchAllEnvironmentalData(lat, lon) {
  try {
    // Fetch weather and elevation in parallel
    const [weatherData, elevation] = await Promise.all([
      fetchWeatherDataFromOpenMeteo(lat, lon),
      fetchElevationData(lat, lon)
    ]);
    
    return {
      ...weatherData,
      elevation: elevation,
      water_flow: 50  // Default value - would need sensor data
    };
  } catch (error) {
    console.error('❌ Error fetching environmental data:', error);
    // Return defaults if everything fails
    return {
      temperature: 25,
      humidity: 60,
      rainfall: 50,
      wind_speed: 10,
      elevation: 500,
      water_flow: 50
    };
  }
}

const StressPrediction = () => {
  const [formData, setFormData] = useState({
    // Manual Farmer Inputs
    crop_type: '',
    growth_stage: '',
    soil_moisture: '',
    soil_ph: '7.0',
    organic_matter: '3.0',
    pest_damage: '0',
    weed_coverage: '0',
    
    // Auto-Fetch from Map/APIs
    temperature: '',
    humidity: '',
    rainfall: '',
    wind_speed: '',
    elevation: '500',
    water_flow: '50',
    drainage: '70',
    
    // Location
    lat: '',
    lng: ''
  });
  
  const [options, setOptions] = useState({
    crop_types: [],
    growth_stages: [],
    stress_types: [],
    indicators: [],
    severity_levels: []
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showMap, setShowMap] = useState(false);
  const [mapLoading, setMapLoading] = useState(false);
  const [loadingCurrentLocation, setLoadingCurrentLocation] = useState(false);
  const [soilDataFetched, setSoilDataFetched] = useState(false);
  const [locationData, setLocationData] = useState(null);
  const [optionsLoading, setOptionsLoading] = useState(true);

  // Load dropdown options on mount
  useEffect(() => {
    const loadOptions = async () => {
      setOptionsLoading(true);
      try {
        const response = await axios.get(`${API_URL}/api/stress/options`);
        if (response.data.status === 'success' || response.data.data) {
          const data = response.data.data || response.data;
          // Validate and set options with safe defaults
          setOptions({
            crop_types: Array.isArray(data.crop_types) ? data.crop_types : [],
            growth_stages: Array.isArray(data.growth_stages) ? data.growth_stages : [],
            stress_types: Array.isArray(data.stress_types) ? data.stress_types : [],
            indicators: Array.isArray(data.indicators) ? data.indicators : [],
            severity_levels: Array.isArray(data.severity_levels) ? data.severity_levels : []
          });
        }
      } catch (error) {
        console.error('Failed to load stress options:', error);
        // Ensure options has safe defaults even on error
        setOptions({
          crop_types: [],
          growth_stages: [],
          stress_types: [],
          indicators: [],
          severity_levels: []
        });
      } finally {
        setOptionsLoading(false);
      }
    };
    loadOptions();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleReset = () => {
    setFormData({
      crop_type: '',
      growth_stage: '',
      soil_moisture: '',
      soil_ph: '7.0',
      organic_matter: '3.0',
      pest_damage: '0',
      weed_coverage: '0',
      temperature: '',
      humidity: '',
      rainfall: '',
      wind_speed: '',
      elevation: '500',
      water_flow: '50',
      drainage: '70',
      lat: '',
      lng: ''
    });
    setResult(null);
  };

  const handleMapLocationSelect = async (lat, lng) => {
    setMapLoading(true);
    try {
      // Fetch environmental data from external APIs
      const environmentalData = await fetchAllEnvironmentalData(lat, lng);
      
      // Update form with auto-filled environmental data
      setFormData(prev => ({
        ...prev,
        // Environmental data (AUTO-FILLED, READ-ONLY)
        temperature: environmentalData.temperature.toFixed(1),
        humidity: environmentalData.humidity.toFixed(0),
        rainfall: environmentalData.rainfall.toFixed(1),
        wind_speed: environmentalData.wind_speed.toFixed(1),
        elevation: environmentalData.elevation.toString(),
        water_flow: environmentalData.water_flow.toString(),
        
        // Location
        lat: lat.toString(),
        lng: lng.toString()
      }));

      setLocationData(environmentalData);
      setShowMap(false);
      
      // Build success notification
      const notification = `✅ Location Data Fetched!\n\nWeather Information:\n` +
        `🌡️ Temperature: ${environmentalData.temperature.toFixed(1)}°C\n` +
        `💧 Humidity: ${environmentalData.humidity.toFixed(0)}%\n` +
        `🌧️ Rainfall: ${environmentalData.rainfall.toFixed(1)}mm\n` +
        `💨 Wind Speed: ${environmentalData.wind_speed.toFixed(1)}km/h\n` +
        `⛰️ Elevation: ${environmentalData.elevation}m\n` +
        `💦 Water Flow: ${environmentalData.water_flow}L/min`;
      
      alert(notification);
    } catch (err) {
      console.error('Error fetching environmental data:', err);
      alert('Failed to fetch environmental data. Please try again.');
    }
    setMapLoading(false);
  };

  // Get user's current location
  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    setLoadingCurrentLocation(true);
    
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        console.log('📍 Got current location:', latitude, longitude);
        
        // Use the same function as map click to populate fields
        await handleMapLocationSelect(latitude, longitude);
        setLoadingCurrentLocation(false);
      },
      (error) => {
        console.error('Error getting location:', error);
        let errorMessage = 'Could not get your location. ';
        
        switch(error.code) {
          case error.PERMISSION_DENIED:
            errorMessage += 'Please allow location access in your browser.';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage += 'Location information is unavailable.';
            break;
          case error.TIMEOUT:
            errorMessage += 'Location request timed out.';
            break;
          default:
            errorMessage += 'An unknown error occurred.';
        }
        
        alert(errorMessage);
        setLoadingCurrentLocation(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };

  // Helper for safe numeric parsing
  const safeParseFloat = (value, fieldName, defaultValue = null) => {
    if (value === '' || value === null || value === undefined) {
      if (defaultValue !== null) return defaultValue;
      console.error(`⚠️ Field ${fieldName} is empty`);
      return null;
    }
    const num = parseFloat(value);
    if (isNaN(num)) {
      console.error(`❌ Invalid number for ${fieldName}: "${value}"`);
      return null;
    }
    return num;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      // STEP 1: Validate required fields
      console.log('📋 Stress Form Data:', formData);
      
      if (!formData.crop_type) {
        console.error('❌ Missing crop_type');
        alert('❌ Please select a crop type');
        setLoading(false);
        return;
      }
      if (!formData.growth_stage) {
        console.error('❌ Missing growth_stage');
        alert('❌ Please select a growth stage');
        setLoading(false);
        return;
      }

      // STEP 2: Parse all numeric fields
      const soil_moisture = safeParseFloat(formData.soil_moisture, 'soil_moisture');
      const soil_ph = safeParseFloat(formData.soil_ph, 'soil_ph', 7.0);
      const organic_matter = safeParseFloat(formData.organic_matter, 'organic_matter', 3.0);
      const pest_damage = safeParseFloat(formData.pest_damage, 'pest_damage', 0);
      const weed_coverage = safeParseFloat(formData.weed_coverage, 'weed_coverage', 0);
      
      // These come from location
      const temperature = safeParseFloat(formData.temperature, 'temperature');
      const humidity = safeParseFloat(formData.humidity, 'humidity');
      const rainfall = safeParseFloat(formData.rainfall, 'rainfall');

      // Validate core required fields
      if (soil_moisture === null) {
        console.error('❌ Soil moisture is required and must be a valid number');
        alert('❌ Soil moisture is required. Please enter a valid number.');
        setLoading(false);
        return;
      }

      if (temperature === null || humidity === null || rainfall === null) {
        console.warn('⚠️ Environmental data incomplete. Please select a location to auto-fill weather data.');
        alert('⚠️ Environmental data missing. Please click "Use My Location" or "Select from Map" to fetch weather data.');
        setLoading(false);
        return;
      }

      // STEP 3: Build payload for /analyze endpoint
      const payload = {
        crop: formData.crop_type,
        parameters: {
          temperature,
          humidity,
          soil_moisture,
          rainfall,
          soil_ph,
          organic_matter,
          pest_damage,
          weed_coverage,
          growth_stage: formData.growth_stage,
          wind_speed: safeParseFloat(formData.wind_speed, 'wind_speed', 10),
          elevation: safeParseFloat(formData.elevation, 'elevation', 500),
          water_flow: safeParseFloat(formData.water_flow, 'water_flow', 50),
          drainage: safeParseFloat(formData.drainage, 'drainage', 70)
        }
      };

      console.log('📤 Sending stress prediction to /analyze with payload:', JSON.stringify(payload, null, 2));
      console.log('✅ All required fields validated:');
      console.log(`   - crop_type: ${formData.crop_type}`);
      console.log(`   - growth_stage: ${formData.growth_stage}`);
      console.log(`   - soil_moisture: ${soil_moisture}%`);
      console.log(`   - temperature: ${temperature}°C`);
      console.log(`   - humidity: ${humidity}%`);
      console.log(`   - rainfall: ${rainfall}mm`);
      
      // Use /analyze endpoint instead of /predict
      const response = await axios.post(`${API_URL}/api/stress/analyze`, payload);
      
      console.log('📥 Received response:', response.data);
      
      if (response.data.status === 'success' || response.data.stress_factors !== undefined) {
        setResult(response.data);
      } else {
        throw new Error(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('❌ Error:', err);
      
      // Extract detailed error information
      let errorMessage = 'Failed to predict stress level. Please check all inputs.';
      
      if (err.response?.status === 422) {
        console.error('🔴 Validation error (422):', err.response?.data);
        if (err.response?.data?.detail) {
          const detail = err.response.data.detail;
          if (Array.isArray(detail)) {
            errorMessage = detail.map(e => e.msg || e.type || JSON.stringify(e)).join('; ');
          } else if (typeof detail === 'string') {
            errorMessage = detail;
          } else if (typeof detail === 'object') {
            errorMessage = JSON.stringify(detail);
          }
        }
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      console.error(`❌ Error message: ${errorMessage}`);
      alert(errorMessage);
    }
    setLoading(false);
  };

  if (optionsLoading || !options.crop_types || options.crop_types.length === 0) {
    return (
      <div className="page-container">
        <Navbar />
        <div className="page-content flex items-center justify-center">
          <LoadingSpinner text="Loading stress prediction system..." />
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <AlertTriangle className="w-8 h-8 mr-3 text-primary-600" />
              Crop Stress Level Prediction
            </h1>
            <p className="text-gray-600">
              ML-based stress prediction using farmer-friendly inputs
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  {/* Location Section with Live Location */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <MapPin className="w-5 h-5 mr-2 text-primary-600" />
                        Location & Weather Data
                      </h3>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={getCurrentLocation}
                          disabled={loadingCurrentLocation}
                          className="text-sm px-3 py-1 bg-green-100 hover:bg-green-200 text-green-700 rounded-md flex items-center gap-1 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <MapPin className="w-4 h-4" />
                          {loadingCurrentLocation ? 'Getting Location...' : 'Use My Location'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowMap(!showMap)}
                          className="text-sm px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-md flex items-center gap-1 transition"
                        >
                          <Map className="w-4 h-4" />
                          {showMap ? 'Hide Map' : 'Select from Map'}
                        </button>
                      </div>
                    </div>

                    <div className="text-xs text-gray-600 mb-2">
                      Weather data will be auto-filled when you select a location
                    </div>
                  </div>

                  {/* Crop Information */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <Sprout className="w-5 h-5 mr-2 text-primary-600" />
                      Crop Information
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Crop Type <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="crop_type"
                          value={formData.crop_type}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select Crop</option>
                          {options?.crop_types?.map(crop => (
                            <option key={crop} value={crop}>{crop}</option>
                          )) || null}
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Growth Stage <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="growth_stage"
                          value={formData.growth_stage}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select Stage</option>
                          {options?.growth_stages?.map(stage => (
                            <option key={stage} value={stage}>{stage}</option>
                          )) || null}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Soil Parameters */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-primary-600 mr-2"></div>
                      Soil Parameters
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InputField
                        label="Soil Moisture (%)"
                        name="soil_moisture"
                        type="number"
                        value={formData.soil_moisture}
                        onChange={handleChange}
                        placeholder="50"
                        required
                        min="0"
                        max="100"
                        step="0.1"
                      />
                      <InputField
                        label="Soil pH"
                        name="soil_ph"
                        type="number"
                        value={formData.soil_ph}
                        onChange={handleChange}
                        placeholder="7.0"
                        required
                        min="0"
                        max="14"
                        step="0.1"
                      />
                      <InputField
                        label="Organic Matter (%)"
                        name="organic_matter"
                        type="number"
                        value={formData.organic_matter}
                        onChange={handleChange}
                        placeholder="3.0"
                        required
                        min="0"
                        max="10"
                        step="0.1"
                      />
                      <InputField
                        label="Drainage Quality (0-100)"
                        name="drainage"
                        type="number"
                        value={formData.drainage}
                        onChange={handleChange}
                        placeholder="70"
                        required
                        min="0"
                        max="100"
                        step="1"
                      />
                    </div>
                  </div>

                  {/* Pest & Weed */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-red-600 mr-2"></div>
                      Pest & Weed Status
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InputField
                        label="Pest Damage (%)"
                        name="pest_damage"
                        type="number"
                        value={formData.pest_damage}
                        onChange={handleChange}
                        placeholder="0"
                        required
                        min="0"
                        max="100"
                        step="1"
                      />
                      <InputField
                        label="Weed Coverage (%)"
                        name="weed_coverage"
                        type="number"
                        value={formData.weed_coverage}
                        onChange={handleChange}
                        placeholder="0"
                        required
                        min="0"
                        max="100"
                        step="1"
                      />
                    </div>
                  </div>

                  {/* Environmental Factors (Auto-filled from location, READ-ONLY) */}
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="text-sm font-semibold text-gray-800 mb-3">
                      🌍 Environmental Factors (Auto-filled from location - Read Only)
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Temperature (°C)
                        </label>
                        <input
                          type="number"
                          value={formData.temperature}
                          disabled
                          placeholder="Auto-filled"
                          step="0.1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Humidity (%)
                        </label>
                        <input
                          type="number"
                          value={formData.humidity}
                          disabled
                          placeholder="Auto-filled"
                          step="0.1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Rainfall (mm)
                        </label>
                        <input
                          type="number"
                          value={formData.rainfall}
                          disabled
                          placeholder="Auto-filled"
                          step="0.1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Wind Speed (km/h)
                        </label>
                        <input
                          type="number"
                          value={formData.wind_speed}
                          disabled
                          placeholder="Auto-filled"
                          step="0.1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Elevation (m)
                        </label>
                        <input
                          type="number"
                          value={formData.elevation}
                          disabled
                          placeholder="Auto-filled"
                          step="1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Water Flow (L/min)
                        </label>
                        <input
                          type="number"
                          value={formData.water_flow}
                          disabled
                          placeholder="Auto-filled"
                          step="1"
                          className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={handleReset}
                      className="btn-secondary flex-1"
                    >
                      Reset Form
                    </button>
                    <button
                      type="submit"
                      disabled={loading}
                      className="btn-primary flex-1"
                    >
                      {loading ? 'Analyzing...' : 'Predict Stress Level'}
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Analyzing stress indicators..." />
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    {/* Stress Level */}
                    <div className={`card ${
                      result.stress_level === 'Low' ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200' :
                      result.stress_level === 'Moderate' ? 'bg-gradient-to-br from-yellow-50 to-amber-50 border-2 border-yellow-200' :
                      'bg-gradient-to-br from-red-50 to-rose-50 border-2 border-red-200'
                    }`}>
                      <div className="flex items-center mb-4">
                        <div className={`p-3 rounded-lg mr-3 ${
                          result.stress_level === 'Low' ? 'bg-green-500' :
                          result.stress_level === 'Moderate' ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}>
                          <AlertTriangle className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-sm font-medium text-gray-600">Stress Level</h3>
                          <p className={`text-2xl font-bold mt-1 ${
                            result.stress_level === 'Low' ? 'text-green-700' :
                            result.stress_level === 'Moderate' ? 'text-yellow-700' :
                            'text-red-700'
                          }`}>{result.stress_level}</p>
                        </div>
                      </div>

                      <div className="mb-3">
                        <p className="text-xs text-gray-600">Model Confidence</p>
                        <p className="text-lg font-semibold text-gray-800">{result.confidence_percentage}</p>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                          <div 
                            className={`h-2 rounded-full ${
                              result.stress_level === 'Low' ? 'bg-green-500' :
                              result.stress_level === 'Moderate' ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`}
                            style={{ width: result.confidence_percentage }}
                          ></div>
                        </div>
                      </div>

                      <div className="pt-3 border-t border-gray-200">
                        <p className="text-sm text-gray-700">{result.advice}</p>
                      </div>
                    </div>

                    {/* Stress Factors */}
                    {result.stress_factors && result.stress_factors.length > 0 && (
                      <div className="card bg-orange-50 border border-orange-200">
                        <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center">
                          <AlertTriangle className="w-4 h-4 mr-2 text-orange-600" />
                          Identified Stress Factors
                        </h4>
                        <ul className="space-y-2">
                          {result.stress_factors.map((factor, idx) => (
                            <li key={idx} className="flex items-start text-sm text-gray-700">
                              <span className="text-orange-600 mr-2 font-bold">•</span>
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Recommendations */}
                    {result.recommendations && result.recommendations.length > 0 && (
                      <div className="card bg-blue-50 border border-blue-200">
                        <h4 className="text-sm font-semibold text-gray-800 mb-3">
                          Recommended Actions
                        </h4>
                        <div className="space-y-3">
                          {result.recommendations.map((rec, idx) => (
                            <div key={idx} className="bg-white p-3 rounded-md">
                              <p className="text-sm font-semibold text-blue-700">{rec.factor}</p>
                              <p className="text-xs text-gray-600 mt-1">{rec.action}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* AI-Powered Explanation (NEW) */}
                    {result.enhanced_with_ai && result.ai_explanation && (
                      <div className="card bg-purple-50 border border-purple-200">
                        <h4 className="text-sm font-semibold text-gray-800 mb-2 flex items-center">
                          <span className="text-lg mr-2">🧠</span>
                          AI Expert Analysis
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed mb-3">
                          {result.ai_explanation}
                        </p>
                        <div className="text-xs text-purple-600 italic">
                          Powered by {result.reasoning_source || 'Groq LLM'}
                        </div>
                      </div>
                    )}

                    {/* AI Recommendations (NEW) */}
                    {result.enhanced_with_ai && result.ai_recommendations && result.ai_recommendations.length > 0 && (
                      <div className="card bg-orange-50 border border-orange-200">
                        <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center">
                          <span className="text-lg mr-2">💡</span>
                          Expert Recommendations
                        </h4>
                        <div className="space-y-2">
                          {result.ai_recommendations.map((rec, idx) => (
                            <div key={idx} className="bg-white p-2 rounded-md border-l-3 border-orange-400">
                              <div className="flex items-start justify-between">
                                <p className="text-xs font-semibold text-gray-800">
                                  {rec.action}
                                </p>
                                <span className={`text-xs px-2 py-0.5 rounded font-semibold ${
                                  rec.priority === 'URGENT' ? 'bg-red-100 text-red-700' :
                                  rec.priority === 'High' ? 'bg-yellow-100 text-yellow-700' :
                                  rec.priority === 'Medium' ? 'bg-blue-100 text-blue-700' :
                                  'bg-gray-100 text-gray-700'
                                }`}>
                                  {rec.priority}
                                </span>
                              </div>
                              {rec.factor && (
                                <p className="text-xs text-gray-500 mt-1">
                                  <span className="font-medium">Factor:</span> {rec.factor}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <div className="text-center py-8">
                      <AlertTriangle className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                      <p className="text-gray-600 mb-2">No prediction yet</p>
                      <p className="text-sm text-gray-500">
                        Fill in the form and click "Predict Stress Level" to get results
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Stress Level Guide */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="card bg-green-50 border-l-4 border-green-500">
              <h4 className="font-semibold text-green-800 mb-2">Low Stress</h4>
              <p className="text-sm text-gray-700">
                Crops are in optimal health. Continue current management practices and monitor regularly.
              </p>
            </div>
            <div className="card bg-yellow-50 border-l-4 border-yellow-500">
              <h4 className="font-semibold text-yellow-800 mb-2">Moderate Stress</h4>
              <p className="text-sm text-gray-700">
                Monitor closely. Adjust irrigation, pest control, or nutrient management as needed.
              </p>
            </div>
            <div className="card bg-red-50 border-l-4 border-red-500">
              <h4 className="font-semibold text-red-800 mb-2">High Stress</h4>
              <p className="text-sm text-gray-700">
                Immediate intervention required. Take urgent action to prevent yield loss.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StressPrediction;
