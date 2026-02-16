import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import LoadingSpinner from '../components/LoadingSpinner';
import axios from 'axios';
import { AlertTriangle, MapPin, Sprout, Map } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

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
  
  const [options, setOptions] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showMap, setShowMap] = useState(false);
  const [mapLoading, setMapLoading] = useState(false);
  const [loadingCurrentLocation, setLoadingCurrentLocation] = useState(false);

  // Load dropdown options on mount
  useEffect(() => {
    const loadOptions = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/stress/options`);
        if (response.data.success) {
          setOptions(response.data.options);
        }
      } catch (error) {
        console.error('Failed to load stress options:', error);
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
      const response = await axios.post(`${API_URL}/api/stress/location-data`, {
        latitude: lat,
        longitude: lng
      });

      if (response.data.success) {
        const data = response.data;
        
        // Auto-fill weather and location data
        setFormData(prev => ({
          ...prev,
          temperature: data.temperature?.toFixed(1) || prev.temperature,
          humidity: data.humidity?.toFixed(0) || prev.humidity,
          rainfall: data.rainfall?.toFixed(1) || prev.rainfall,
          wind_speed: data.wind_speed?.toFixed(1) || prev.wind_speed,
          elevation: data.elevation?.toString() || prev.elevation,
          water_flow: data.water_flow?.toString() || prev.water_flow,
          drainage: data.drainage?.toString() || prev.drainage,
          lat: lat.toString(),
          lng: lng.toString()
        }));

        setShowMap(false);
        alert(`Location data fetched!\nWeather data has been auto-filled.`);
      }
    } catch (err) {
      console.error('Error fetching location data:', err);
      alert('Failed to fetch location data. Please try again or enter manually.');
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        // Manual inputs
        crop_type: formData.crop_type,
        growth_stage: formData.growth_stage,
        soil_moisture: parseFloat(formData.soil_moisture),
        soil_ph: parseFloat(formData.soil_ph),
        organic_matter: parseFloat(formData.organic_matter),
        pest_damage: parseFloat(formData.pest_damage),
        weed_coverage: parseFloat(formData.weed_coverage),
        
        // Auto-fetch/Manual weather
        temperature: parseFloat(formData.temperature),
        humidity: parseFloat(formData.humidity),
        rainfall: parseFloat(formData.rainfall),
        wind_speed: parseFloat(formData.wind_speed),
        elevation: parseFloat(formData.elevation),
        water_flow: parseFloat(formData.water_flow),
        drainage: parseFloat(formData.drainage),
        
        // Location
        lat: parseFloat(formData.lat) || 20.5937,
        lng: parseFloat(formData.lng) || 78.9629
      };

      console.log('📤 Sending stress prediction request:', payload);
      
      const response = await axios.post(`${API_URL}/api/stress/predict`, payload);
      
      if (response.data.success) {
        setResult(response.data);
        console.log('📥 Received prediction:', response.data);
      } else {
        throw new Error(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Error:', err);
      alert(`Prediction failed: ${err.response?.data?.error || err.message}`);
    }
    setLoading(false);
  };

  if (!options) {
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
                          {options.crop_types.map(crop => (
                            <option key={crop} value={crop}>{crop}</option>
                          ))}
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
                          {options.growth_stages.map(stage => (
                            <option key={stage} value={stage}>{stage}</option>
                          ))}
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

                  {/* Environmental Factors (Auto-filled from location) */}
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="text-sm font-semibold text-gray-800 mb-3">
                      Environmental Factors (Auto-filled from location)
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InputField
                        label="Temperature (°C)"
                        name="temperature"
                        type="number"
                        value={formData.temperature}
                        onChange={handleChange}
                        placeholder="Auto-filled"
                        required
                        step="0.1"
                      />
                      <InputField
                        label="Humidity (%)"
                        name="humidity"
                        type="number"
                        value={formData.humidity}
                        onChange={handleChange}
                        placeholder="Auto-filled"
                        required
                        step="0.1"
                      />
                      <InputField
                        label="Rainfall (mm)"
                        name="rainfall"
                        type="number"
                        value={formData.rainfall}
                        onChange={handleChange}
                        placeholder="Auto-filled"
                        required
                        step="0.1"
                      />
                      <InputField
                        label="Wind Speed (km/h)"
                        name="wind_speed"
                        type="number"
                        value={formData.wind_speed}
                        onChange={handleChange}
                        placeholder="Auto-filled"
                        required
                        step="0.1"
                      />
                      <InputField
                        label="Elevation (m)"
                        name="elevation"
                        type="number"
                        value={formData.elevation}
                        onChange={handleChange}
                        placeholder="500"
                        required
                        step="1"
                      />
                      <InputField
                        label="Water Flow (L/min)"
                        name="water_flow"
                        type="number"
                        value={formData.water_flow}
                        onChange={handleChange}
                        placeholder="50"
                        required
                        step="1"
                      />
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
