import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import LoadingSpinner from '../components/LoadingSpinner';
import { Droplet, Sparkles, MapPin, X } from 'lucide-react';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const AUTO_FILL_FIELD_MAP = {
  Soil_pH: 'soil_pH',
  Nitrogen_Level: 'nitrogen',
  Phosphorus_Level: 'phosphorus',
  Potassium_Level: 'potassium'
};
// Note: Soil_Moisture, Organic_Carbon, and Electrical_Conductivity are now hidden and use default values

const AUTO_FILL_API_FIELDS = Object.values(AUTO_FILL_FIELD_MAP);

const FertilizerRecommendation = () => {
  const [formData, setFormData] = useState({
    // Soil characteristics
    Soil_Type: '',
    Soil_pH: '',
    
    // NPK Levels
    Nitrogen_Level: '',
    Phosphorus_Level: '',
    Potassium_Level: '',
    
    // Crop information
    Crop_Type: '',
    Crop_Growth_Stage: '',
    Season: '',
    
    // Environmental factors
    Temperature: '',
    Humidity: '',
    Rainfall: '',
    
    // Agricultural metadata
    Irrigation_Type: '',
    Previous_Crop: '',
    Region: ''
  });
  
  const [options, setOptions] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [locationData, setLocationData] = useState(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [loadingCurrentLocation, setLoadingCurrentLocation] = useState(false);
  const [loadingWeather, setLoadingWeather] = useState(false);
  const [useMapData, setUseMapData] = useState(true); // Toggle for auto data vs manual
  const [soilDataFetched, setSoilDataFetched] = useState(false); // Track if soil data was fetched
  const [weatherDataFetched, setWeatherDataFetched] = useState(false); // Track if weather data was fetched
  const [autoFillLoading, setAutoFillLoading] = useState(false);
  const [autoFillMessage, setAutoFillMessage] = useState('');
  const [autoFieldStatus, setAutoFieldStatus] = useState({});

  const isAutoFieldAvailable = (formFieldName) => {
    if (!useMapData) return true;
    if (!soilDataFetched) return true;
    const apiField = AUTO_FILL_FIELD_MAP[formFieldName];
    if (!apiField) return true;
    return autoFieldStatus[apiField] === true;
  };

  const autoUnavailableFields = useMapData
    ? Object.keys(AUTO_FILL_FIELD_MAP).filter((field) => !isAutoFieldAvailable(field))
    : [];

  // Load dropdown options on mount
  useEffect(() => {
    const loadOptions = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/fertilizer/options`);
        if (response.data.success) {
          setOptions(response.data.options);
        }
      } catch (error) {
        console.error('Failed to load fertilizer options:', error);
      }
    };
    
    const loadModelInfo = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/fertilizer/model-info`);
        if (response.data.success) {
          setModelInfo(response.data);
        }
      } catch (error) {
        console.error('Failed to load model info:', error);
      }
    };
    
    loadOptions();
    loadModelInfo();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleReset = () => {
    setFormData({
      Soil_Type: '',
      Soil_pH: '',
      Nitrogen_Level: '',
      Phosphorus_Level: '',
      Potassium_Level: '',
      Crop_Type: '',
      Crop_Growth_Stage: '',
      Season: '',
      Temperature: '',
      Humidity: '',
      Rainfall: '',
      Irrigation_Type: '',
      Previous_Crop: '',
      Region: ''
    });
    setResult(null);
    setLocationData(null);
    setWeatherDataFetched(false);
    setSoilDataFetched(false);
    setAutoFillMessage('');
    setAutoFieldStatus({});
    setUseMapData(true);
  };

  const handleMapLocationSelect = async (lat, lng) => {
    setMapLoading(true);
    setAutoFillLoading(true);
    setAutoFillMessage('Fetching soil data...');

    try {
      const response = await axios.post(`${API_URL}/api/fertilizer/location-data`, {
        latitude: lat,
        longitude: lng
      });

      if (response.data.success) {
        const data = response.data;
        
        // Debug: Log received data
        console.log('📊 Received location data:', {
          temperature: data.temperature,
          humidity: data.humidity,
          rainfall: data.rainfall,
          soil_pH: data.soil_pH,
          soil_type: data.soil_type
        });
        
        setLocationData(data);

        // Build notification message
        let notificationParts = [`📍 Location detected!`];
        if (data.state || data.district) {
          notificationParts.push(`State: ${data.state || 'N/A'}, District: ${data.district || 'N/A'}`);
        }
        
        // Check if weather data was fetched
        const hasWeatherData = data.temperature !== null && data.temperature !== undefined || 
                               data.humidity !== null && data.humidity !== undefined || 
                               data.rainfall !== null && data.rainfall !== undefined;
        if (hasWeatherData) {
          notificationParts.push(`\n✅ Weather data: ${data.temperature?.toFixed(1) || 'N/A'}°C, ${data.humidity?.toFixed(0) || 'N/A'}% humidity`);
          setWeatherDataFetched(true);
        } else {
          setWeatherDataFetched(false);
        }
        
        // Check if soil data was fetched
        const hasSoilData = data.soil_pH || data.soil_type || data.elevation;
        if (hasSoilData) {
          notificationParts.push(`✅ Soil data: pH ${data.soil_pH || 'N/A'}, Type: ${data.soil_type || 'N/A'}`);
        } else {
          notificationParts.push(`⚠️ Soil data unavailable - please enter manually`);
        }

        // Autofill location and weather metadata.
        setFormData(prev => ({
          ...prev,
          Region: data.region !== undefined && data.region !== null ? data.region : prev.Region,
          Temperature: data.temperature !== undefined && data.temperature !== null ? String(data.temperature) : prev.Temperature,
          Humidity: data.humidity !== undefined && data.humidity !== null ? String(data.humidity) : prev.Humidity,
          Rainfall: data.rainfall !== undefined && data.rainfall !== null ? String(data.rainfall) : prev.Rainfall,
          Soil_Type: data.soil_type || prev.Soil_Type
        }));
        
        // Call new fertilizer-only auto-fill endpoint for soil + NPK fields.
        const autoFillResponse = await axios.post(
          `${API_URL}/fertilizer/auto-fill`,
          { latitude: lat, longitude: lng },
          { timeout: 2000 }
        );

        const autoData = autoFillResponse.data || {};
        const newStatus = {};
        AUTO_FILL_API_FIELDS.forEach((field) => {
          newStatus[field] = autoData[field] !== null && autoData[field] !== undefined;
        });
        setAutoFieldStatus(newStatus);

        const hasAnyAutoField = AUTO_FILL_API_FIELDS.some((field) => newStatus[field]);

        if (hasAnyAutoField) {
          setSoilDataFetched(true);
          setAutoFillMessage('Auto data applied. Unavailable fields are hidden in auto mode.');
          setFormData((prev) => ({
            ...prev,
            Soil_pH: newStatus.soil_pH ? String(autoData.soil_pH) : prev.Soil_pH
            // Note: Soil_Moisture, Organic_Carbon, Electrical_Conductivity are not included in form anymore
            // They will be added with default values during submission
          }));
        } else {
          setSoilDataFetched(false);
          setUseMapData(false);
          setAutoFillMessage('Soil data not available. Please enter manually.');
          notificationParts.push('⚠️ Soil data not available. Manual input enabled.');
        }

        setShowMap(false);
        alert(notificationParts.join('\n'));
      }
    } catch (err) {
      console.error('Error fetching location data:', err);
      setUseMapData(false);
      setSoilDataFetched(false);
      setAutoFieldStatus({});
      setAutoFillMessage('Soil data not available. Please enter manually.');
      alert('Failed to fetch location data. Manual input is enabled.');
    } finally {
      setAutoFillLoading(false);
      setMapLoading(false);
    }
  };

  const handleResetLocation = () => {
    setLocationData(null);
    setSoilDataFetched(false);
    setWeatherDataFetched(false);
    setAutoFillMessage('');
    setAutoFieldStatus({});
    setUseMapData(true);
    setFormData(prev => ({
      ...prev,
      // Clear location & weather data
      Region: '',
      Temperature: '',
      Humidity: '',
      Rainfall: '',
      // Clear soil data
      Soil_Type: '',
      Soil_pH: ''
    }));
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

  // Auto-fill ONLY weather data (Temperature, Humidity, Rainfall)
  const autoFillWeatherData = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    setLoadingWeather(true);
    
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        console.log('🌤️ Fetching weather for location:', latitude, longitude);
        
        try {
          // Fetch weather data from Open-Meteo API directly
          const weatherResponse = await fetch(
            `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto`
          );
          
          if (!weatherResponse.ok) {
            throw new Error('Failed to fetch weather data');
          }
          
          const weatherData = await weatherResponse.json();
          const current = weatherData.current;
          
          // Extract weather values
          const temperature = current.temperature_2m || 0;
          const humidity = current.relative_humidity_2m || 0;
          const rainfall = current.precipitation || 0;
          
          console.log('✅ Weather data fetched:', { temperature, humidity, rainfall });
          
          // Update only Temperature, Humidity, and Rainfall fields
          setFormData(prev => ({
            ...prev,
            Temperature: String(temperature),
            Humidity: String(humidity),
            Rainfall: String(rainfall)
          }));
          
          setWeatherDataFetched(true);
          
          alert(`✅ Weather Data Auto-Filled!\n\n🌡️ Temperature: ${temperature.toFixed(1)}°C\n💧 Humidity: ${humidity.toFixed(0)}%\n🌧️ Rainfall: ${rainfall.toFixed(1)}mm\n\nYou can modify these values if needed.`);
        } catch (error) {
          console.error('Error fetching weather:', error);
          alert('Failed to fetch weather data. Please try again or enter manually.');
        }
        
        setLoadingWeather(false);
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
        setLoadingWeather(false);
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

    if (useMapData && autoUnavailableFields.length > 0) {
      setUseMapData(false);
      alert('Some auto data fields are unavailable. Switched to manual mode, please fill missing values.');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // Build payload with hidden default values for simplified soil parameters
      const payload = {
        ...formData,
        // Hidden default values for simplified UI (sent internally to model)
        Soil_Moisture: 50,
        Organic_Carbon: 0.8,
        Electrical_Conductivity: 1.2
      };

      const response = await axios.post(`${API_URL}/api/fertilizer/recommend`, payload);
      
      if (response.data.success) {
        setResult(response.data);
      }
    } catch (err) {
      console.error('Error:', err);
      alert(err.response?.data?.detail || 'Failed to get recommendation. Please check all inputs.');
    }
    setLoading(false);
  };

  if (!options) {
    return (
      <div className="page-container">
        <Navbar />
        <div className="page-content flex items-center justify-center">
          <LoadingSpinner text="Loading fertilizer recommendation system..." />
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
                  <Droplet className="w-8 h-8 mr-3 text-primary-600" />
                  Fertilizer Recommendation
                </h1>
                <p className="text-gray-600">ML-based fertilizer recommendations using soil and crop analysis</p>
              </div>
              {modelInfo && (
                <div className="text-right">
                  <div className="text-sm text-gray-600">Model Accuracy</div>
                  <div className="text-2xl font-bold text-green-600">{modelInfo.accuracy_percentage}%</div>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Form Section */}
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  
                  {/* Location Selector */}
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center">
                        <MapPin className="w-5 h-5 mr-2 text-blue-600" />
                        <h3 className="text-sm font-semibold text-gray-800">Location-Based Autofill</h3>
                      </div>
                      {locationData && (
                        <button
                          type="button"
                          onClick={handleResetLocation}
                          className="text-xs text-red-600 hover:text-red-800 flex items-center"
                        >
                          <X className="w-3 h-3 mr-1" />
                          Reset Location
                        </button>
                      )}
                    </div>

                    <div className="mb-3 flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setUseMapData(true)}
                        className={`px-3 py-1.5 text-xs rounded border ${useMapData ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300'}`}
                      >
                        Use Auto Data
                      </button>
                      <button
                        type="button"
                        onClick={() => setUseMapData(false)}
                        className={`px-3 py-1.5 text-xs rounded border ${!useMapData ? 'bg-gray-700 text-white border-gray-700' : 'bg-white text-gray-700 border-gray-300'}`}
                      >
                        Enter Manually
                      </button>
                    </div>

                    {(autoFillLoading || autoFillMessage) && (
                      <div className={`mb-3 p-2 rounded text-xs ${autoFillLoading ? 'bg-blue-100 text-blue-800 border border-blue-300' : 'bg-amber-50 text-amber-800 border border-amber-300'}`}>
                        {autoFillLoading ? 'Fetching soil data...' : autoFillMessage}
                      </div>
                    )}
                    
                    {locationData ? (
                      <div className="bg-white p-3 rounded border border-blue-300">
                        <div className="grid grid-cols-2 gap-2 text-sm mb-2">
                          <div>
                            <span className="text-gray-600">State:</span>
                            <span className="font-medium ml-2">{locationData.state || 'N/A'}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">District:</span>
                            <span className="font-medium ml-2">{locationData.district || 'N/A'}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Region:</span>
                            <span className="font-medium ml-2">{locationData.region || 'N/A'}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Lat/Lng:</span>
                            <span className="font-medium ml-2 text-xs">
                              {locationData.latitude?.toFixed(3)}, {locationData.longitude?.toFixed(3)}
                            </span>
                          </div>
                        </div>
                        
                        {weatherDataFetched && (
                          <div className="pt-2 border-t border-blue-200 mb-2">
                            <div className="flex items-center mb-1">
                              <span className="text-xs text-blue-600 font-semibold">✅ Weather Data Fetched</span>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              {locationData.temperature && (
                                <div>
                                  <span className="text-gray-600">Temp:</span>
                                  <span className="font-medium ml-1">{locationData.temperature.toFixed(1)}°C</span>
                                </div>
                              )}
                              {locationData.humidity && (
                                <div>
                                  <span className="text-gray-600">Humidity:</span>
                                  <span className="font-medium ml-1">{locationData.humidity.toFixed(0)}%</span>
                                </div>
                              )}
                              {locationData.rainfall !== undefined && (
                                <div>
                                  <span className="text-gray-600">Rainfall:</span>
                                  <span className="font-medium ml-1">{locationData.rainfall.toFixed(1)}mm</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {soilDataFetched && (
                          <div className="pt-2 border-t border-green-200">
                            <div className="flex items-center mb-1">
                              <span className="text-xs text-green-600 font-semibold">✅ Soil Data Fetched</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              {locationData.soil_type && (
                                <div>
                                  <span className="text-gray-600">Type:</span>
                                  <span className="font-medium ml-1">{locationData.soil_type}</span>
                                </div>
                              )}
                              {locationData.soil_pH && (
                                <div>
                                  <span className="text-gray-600">pH:</span>
                                  <span className="font-medium ml-1">{locationData.soil_pH}</span>
                                </div>
                              )}
                              {locationData.elevation && (
                                <div>
                                  <span className="text-gray-600">Elevation:</span>
                                  <span className="font-medium ml-1">{locationData.elevation}m</span>
                                </div>
                              )}
                              {locationData.soil_moisture && (
                                <div>
                                  <span className="text-gray-600">Moisture:</span>
                                  <span className="font-medium ml-1">{locationData.soil_moisture.toFixed(1)}%</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={getCurrentLocation}
                          disabled={loadingCurrentLocation}
                          className="btn-secondary flex-1 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed bg-green-100 hover:bg-green-200 text-green-700"
                        >
                          <MapPin className="w-4 h-4 mr-2" />
                          {loadingCurrentLocation ? 'Getting Location...' : 'Use My Location'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowMap(true)}
                          className="btn-secondary flex-1 flex items-center justify-center"
                        >
                          <MapPin className="w-4 h-4 mr-2" />
                          Select from Map
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Map Modal */}
                  {showMap && (
                    <MapSelector 
                      onLocationSelect={handleMapLocationSelect}
                      onClose={() => setShowMap(false)}
                      loading={mapLoading}
                    />
                  )}
                  
                  {/* Soil Characteristics */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-primary-600 mr-2"></div>
                      Soil Characteristics
                      {soilDataFetched && (
                        <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                          📍 Auto-filled from map
                        </span>
                      )}
                    </h3>
                    
                    {soilDataFetched && (
                      <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-800">
                        <span className="font-medium">✓ Soil data loaded from location.</span> Toggle to manual mode to enter all fields.
                      </div>
                    )}
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Soil Type <span className="text-red-500">*</span>
                          {locationData?.soil_type && formData.Soil_Type === locationData.soil_type && (
                            <span className="ml-1 text-xs text-green-600">📍</span>
                          )}
                        </label>
                        <select
                          name="Soil_Type"
                          value={formData.Soil_Type}
                          onChange={handleChange}
                          required
                          className={`input-field ${locationData?.soil_type && formData.Soil_Type === locationData.soil_type ? 'bg-green-50 border-green-300' : ''}`}
                        >
                          <option value="">Select soil type</option>
                          {options.Soil_Type?.map(type => (
                            <option key={type} value={type}>{type}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Soil pH <span className="text-red-500">*</span>
                          {useMapData && isAutoFieldAvailable('Soil_pH') && (
                            <span className="ml-1 text-xs text-green-600">Auto-filled</span>
                          )}
                        </label>
                        {(!useMapData || isAutoFieldAvailable('Soil_pH')) && (
                        <input
                          type="number"
                          name="Soil_pH"
                          value={formData.Soil_pH}
                          onChange={handleChange}
                          placeholder="4.0 - 9.0"
                          required
                          min="4"
                          max="9"
                          step="0.1"
                          className={`input-field ${useMapData && isAutoFieldAvailable('Soil_pH') ? 'bg-green-50 border-green-300' : ''}`}
                        />
                        )}
                      </div>
                      

                    </div>
                  </div>

                  {/* NPK Levels */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-green-600 mr-2"></div>
                      NPK Nutrient Levels (mg/kg)
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {(!useMapData || isAutoFieldAvailable('Nitrogen_Level')) && (
                      <InputField
                        label={useMapData && isAutoFieldAvailable('Nitrogen_Level') ? 'Nitrogen Level (Auto-filled)' : 'Nitrogen Level'}
                        name="Nitrogen_Level"
                        type="number"
                        value={formData.Nitrogen_Level}
                        onChange={handleChange}
                        placeholder="0 - 150"
                        required
                        min="0"
                        max="200"
                        step="0.1"
                      />
                      )}

                      {(!useMapData || isAutoFieldAvailable('Phosphorus_Level')) && (
                      <InputField
                        label={useMapData && isAutoFieldAvailable('Phosphorus_Level') ? 'Phosphorus Level (Auto-filled)' : 'Phosphorus Level'}
                        name="Phosphorus_Level"
                        type="number"
                        value={formData.Phosphorus_Level}
                        onChange={handleChange}
                        placeholder="0 - 150"
                        required
                        min="0"
                        max="200"
                        step="0.1"
                      />
                      )}

                      {(!useMapData || isAutoFieldAvailable('Potassium_Level')) && (
                      <InputField
                        label={useMapData && isAutoFieldAvailable('Potassium_Level') ? 'Potassium Level (Auto-filled)' : 'Potassium Level'}
                        name="Potassium_Level"
                        type="number"
                        value={formData.Potassium_Level}
                        onChange={handleChange}
                        placeholder="0 - 300"
                        required
                        min="0"
                        max="400"
                        step="0.1"
                      />
                      )}
                    </div>
                  </div>

                  {/* Crop Information */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-yellow-600 mr-2"></div>
                      Crop Information
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Crop Type <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Crop_Type"
                          value={formData.Crop_Type}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select crop</option>
                          {options.Crop_Type?.map(crop => (
                            <option key={crop} value={crop}>{crop}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Growth Stage <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Crop_Growth_Stage"
                          value={formData.Crop_Growth_Stage}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select stage</option>
                          {options.Crop_Growth_Stage?.map(stage => (
                            <option key={stage} value={stage}>{stage}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Season <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Season"
                          value={formData.Season}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select season</option>
                          {options.Season?.map(season => (
                            <option key={season} value={season}>{season}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Environmental Factors */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <div className="w-2 h-6 bg-blue-600 mr-2"></div>
                        Environmental Conditions
                        {weatherDataFetched && (
                          <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                            🌤️ Auto-filled
                          </span>
                        )}
                      </h3>
                      
                      {/* Auto-Fill Weather Button */}
                      <button
                        type="button"
                        onClick={autoFillWeatherData}
                        disabled={loadingWeather}
                        className="px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-lg text-sm font-medium shadow-md transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loadingWeather ? (
                          <>
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Fetching...</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-4 h-4" />
                            <span>Auto-Fill Weather</span>
                          </>
                        )}
                      </button>
                    </div>
                    
                    {weatherDataFetched && (
                      <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
                        <span className="font-medium">✓ Weather data loaded automatically.</span> You can still modify any values manually if needed.
                      </div>
                    )}
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <InputField
                        label={
                          <span>
                            Temperature (°C)
                            {weatherDataFetched && (
                              <span className="ml-1 text-xs text-blue-600">🌤️ Auto-filled</span>
                            )}
                          </span>
                        }
                        name="Temperature"
                        type="number"
                        value={formData.Temperature}
                        onChange={handleChange}
                        placeholder="0 - 50"
                        required
                        min="0"
                        max="50"
                        step="0.1"
                        className={weatherDataFetched ? 'bg-blue-50 border-blue-300' : ''}
                      />
                      
                      <InputField
                        label={
                          <span>
                            Humidity (%)
                            {weatherDataFetched && (
                              <span className="ml-1 text-xs text-blue-600">🌤️ Auto-filled</span>
                            )}
                          </span>
                        }
                        name="Humidity"
                        type="number"
                        value={formData.Humidity}
                        onChange={handleChange}
                        placeholder="0 - 100"
                        required
                        min="0"
                        max="100"
                        step="0.1"
                        className={weatherDataFetched ? 'bg-blue-50 border-blue-300' : ''}
                      />
                      
                      <InputField
                        label={
                          <span>
                            Rainfall (mm)
                            {weatherDataFetched && (
                              <span className="ml-1 text-xs text-blue-600">🌤️ Auto-filled</span>
                            )}
                          </span>
                        }
                        name="Rainfall"
                        type="number"
                        value={formData.Rainfall}
                        onChange={handleChange}
                        placeholder="0 - 500"
                        required
                        min="0"
                        max="1000"
                        step="0.1"
                        className={weatherDataFetched ? 'bg-blue-50 border-blue-300' : ''}
                      />
                    </div>
                  </div>

                  {/* Agricultural Metadata */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-purple-600 mr-2"></div>
                      Agricultural Background
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Irrigation Type <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Irrigation_Type"
                          value={formData.Irrigation_Type}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select irrigation</option>
                          {options.Irrigation_Type?.map(type => (
                            <option key={type} value={type}>{type}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Previous Crop <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Previous_Crop"
                          value={formData.Previous_Crop}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select previous crop</option>
                          {options.Previous_Crop?.map(crop => (
                            <option key={crop} value={crop}>{crop}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Region <span className="text-red-500">*</span>
                          {locationData?.region && formData.Region === locationData.region && (
                            <span className="ml-1 text-xs text-purple-600">📍</span>
                          )}
                        </label>
                        <select
                          name="Region"
                          value={formData.Region}
                          onChange={handleChange}
                          required
                          className={`input-field ${locationData?.region && formData.Region === locationData.region ? 'bg-purple-50 border-purple-300' : ''}`}
                        >
                          <option value="">Select region</option>
                          {options.Region?.map(region => (
                            <option key={region} value={region}>{region}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
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
                      {loading ? 'Analyzing...' : 'Get Recommendation'}
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {/* Results Section */}
            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Analyzing soil and crop data..." />
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    {/* Main Recommendation */}
                    <div className="card bg-gradient-to-br from-green-50 to-emerald-50">
                      <div className="flex items-center mb-4">
                        <div className="p-3 bg-green-500 rounded-lg mr-3">
                          <Sparkles className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-sm font-medium text-gray-600">Recommended Fertilizer</h3>
                          <p className="text-2xl font-bold text-green-700 mt-1">{result.fertilizer}</p>
                        </div>
                      </div>

                      <div className="mt-4 pt-4 border-t border-green-200">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-600">Confidence</span>
                          <span className="text-lg font-bold text-green-600">
                            {result.confidence_percentage}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${result.confidence_percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    {/* Top 3 Recommendations */}
                    {result.top_3_recommendations && (
                      <div className="card">
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">
                          Top 3 Recommendations
                        </h4>
                        <div className="space-y-2">
                          {result.top_3_recommendations.slice(0, 3).map((fert, idx) => {
                            const prob = result.all_probabilities[fert];
                            return (
                              <div key={idx} className="flex items-center justify-between text-sm">
                                <span className="text-gray-700">
                                  {idx + 1}. {fert}
                                </span>
                                <span className="text-gray-600 font-medium">
                                  {(prob * 100).toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Model Info */}
                    {modelInfo && (
                      <div className="card bg-blue-50">
                        <h4 className="text-xs font-semibold text-gray-600 mb-2">MODEL INFO</h4>
                        <div className="space-y-1 text-xs text-gray-600">
                          <div className="flex justify-between">
                            <span>Algorithm:</span>
                            <span className="font-medium">{modelInfo.model_type}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Fertilizer Types:</span>
                            <span className="font-medium">{modelInfo.n_classes}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Features Used:</span>
                            <span className="font-medium">{modelInfo.n_features}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <div className="text-center py-8">
                      <Droplet className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                      <p className="text-gray-500 text-sm mb-4">
                        Fill in all soil, crop, and environmental details to get ML-powered fertilizer recommendations
                      </p>
                      <div className="bg-blue-100 border border-blue-300 rounded-lg p-3 text-left text-xs text-blue-800 space-y-2">
                        <p className="font-semibold flex items-center">
                          <Sparkles className="w-4 h-4 mr-1" />
                          Quick Tip:
                        </p>
                        <p>Click the <strong>"Auto-Fill Weather"</strong> button to automatically populate Temperature, Humidity, and Rainfall with real-time data!</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// MapSelector Component for Fertilizer Module
const MapSelector = ({ onLocationSelect, onClose, loading }) => {
  const [map, setMap] = useState(null);
  const [selectedPos, setSelectedPos] = useState(null);

  useEffect(() => {
    // Dynamically import Leaflet to avoid SSR issues
    if (typeof window !== 'undefined') {
      import('leaflet').then((L) => {
        // Initialize map centered on India
        const mapInstance = L.map('fertilizer-map').setView([20.5937, 78.9629], 5);
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors',
          maxZoom: 18
        }).addTo(mapInstance);

        let marker = null;

        mapInstance.on('click', (e) => {
          const { lat, lng } = e.latlng;
          
          // Remove existing marker
          if (marker) {
            mapInstance.removeLayer(marker);
          }
          
          // Add new marker
          marker = L.marker([lat, lng]).addTo(mapInstance);
          setSelectedPos({ lat, lng });
        });

        setMap(mapInstance);

        // Cleanup
        return () => {
          if (mapInstance) {
            mapInstance.remove();
          }
        };
      });
    }
  }, []);

  const handleConfirm = () => {
    if (selectedPos) {
      onLocationSelect(selectedPos.lat, selectedPos.lng);
    } else {
      alert('Please select a location on the map');
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        <div className="p-3 md:p-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center">
            <MapPin className="w-4 h-4 md:w-5 md:h-5 mr-2 text-primary-600" />
            <h3 className="text-base md:text-lg font-semibold text-gray-800">Select Location</h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
            disabled={loading}
          >
            <X className="w-5 h-5 md:w-6 md:h-6" />
          </button>
        </div>
        
        <div className="p-3 md:p-4 overflow-y-auto flex-1">
          <div className="mb-3 p-2 md:p-3 bg-blue-50 border border-blue-200 rounded text-xs md:text-sm text-blue-800">
            <p className="font-medium mb-1">📍 Click on the map to select a location</p>
            <p className="text-xs">Fetches soil, weather, and location data automatically.</p>
          </div>
          
          <div 
            id="fertilizer-map" 
            style={{ height: 'min(400px, 50vh)', width: '100%' }}
            className="rounded-lg border border-gray-300"
          ></div>
          
          {selectedPos && (
            <div className="mt-2 md:mt-3 p-2 md:p-3 bg-blue-50 rounded-lg text-xs md:text-sm">
              <span className="font-medium text-gray-700">Selected:</span>{' '}
              <span className="text-gray-600">
                {selectedPos.lat.toFixed(4)}, {selectedPos.lng.toFixed(4)}
              </span>
            </div>
          )}
          
          {loading && (
            <div className="mt-2 md:mt-3 p-2 md:p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-xs md:text-sm">
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 mr-2"></div>
                <span className="text-yellow-800">Fetching data...</span>
              </div>
            </div>
          )}
        </div>
        
        <div className="p-3 md:p-4 border-t border-gray-200 flex gap-2 md:gap-3 flex-shrink-0">
          <button
            type="button"
            onClick={onClose}
            className="btn-secondary flex-1 text-sm md:text-base"
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed text-sm md:text-base"
            disabled={!selectedPos || loading}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="hidden md:inline">Fetching Data...</span>
                <span className="md:hidden">Loading...</span>
              </span>
            ) : (
              <>
                <span className="hidden md:inline">Confirm & Fetch Data</span>
                <span className="md:hidden">Confirm</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FertilizerRecommendation;

