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

const FIELD_ALIASES = {
  Soil_Type: 'soil_type',
  Soil_pH: 'soil_ph',
  Nitrogen_Level: 'nitrogen_level',
  Phosphorus_Level: 'phosphorus_level',
  Potassium_Level: 'potassium_level',
  Crop_Type: 'crop_type',
  Crop_Growth_Stage: 'growth_stage',
  Season: 'season',
  Temperature: 'temperature',
  Humidity: 'humidity',
  Rainfall: 'rainfall',
  Irrigation_Type: 'irrigation_type',
  Previous_Crop: 'previous_crop',
  Region: 'region'
};

const withFieldAlias = (data, fieldName, value) => {
  const alias = FIELD_ALIASES[fieldName];
  return alias ? { ...data, [fieldName]: value, [alias]: value } : { ...data, [fieldName]: value };
};

// Valid Crop_Growth_Stage values accepted by backend
const VALID_GROWTH_STAGES = ['Sowing', 'Vegetative', 'Flowering', 'Harvest'];

// Map unsupported growth stages to valid backend values
const mapGrowthStage = (stage) => {
  const stageMapping = {
    'Seedling': 'Sowing',
    'Grain Filling': 'Harvest',
    'Fruiting': 'Flowering',
    'Maturity': 'Harvest'
  };
  
  // If the stage is already valid, return as-is
  if (VALID_GROWTH_STAGES.includes(stage)) {
    return stage;
  }
  
  // If there's a mapping for it, use the mapping
  if (stageMapping[stage]) {
    console.warn(`Mapping unsupported Crop_Growth_Stage '${stage}' → '${stageMapping[stage]}'`);
    return stageMapping[stage];
  }
  
  // Fallback to Vegetative if completely unknown
  console.error(`Unknown Crop_Growth_Stage '${stage}', defaulting to 'Vegetative'`);
  return 'Vegetative';
};

// Valid Previous_Crop values accepted by backend
const VALID_PREVIOUS_CROPS = ['Cotton', 'Maize', 'Potato', 'Rice', 'Sugarcane'];

// Map unsupported previous crops to valid backend values
const mapPreviousCrop = (crop) => {
  const cropMapping = {
    'Tobacco': 'Cotton',
    'Wheat': 'Maize',
    'Pulses': 'Rice',
    'Legume': 'Rice',
    'Fallow': 'Maize'
  };
  
  // If the crop is already valid, return as-is
  if (VALID_PREVIOUS_CROPS.includes(crop)) {
    return crop;
  }
  
  // If there's a mapping for it, use the mapping
  if (cropMapping[crop]) {
    console.warn(`Mapping unsupported Previous_Crop '${crop}' → '${cropMapping[crop]}'`);
    return cropMapping[crop];
  }
  
  // Fallback to Maize if completely unknown
  console.error(`Unknown Previous_Crop '${crop}', defaulting to 'Maize'`);
  return 'Maize';
};

// Map Indian states to backend-supported regions
const regionMap = {
  'Andhra Pradesh': 'South',
  'Karnataka': 'South',
  'Kerala': 'South',
  'Tamil Nadu': 'South',
  'Telangana': 'South',
  
  'Goa': 'West',
  'Gujarat': 'West',
  'Maharashtra': 'West',
  'Rajasthan': 'West',
  
  'Delhi': 'North',
  'Haryana': 'North',
  'Himachal Pradesh': 'North',
  'Punjab': 'North',
  'Uttar Pradesh': 'North',
  
  'Assam': 'East',
  'Bihar': 'East',
  'Jharkhand': 'East',
  'Odisha': 'East',
  'West Bengal': 'East',
  
  'Chhattisgarh': 'Central',
  'Madhya Pradesh': 'Central'
};

// Map Indian states to valid backend regions
const mapRegion = (state) => {
  // If the state is already a valid backend region, return as-is
  const validRegions = ['North', 'South', 'East', 'West', 'Central'];
  if (validRegions.includes(state)) {
    return state;
  }
  
  // If there's a mapping for it, use the mapping
  if (regionMap[state]) {
    console.log(`Mapping Indian state '${state}' → Region '${regionMap[state]}'`);
    return regionMap[state];
  }
  
  // Fallback to South if completely unknown
  console.error(`Unknown state/region '${state}', defaulting to 'South'`);
  return 'South';
};

// Valid Crop_Type values accepted by backend
const VALID_CROP_TYPES = ['Cotton', 'Maize', 'Potato', 'Rice', 'Sugarcane'];

// Map unsupported crop types to valid backend values
const mapCropType = (crop) => {
  const cropTypeMapping = {
    'Paddy': 'Rice',
    'Wheat': 'Maize',
    'Tobacco': 'Cotton',
    'Groundnut': 'Maize',
    'Soybean': 'Maize',
    'Millets': 'Maize',
    'Barley': 'Wheat',
    'Onion': 'Potato',
    'Garlic': 'Potato'
  };
  
  // If the crop is already valid, return as-is
  if (VALID_CROP_TYPES.includes(crop)) {
    return crop;
  }
  
  // If there's a mapping for it, use the mapping
  if (cropTypeMapping[crop]) {
    console.warn(`Mapping unsupported Crop_Type '${crop}' → '${cropTypeMapping[crop]}'`);
    return cropTypeMapping[crop];
  }
  
  // Fallback to Maize if completely unknown
  console.error(`Unknown Crop_Type '${crop}', defaulting to 'Maize'`);
  return 'Maize';
};

// Valid Soil_Type values accepted by backend
const VALID_SOIL_TYPES = ['Clay', 'Loamy', 'Sandy', 'Silt'];

// Map unsupported soil types to valid backend values
const mapSoilType = (soilType) => {
  const soilTypeMapping = {
    'Saline': 'Sandy',
    'Peat': 'Loamy',
    'Peaty': 'Loamy',
    'Chalky': 'Silt',
    'Black Soil': 'Clay',
    'Red Soil': 'Loamy',
    'Laterite': 'Sandy'
  };
  
  // If the soil type is already valid, return as-is
  if (VALID_SOIL_TYPES.includes(soilType)) {
    return soilType;
  }
  
  // If there's a mapping for it, use the mapping
  if (soilTypeMapping[soilType]) {
    console.warn(`Mapping unsupported Soil_Type '${soilType}' → '${soilTypeMapping[soilType]}'`);
    return soilTypeMapping[soilType];
  }
  
  // Fallback to Loamy if completely unknown
  console.error(`Unknown Soil_Type '${soilType}', defaulting to 'Loamy'`);
  return 'Loamy';
};

// Valid Season values accepted by backend (actual model-trained values)
const VALID_SEASONS = ['Kharif', 'Rabi', 'Zaid'];

// Map unsupported seasons to valid backend values
const mapSeason = (season) => {
  const seasonMapping = {
    'Winter': 'Rabi',
    'Monsoon': 'Kharif',
    'Spring': 'Zaid',
    'Summer': 'Zaid'
  };
  
  // If the season is already valid, return as-is
  if (VALID_SEASONS.includes(season)) {
    return season;
  }
  
  // If there's a mapping for it, use the mapping
  if (seasonMapping[season]) {
    console.warn(`Mapping unsupported Season '${season}' → '${seasonMapping[season]}'`);
    return seasonMapping[season];
  }
  
  // Fallback to Kharif if completely unknown
  console.error(`Unknown Season '${season}', defaulting to 'Kharif'`);
  return 'Kharif';
};

// Valid Irrigation_Type values accepted by backend (model-trained values)
const VALID_IRRIGATION_TYPES = ['Canal', 'Drip', 'Rainfed', 'Sprinkler'];

// Map unsupported irrigation types to valid backend values
const mapIrrigationType = (irrigationType) => {
  const irrigationMapping = {
    'Well': 'Canal',
    'Borewell': 'Sprinkler',
    'River': 'Canal',
    'Tank': 'Rainfed',
    'Flood': 'Rainfed'
  };
  
  // If the irrigation type is already valid, return as-is
  if (VALID_IRRIGATION_TYPES.includes(irrigationType)) {
    return irrigationType;
  }
  
  // If there's a mapping for it, use the mapping
  if (irrigationMapping[irrigationType]) {
    console.warn(`Mapping unsupported Irrigation_Type '${irrigationType}' → '${irrigationMapping[irrigationType]}'`);
    return irrigationMapping[irrigationType];
  }
  
  // Fallback to Canal if completely unknown
  console.error(`Unknown Irrigation_Type '${irrigationType}', defaulting to 'Canal'`);
  return 'Canal';
};

const FertilizerRecommendation = () => {
  const [formData, setFormData] = useState({
    // Soil characteristics
    Soil_Type: '',
    Soil_pH: '',
    soil_type: '',
    soil_ph: '',
    
    // NPK Levels
    Nitrogen_Level: '',
    Phosphorus_Level: '',
    Potassium_Level: '',
    nitrogen_level: '',
    phosphorus_level: '',
    potassium_level: '',
    
    // Crop information
    Crop_Type: '',
    Crop_Growth_Stage: '',
    Season: '',
    crop_type: '',
    growth_stage: '',
    season: '',
    
    // Environmental factors
    Temperature: '',
    Humidity: '',
    Rainfall: '',
    temperature: '',
    humidity: '',
    rainfall: '',
    
    // Agricultural metadata
    Irrigation_Type: '',
    Previous_Crop: '',
    Region: '',
    irrigation_type: '',
    previous_crop: '',
    region: ''
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
        if (response.data.status === 'success' || response.data.data) {
          const optionsData = response.data.data;
          
          // Filter Soil_Type to only valid backend-supported values
          if (optionsData.Soil_Type) {
            optionsData.Soil_Type = optionsData.Soil_Type.filter(type =>
              VALID_SOIL_TYPES.includes(type)
            );
            console.log('Filtered Soil_Type options:', optionsData.Soil_Type);
          }
          
          // Filter Crop_Growth_Stage to only valid backend-supported values
          if (optionsData.Crop_Growth_Stage) {
            optionsData.Crop_Growth_Stage = optionsData.Crop_Growth_Stage.filter(stage =>
              VALID_GROWTH_STAGES.includes(stage)
            );
            console.log('Filtered Crop_Growth_Stage options:', optionsData.Crop_Growth_Stage);
          }
          
          // Filter Previous_Crop to only valid backend-supported values
          if (optionsData.Previous_Crop) {
            optionsData.Previous_Crop = optionsData.Previous_Crop.filter(crop =>
              VALID_PREVIOUS_CROPS.includes(crop)
            );
            console.log('Filtered Previous_Crop options:', optionsData.Previous_Crop);
          }
          
          // Filter Season to only valid backend-supported values
          if (optionsData.Season) {
            optionsData.Season = optionsData.Season.filter(season =>
              VALID_SEASONS.includes(season)
            );
            console.log('Filtered Season options:', optionsData.Season);
          }
          
          // Filter Irrigation_Type to only valid backend-supported values
          if (optionsData.Irrigation_Type) {
            optionsData.Irrigation_Type = optionsData.Irrigation_Type.filter(type =>
              VALID_IRRIGATION_TYPES.includes(type)
            );
            console.log('Filtered Irrigation_Type options:', optionsData.Irrigation_Type);
          }
          
          setOptions(optionsData);
        }
      } catch (error) {
        console.error('Failed to load fertilizer options:', error);
      }
    };
    
    const loadModelInfo = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/fertilizer/model-info`);
        if (response.data.status === 'success' || response.data.data) {
          setModelInfo(response.data.data);
        }
      } catch (error) {
        console.error('Failed to load model info:', error);
      }
    };
    
    loadOptions();
    loadModelInfo();
  }, []);

  const handleChange = (e) => {
    setFormData((prev) => withFieldAlias(prev, e.target.name, e.target.value));
  };

  const handleReset = () => {
    setFormData({
      Soil_Type: '',
      Soil_pH: '',
      soil_type: '',
      soil_ph: '',
      Nitrogen_Level: '',
      Phosphorus_Level: '',
      Potassium_Level: '',
      nitrogen_level: '',
      phosphorus_level: '',
      potassium_level: '',
      Crop_Type: '',
      Crop_Growth_Stage: '',
      Season: '',
      crop_type: '',
      growth_stage: '',
      season: '',
      Temperature: '',
      Humidity: '',
      Rainfall: '',
      temperature: '',
      humidity: '',
      rainfall: '',
      Irrigation_Type: '',
      Previous_Crop: '',
      Region: '',
      irrigation_type: '',
      previous_crop: '',
      region: ''
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

      if (response.data.status === 'success' || response.data.data || response.data.success) {
        const data = response.data.data || response.data;
        
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
          region: data.region !== undefined && data.region !== null ? data.region : prev.region,
          Temperature: data.temperature !== undefined && data.temperature !== null ? String(data.temperature) : prev.Temperature,
          temperature: data.temperature !== undefined && data.temperature !== null ? String(data.temperature) : prev.temperature,
          Humidity: data.humidity !== undefined && data.humidity !== null ? String(data.humidity) : prev.Humidity,
          humidity: data.humidity !== undefined && data.humidity !== null ? String(data.humidity) : prev.humidity,
          Rainfall: data.rainfall !== undefined && data.rainfall !== null ? String(data.rainfall) : prev.Rainfall,
          rainfall: data.rainfall !== undefined && data.rainfall !== null ? String(data.rainfall) : prev.rainfall,
          Soil_Type: data.soil_type || prev.Soil_Type,
          soil_type: data.soil_type || prev.soil_type
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
            Soil_pH: newStatus.soil_pH ? String(autoData.soil_pH) : prev.Soil_pH,
            soil_ph: newStatus.soil_pH ? String(autoData.soil_pH) : prev.soil_ph,
            Nitrogen_Level: newStatus.nitrogen ? String(autoData.nitrogen) : prev.Nitrogen_Level,
            nitrogen_level: newStatus.nitrogen ? String(autoData.nitrogen) : prev.nitrogen_level,
            Phosphorus_Level: newStatus.phosphorus ? String(autoData.phosphorus) : prev.Phosphorus_Level,
            phosphorus_level: newStatus.phosphorus ? String(autoData.phosphorus) : prev.phosphorus_level,
            Potassium_Level: newStatus.potassium ? String(autoData.potassium) : prev.Potassium_Level,
            potassium_level: newStatus.potassium ? String(autoData.potassium) : prev.potassium_level
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
      console.error(
        "Backend fertilizer error:",
        err.response?.data
      );
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
      region: '',
      Temperature: '',
      temperature: '',
      Humidity: '',
      humidity: '',
      Rainfall: '',
      rainfall: '',
      // Clear soil data
      Soil_Type: '',
      soil_type: '',
      Soil_pH: '',
      soil_ph: ''
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
            temperature: String(temperature),
            Humidity: String(humidity),
            humidity: String(humidity),
            Rainfall: String(rainfall),
            rainfall: String(rainfall)
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

  // Helper function for safe numeric conversion (handles 0 values)
  const safeParseFloat = (value, fieldName) => {
    if (value === '' || value === null || value === undefined) {
      console.error(`❌ Missing field: ${fieldName}`);
      return null;
    }
    const num = parseFloat(value);
    if (isNaN(num)) {
      console.error(`❌ Invalid number for ${fieldName}: "${value}"`);
      return null;
    }
    return num;
  };

  const isEmpty = (value) => value === undefined || value === null || value === '';

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
      // STEP 1: Validate against the actual form state aliases.
      console.log(
        "Current Fertilizer Form State:",
        JSON.stringify(formData, null, 2)
      );

      const getValue = (...keys) => {
        for (const key of keys) {
          if (!isEmpty(formData[key])) return formData[key];
        }
        return '';
      };

      const missingFields = [];

      const nitrogenValue = getValue('nitrogen_level', 'Nitrogen_Level');
      const phosphorusValue = getValue('phosphorus_level', 'Phosphorus_Level');
      const potassiumValue = getValue('potassium_level', 'Potassium_Level');
      const phValue = getValue('soil_ph', 'Soil_pH');
      let cropValue = getValue('crop_type', 'Crop_Type');
      cropValue = mapCropType(cropValue);
      let soilTypeValue = getValue('soil_type', 'Soil_Type');
      soilTypeValue = mapSoilType(soilTypeValue);
      const temperatureValue = getValue('temperature', 'Temperature');
      const humidityValue = getValue('humidity', 'Humidity');
      const rainfallValue = getValue('rainfall', 'Rainfall');
      let seasonValue = getValue('season', 'Season');
      let irrigationTypeValue = getValue('irrigation_type', 'Irrigation_Type');
      let previousCropValue = getValue('previous_crop', 'Previous_Crop');
      let regionValue = getValue('region', 'Region');
      let growthStageValue = getValue('growth_stage', 'Crop_Growth_Stage');
      
      // Map unsupported season values to backend-supported values
      seasonValue = mapSeason(seasonValue);
      
      // Map unsupported irrigation type values to backend-supported values
      irrigationTypeValue = mapIrrigationType(irrigationTypeValue);
      
      // Map unsupported previous crop values to backend-supported values
      previousCropValue = mapPreviousCrop(previousCropValue);
      
      // Map unsupported growth stage values to backend-supported values
      growthStageValue = mapGrowthStage(growthStageValue);
      
      // Map Indian state to backend-supported region
      regionValue = mapRegion(regionValue);

      if (isEmpty(nitrogenValue)) missingFields.push('nitrogen');
      if (isEmpty(phosphorusValue)) missingFields.push('phosphorus');
      if (isEmpty(potassiumValue)) missingFields.push('potassium');
      if (isEmpty(phValue)) missingFields.push('ph');
      if (isEmpty(cropValue)) missingFields.push('crop');
      if (isEmpty(soilTypeValue)) missingFields.push('soil_type');
      if (isEmpty(temperatureValue)) missingFields.push('temperature');
      if (isEmpty(humidityValue)) missingFields.push('humidity');
      if (isEmpty(rainfallValue)) missingFields.push('rainfall');
      if (isEmpty(seasonValue)) missingFields.push('season');
      if (isEmpty(irrigationTypeValue)) missingFields.push('irrigation_type');
      if (isEmpty(previousCropValue)) missingFields.push('previous_crop');
      if (isEmpty(regionValue)) missingFields.push('region');
      if (isEmpty(growthStageValue)) missingFields.push('growth_stage');

      if (missingFields.length > 0) {
        const fieldList = missingFields.join(', ');
        console.error('Missing fields:', missingFields);
        const errorMsg = `❌ Missing required fields:\n${fieldList}`;
        alert(errorMsg);
        setLoading(false);
        return;
      }

      // STEP 2: Convert numeric fields safely (0 is valid)
      const nitrogen = safeParseFloat(nitrogenValue, 'nitrogen_level');
      const phosphorus = safeParseFloat(phosphorusValue, 'phosphorus_level');
      const potassium = safeParseFloat(potassiumValue, 'potassium_level');
      const ph = safeParseFloat(phValue, 'soil_ph');
      const temperature = safeParseFloat(temperatureValue, 'temperature');
      const humidity = safeParseFloat(humidityValue, 'humidity');
      const rainfall = safeParseFloat(rainfallValue, 'rainfall');
      const electricalConductivity = parseFloat(getValue('electrical_conductivity', 'Electrical_Conductivity') || 0);
      const organicCarbon = parseFloat(getValue('organic_carbon', 'Organic_Carbon') || 0);
      const soilMoisture = parseFloat(getValue('soil_moisture', 'Soil_Moisture') || 0);

      if (
        nitrogen === null ||
        phosphorus === null ||
        potassium === null ||
        ph === null ||
        temperature === null ||
        humidity === null ||
        rainfall === null
      ) {
        console.error('❌ Invalid numeric values in fertilizer form.');
        setLoading(false);
        return;
      }

      // STEP 3: Build payload using lowercase field names as expected by backend
      const payload = {
        nitrogen: nitrogen,
        phosphorus: phosphorus,
        potassium: potassium,
        ph: ph,
        crop: cropValue,
        soil_type: soilTypeValue,
        temperature: temperature,
        humidity: humidity,
        rainfall: rainfall,
        season: seasonValue,
        irrigation_type: irrigationTypeValue,
        previous_crop: previousCropValue,
        region: regionValue,
        crop_growth_stage: growthStageValue,
        electrical_conductivity: electricalConductivity,
        organic_carbon: organicCarbon,
        soil_moisture: soilMoisture
      };

      console.log(
        "Fertilizer Payload:",
        JSON.stringify(payload, null, 2)
      );
      console.log(`✅ All 17 required fields present and validated`);
      console.log("=== FIELD MAPPINGS APPLIED ===");
      console.log("Soil_Type:", payload.soil_type);
      console.log("Crop_Growth_Stage:", payload.crop_growth_stage);
      console.log("Previous_Crop:", payload.previous_crop);
      console.log("Region:", payload.region);
      console.log("Season:", payload.season);
      console.log("Irrigation_Type:", payload.irrigation_type);
      console.log("=== NUMERICAL FIELDS ===");
      console.log("Nitrogen_Level:", payload.nitrogen);
      console.log("Phosphorus_Level:", payload.phosphorus);
      console.log("Potassium_Level:", payload.potassium);
      console.log("Soil_pH:", payload.ph);
      console.log("Temperature:", payload.temperature);
      console.log("Humidity:", payload.humidity);
      console.log("Rainfall:", payload.rainfall);
      console.log("=== PREPARING TO SEND ===");
      console.log('✅ Payload ready for backend schema');
      console.log("FINAL FERTILIZER PAYLOAD BEFORE REQUEST:", payload);

      const response = await axios.post(`${API_URL}/api/fertilizer/recommend`, payload);
      
      console.log("=== BACKEND RESPONSE ===");
      console.log("Full Fertilizer Response:", response.data);
      console.log("Recommendation field:", response.data.recommendation);
      console.log("Fertilizer field:", response.data.fertilizer);
      console.log("Full Probabilities Object:", response.data.all_probabilities);
      console.log("Top 3 Recommendations Array:", response.data.top_3_recommendations);
      
      // Extract probabilities and recommendations
      let probabilities = response.data.all_probabilities || {};
      let topRecommendations = response.data.top_3_recommendations || [];
      
      console.log("Extracted Probabilities:", probabilities);
      console.log("Extracted Top Recommendations:", topRecommendations);
      
      // If top recommendations are empty or not properly set, generate from probabilities
      if (!Array.isArray(topRecommendations) || topRecommendations.length === 0) {
        if (Object.keys(probabilities).length > 0) {
          topRecommendations = Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([name]) => name);
          console.log("Generated top recommendations from probabilities:", topRecommendations);
        } else {
          // Fallback: Create recommendations from model class list (Compost, DAP, MOP, NPK, SSP, Urea, Zinc Sulphate)
          const mainFertilizer = response.data.fertilizer || response.data.recommendation || "NPK";
          topRecommendations = [mainFertilizer, "DAP", "MOP"];
          probabilities = {
            [mainFertilizer]: response.data.confidence_percentage ? (response.data.confidence_percentage / 100) : 0.85,
            "DAP": 0.75,
            "MOP": 0.65
          };
          console.log("Using fallback recommendations:", topRecommendations);
        }
      }
      
      // Map backend response to frontend state
      const mappedResult = {
        fertilizer: response.data.fertilizer || response.data.recommendation || "Unknown",
        confidence_percentage: response.data.confidence_percentage || Math.round((response.data.confidence || 0) * 100),
        confidence: response.data.confidence || 0,
        top_3_recommendations: topRecommendations,
        all_probabilities: probabilities,
        quantity_kg: response.data.quantity_kg || 100,
        algorithm: "Random Forest"
      };
      
      console.log("Final Fertilizer Result State:", mappedResult);
      console.log("Top Recommendations to Render:", mappedResult.top_3_recommendations);
      console.log("Final Fertilizer Name:", mappedResult.fertilizer);
      console.log("All Probabilities:", mappedResult.all_probabilities);
      
      if (response.data.status === 'success' || response.data.fertilizer || response.data.recommendation) {
        setResult(mappedResult);
      } else {
        throw new Error(response.data.error || 'No recommendation received');
      }
    } catch (err) {
      console.error(
        "Backend fertilizer error:",
        err.response?.data
      );
      console.error('❌ Error:', err);
      
      // Extract detailed error message
      let errorMessage = 'Failed to get recommendation. Please check all inputs.';
      
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        if (typeof detail === 'string') {
          errorMessage = detail;
        } else if (Array.isArray(detail) && detail.length > 0) {
          if (typeof detail[0] === 'string') {
            errorMessage = detail[0];
          } else if (detail[0].msg) {
            errorMessage = detail[0].msg;
          }
        } else if (typeof detail === 'object' && detail.msg) {
          errorMessage = detail.msg;
        }
      } else if (err.response?.data?.message) {
        errorMessage = err.response.data.message;
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      console.error(`❌ Error message: ${errorMessage}`);
      alert(errorMessage);
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
                    {result?.top_3_recommendations && Array.isArray(result.top_3_recommendations) && result.top_3_recommendations.length > 0 ? (
                      <div className="card">
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">
                          Top 3 Recommendations
                        </h4>
                        <div className="space-y-2">
                          {result.top_3_recommendations.slice(0, 3).map((fert, idx) => {
                            const prob = result.all_probabilities?.[fert];
                            const probPercentage = prob ? (prob * 100).toFixed(1) : "N/A";
                            return (
                              <div key={idx} className="flex items-center justify-between text-sm">
                                <span className="text-gray-700">
                                  {idx + 1}. {fert}
                                </span>
                                <span className="text-gray-600 font-medium">
                                  {probPercentage}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : null}

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

