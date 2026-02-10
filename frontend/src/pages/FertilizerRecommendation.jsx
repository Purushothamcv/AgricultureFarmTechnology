import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import LoadingSpinner from '../components/LoadingSpinner';
import { Droplet, Sparkles, MapPin, X } from 'lucide-react';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const FertilizerRecommendation = () => {
  const [formData, setFormData] = useState({
    // Soil characteristics
    Soil_Type: '',
    Soil_pH: '',
    Soil_Moisture: '',
    Organic_Carbon: '',
    Electrical_Conductivity: '',
    
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
      Soil_Moisture: '',
      Organic_Carbon: '',
      Electrical_Conductivity: '',
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
  };

  const handleMapLocationSelect = async (lat, lng) => {
    setMapLoading(true);
    try {
      const response = await axios.post(`${API_URL}/api/fertilizer/location-data`, {
        latitude: lat,
        longitude: lng
      });

      if (response.data.success) {
        const data = response.data;
        setLocationData(data);

        // Autofill form with location and weather data
        setFormData(prev => ({
          ...prev,
          Region: data.region || prev.Region,
          Temperature: data.temperature?.toFixed(1) || prev.Temperature,
          Humidity: data.humidity?.toFixed(0) || prev.Humidity,
          Rainfall: data.rainfall?.toFixed(1) || prev.Rainfall
        }));

        setShowMap(false);
        alert(`Location detected!\nState: ${data.state || 'N/A'}\nDistrict: ${data.district || 'N/A'}\nRegion: ${data.region || 'N/A'}`);
      }
    } catch (err) {
      console.error('Error fetching location data:', err);
      alert('Failed to fetch location data. Please try again or enter manually.');
    }
    setMapLoading(false);
  };

  const handleResetLocation = () => {
    setLocationData(null);
    setFormData(prev => ({
      ...prev,
      Region: '',
      Temperature: '',
      Humidity: '',
      Rainfall: ''
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/api/fertilizer/recommend`, formData);
      
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
                    
                    {locationData ? (
                      <div className="bg-white p-3 rounded border border-blue-300">
                        <div className="grid grid-cols-2 gap-2 text-sm">
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
                            <span className="text-gray-600">Weather:</span>
                            <span className="font-medium ml-2">
                              {locationData.temperature ? `${locationData.temperature.toFixed(1)}°C, ${locationData.humidity?.toFixed(0)}%` : 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setShowMap(true)}
                        className="btn-secondary w-full flex items-center justify-center"
                      >
                        <MapPin className="w-4 h-4 mr-2" />
                        Select Location from Map
                      </button>
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
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Soil Type <span className="text-red-500">*</span>
                        </label>
                        <select
                          name="Soil_Type"
                          value={formData.Soil_Type}
                          onChange={handleChange}
                          required
                          className="input-field"
                        >
                          <option value="">Select soil type</option>
                          {options.Soil_Type?.map(type => (
                            <option key={type} value={type}>{type}</option>
                          ))}
                        </select>
                      </div>
                      
                      <InputField
                        label="Soil pH"
                        name="Soil_pH"
                        type="number"
                        value={formData.Soil_pH}
                        onChange={handleChange}
                        placeholder="4.0 - 9.0"
                        required
                        min="4"
                        max="9"
                        step="0.1"
                      />
                      
                      <InputField
                        label="Soil Moisture (%)"
                        name="Soil_Moisture"
                        type="number"
                        value={formData.Soil_Moisture}
                        onChange={handleChange}
                        placeholder="0 - 100"
                        required
                        min="0"
                        max="100"
                        step="0.1"
                      />
                      
                      <InputField
                        label="Organic Carbon (%)"
                        name="Organic_Carbon"
                        type="number"
                        value={formData.Organic_Carbon}
                        onChange={handleChange}
                        placeholder="0 - 5"
                        required
                        min="0"
                        max="5"
                        step="0.01"
                      />
                      
                      <InputField
                        label="Electrical Conductivity (dS/m)"
                        name="Electrical_Conductivity"
                        type="number"
                        value={formData.Electrical_Conductivity}
                        onChange={handleChange}
                        placeholder="0 - 4"
                        required
                        min="0"
                        max="4"
                        step="0.01"
                      />
                    </div>
                  </div>

                  {/* NPK Levels */}
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-green-600 mr-2"></div>
                      NPK Nutrient Levels (mg/kg)
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <InputField
                        label="Nitrogen Level"
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
                      
                      <InputField
                        label="Phosphorus Level"
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
                      
                      <InputField
                        label="Potassium Level"
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
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                      <div className="w-2 h-6 bg-blue-600 mr-2"></div>
                      Environmental Conditions
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <InputField
                        label="Temperature (°C)"
                        name="Temperature"
                        type="number"
                        value={formData.Temperature}
                        onChange={handleChange}
                        placeholder="0 - 50"
                        required
                        min="0"
                        max="50"
                        step="0.1"
                      />
                      
                      <InputField
                        label="Humidity (%)"
                        name="Humidity"
                        type="number"
                        value={formData.Humidity}
                        onChange={handleChange}
                        placeholder="0 - 100"
                        required
                        min="0"
                        max="100"
                        step="0.1"
                      />
                      
                      <InputField
                        label="Rainfall (mm)"
                        name="Rainfall"
                        type="number"
                        value={formData.Rainfall}
                        onChange={handleChange}
                        placeholder="0 - 500"
                        required
                        min="0"
                        max="1000"
                        step="0.1"
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
                        </label>
                        <select
                          name="Region"
                          value={formData.Region}
                          onChange={handleChange}
                          required
                          className="input-field"
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
                      <p className="text-gray-500 text-sm">
                        Fill in all soil, crop, and environmental details to get ML-powered fertilizer recommendations
                      </p>
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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl mx-4">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center">
            <MapPin className="w-5 h-5 mr-2 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-800">Select Location on Map</h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
            disabled={loading}
          >
            <X className="w-6 h-6" />
          </button>
        </div>
        
        <div className="p-4">
          <div 
            id="fertilizer-map" 
            style={{ height: '500px', width: '100%' }}
            className="rounded-lg border border-gray-300"
          ></div>
          
          {selectedPos && (
            <div className="mt-3 p-3 bg-blue-50 rounded-lg text-sm">
              <span className="font-medium text-gray-700">Selected Location:</span>{' '}
              <span className="text-gray-600">
                Latitude: {selectedPos.lat.toFixed(5)}, Longitude: {selectedPos.lng.toFixed(5)}
              </span>
            </div>
          )}
        </div>
        
        <div className="p-4 border-t border-gray-200 flex gap-3">
          <button
            type="button"
            onClick={onClose}
            className="btn-secondary flex-1"
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="btn-primary flex-1"
            disabled={!selectedPos || loading}
          >
            {loading ? 'Fetching Location Data...' : 'Confirm Location'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FertilizerRecommendation;

