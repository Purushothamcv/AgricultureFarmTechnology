import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService } from '../services/services';
import { TrendingUp, MapPin, Calendar, Sprout, Map } from 'lucide-react';

const YieldPrediction = () => {
  const [formData, setFormData] = useState({
    state: '',
    district: '',
    crop: '',
    year: new Date().getFullYear(),
    season: '',
    area: ''
  });
  const [options, setOptions] = useState({
    states: [],
    districts: [],
    crops: [],
    seasons: []
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [loadingDistricts, setLoadingDistricts] = useState(false);
  const [showMap, setShowMap] = useState(false);

  // Load available options on component mount
  useEffect(() => {
    loadOptions();
  }, []);

  // Load districts when state changes
  useEffect(() => {
    if (formData.state) {
      loadDistrictsByState(formData.state);
    } else {
      // Reset districts when no state is selected
      setOptions(prev => ({ ...prev, districts: [] }));
    }
  }, [formData.state]);

  const loadOptions = async () => {
    try {
      setLoadingOptions(true);
      const response = await fetch('http://localhost:8000/yield/states');
      const data = await response.json();
      
      if (data.success) {
        setOptions(prev => ({
          ...prev,
          states: data.states || []
        }));
        console.log('✅ Loaded states:', data.states?.length);
      }

      // Load crops and seasons separately
      const optionsResponse = await fetch('http://localhost:8000/api/yield/options');
      const optionsData = await optionsResponse.json();
      
      if (optionsData.success) {
        setOptions(prev => ({
          ...prev,
          crops: optionsData.crops || [],
          seasons: optionsData.seasons || []
        }));
        console.log('✅ Loaded crops and seasons');
      }
    } catch (err) {
      console.error('Error loading options:', err);
    } finally {
      setLoadingOptions(false);
    }
  };

  const loadDistrictsByState = async (state) => {
    try {
      setLoadingDistricts(true);
      const response = await fetch(`http://localhost:8000/yield/districts/${encodeURIComponent(state)}`);
      const data = await response.json();
      
      if (data.success) {
        setOptions(prev => ({
          ...prev,
          districts: data.districts || []
        }));
        console.log(`✅ Loaded ${data.districts?.length} districts for ${state}`);
      }
    } catch (err) {
      console.error('Error loading districts:', err);
    } finally {
      setLoadingDistricts(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // If state changes, clear district selection
    if (name === 'state') {
      setFormData({
        ...formData,
        state: value,
        district: '' // Clear district when state changes
      });
    } else {
      setFormData({
        ...formData,
        [name]: value
      });
    }
  };

  const handleMapLocationSelect = async (lat, lng) => {
    try {
      // Reverse geocoding using Nominatim (OpenStreetMap)
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&addressdetails=1`
      );
      const data = await response.json();
      
      if (data.address) {
        const state = data.address.state || '';
        const district = data.address.state_district || data.address.county || '';
        
        console.log('📍 Location from map:', { state, district, lat, lng });
        
        // Update form with map location
        setFormData(prev => ({
          ...prev,
          state: state,
          district: district
        }));
        
        setShowMap(false);
        alert(`Location selected:\nState: ${state}\nDistrict: ${district}`);
      }
    } catch (err) {
      console.error('Error in reverse geocoding:', err);
      alert('Could not determine location. Please select manually.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        state: formData.state,
        district: formData.district,
        crop: formData.crop,
        year: parseInt(formData.year),
        season: formData.season,
        area: parseFloat(formData.area)
      };

      console.log('📤 Sending yield prediction request:', payload);
      
      const response = await fetch('http://localhost:8000/predict-yield', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log('📥 Received yield prediction:', data);

      if (data.success) {
        setResult(data);
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Error:', err);
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 30 }, (_, i) => currentYear - 10 + i);

  // Simple Map Selector Component
  const MapSelector = ({ onLocationSelect }) => {
    const [selectedLocation, setSelectedLocation] = useState(null);

    useEffect(() => {
      // Dynamically load Leaflet CSS
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);

      // Dynamically load Leaflet JS
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      script.onload = initMap;
      document.body.appendChild(script);

      return () => {
        document.head.removeChild(link);
        document.body.removeChild(script);
      };
    }, []);

    const initMap = () => {
      if (typeof window.L === 'undefined') return;

      const map = window.L.map('yield-map').setView([20.5937, 78.9629], 5); // Center of India

      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(map);

      let marker = null;

      map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        
        if (marker) {
          map.removeLayer(marker);
        }

        marker = window.L.marker([lat, lng]).addTo(map);
        setSelectedLocation({ lat, lng });
      });
    };

    return (
      <div>
        <div id="yield-map" style={{ height: '300px', width: '100%', borderRadius: '8px' }}></div>
        {selectedLocation && (
          <div className="mt-3 flex items-center justify-between">
            <p className="text-sm text-gray-700">
              Selected: {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
            </p>
            <button
              type="button"
              onClick={() => onLocationSelect(selectedLocation.lat, selectedLocation.lng)}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm"
            >
              Use This Location
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <TrendingUp className="w-8 h-8 mr-3 text-primary-600" />
              Yield Prediction
            </h1>
            <p className="text-gray-600">
              Predict crop yield based on historical agricultural data
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                {loadingOptions ? (
                  <LoadingSpinner text="Loading options..." />
                ) : (
                  <form onSubmit={handleSubmit}>
                    {/* Location Section */}
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                          <MapPin className="w-5 h-5 mr-2 text-primary-600" />
                          Location
                        </h3>
                        <button
                          type="button"
                          onClick={() => setShowMap(!showMap)}
                          className="text-sm px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-md flex items-center gap-1 transition"
                        >
                          <Map className="w-4 h-4" />
                          {showMap ? 'Hide Map' : 'Select from Map'}
                        </button>
                      </div>

                      {showMap && (
                        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <MapSelector onLocationSelect={handleMapLocationSelect} />
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            State <span className="text-red-500">*</span>
                          </label>
                          <select
                            name="state"
                            value={formData.state}
                            onChange={handleChange}
                            required
                            className="input-field"
                          >
                            <option value="">Select State</option>
                            {options.states.map(state => (
                              <option key={state} value={state}>{state}</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            District <span className="text-red-500">*</span>
                          </label>
                          <select
                            name="district"
                            value={formData.district}
                            onChange={handleChange}
                            required
                            disabled={!formData.state || loadingDistricts}
                            className="input-field"
                          >
                            <option value="">
                              {!formData.state 
                                ? 'Select a state first' 
                                : loadingDistricts 
                                ? 'Loading districts...' 
                                : 'Select District'}
                            </option>
                            {options.districts.map(district => (
                              <option key={district} value={district}>{district}</option>
                            ))}
                          </select>
                          {formData.state && options.districts.length === 0 && !loadingDistricts && (
                            <p className="text-xs text-amber-600 mt-1">
                              No districts found for selected state
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Crop & Season Section */}
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                        <Sprout className="w-5 h-5 mr-2 text-green-600" />
                        Crop Details
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Crop <span className="text-red-500">*</span>
                          </label>
                          <select
                            name="crop"
                            value={formData.crop}
                            onChange={handleChange}
                            required
                            className="input-field"
                          >
                            <option value="">Select Crop</option>
                            {options.crops.map(crop => (
                              <option key={crop} value={crop}>{crop}</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Season <span className="text-red-500">*</span>
                          </label>
                          <select
                            name="season"
                            value={formData.season}
                            onChange={handleChange}
                            required
                            className="input-field"
                          >
                            <option value="">Select Season</option>
                            {options.seasons.map(season => (
                              <option key={season} value={season}>{season}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Year & Area Section */}
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                        <Calendar className="w-5 h-5 mr-2 text-blue-600" />
                        Cultivation Details
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Crop Year <span className="text-red-500">*</span>
                          </label>
                          <select
                            name="year"
                            value={formData.year}
                            onChange={handleChange}
                            required
                            className="input-field"
                          >
                            {years.map(year => (
                              <option key={year} value={year}>{year}</option>
                            ))}
                          </select>
                        </div>

                        <InputField
                          label="Area (Hectares)"
                          name="area"
                          type="number"
                          value={formData.area}
                          onChange={handleChange}
                          placeholder="Enter area"
                          required
                          min="0.1"
                          step="0.1"
                        />
                      </div>
                    </div>

                    <div className="flex space-x-3 mt-6">
                      <button type="submit" disabled={loading} className="btn-primary flex-1">
                        {loading ? 'Predicting...' : 'Predict Yield'}
                      </button>
                      <button 
                        type="button" 
                        onClick={() => {
                          setFormData({
                            state: '',
                            district: '',
                            crop: '',
                            year: currentYear,
                            season: '',
                            area: ''
                          });
                          setResult(null);
                        }} 
                        className="btn-secondary"
                      >
                        Reset
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </div>

            {/* Result Section */}
            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Predicting yield..." />
                  </div>
                ) : result ? (
                  <div>
                    <div className="card bg-gradient-to-br from-green-50 to-blue-50 border-2 border-green-200">
                      <h4 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
                        Prediction Result
                      </h4>
                      
                      <div className="mb-4">
                        <div className="text-sm text-gray-600 mb-1">Predicted Yield</div>
                        <div className="text-3xl font-bold text-green-700">
                          {result.predicted_yield}
                        </div>
                        <div className="text-sm text-gray-500">{result.unit}</div>
                      </div>

                      <div className="mb-4">
                        <div className="text-sm text-gray-600 mb-1">Total Production</div>
                        <div className="text-2xl font-semibold text-blue-700">
                          {result.estimated_production}
                        </div>
                        <div className="text-sm text-gray-500">{result.production_unit}</div>
                      </div>

                      <div className="pt-3 border-t border-gray-300">
                        <div className="text-xs text-gray-600 space-y-1">
                          <div className="flex justify-between">
                            <span>Model Confidence (R²):</span>
                            <span className="font-semibold">
                              {(result.confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Model Type:</span>
                            <span className="font-semibold">{result.model_type}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="card mt-4 bg-gray-50">
                      <h4 className="text-sm font-semibold text-gray-800 mb-2">Input Summary</h4>
                      <div className="text-xs text-gray-700 space-y-1">
                        <div><strong>Location:</strong> {result.input_values.district}, {result.input_values.state}</div>
                        <div><strong>Crop:</strong> {result.input_values.crop}</div>
                        <div><strong>Season:</strong> {result.input_values.season}</div>
                        <div><strong>Year:</strong> {result.input_values.year}</div>
                        <div><strong>Area:</strong> {result.input_values.area} hectares</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center text-sm mb-3">
                      Fill in the form and click "Predict Yield"
                    </p>
                    <div className="p-3 bg-blue-100 border border-blue-300 rounded text-xs text-blue-800">
                      💡 Select location manually or use the map to auto-fill state and district. Predictions are based on historical agricultural data.
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

export default YieldPrediction;
