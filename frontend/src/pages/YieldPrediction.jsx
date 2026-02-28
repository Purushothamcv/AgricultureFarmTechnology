import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService } from '../services/services';
import { TrendingUp, MapPin, Calendar, Sprout, Map } from 'lucide-react';

// Use environment variable or fallback to localhost:8001
const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

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
  const [loadingCurrentLocation, setLoadingCurrentLocation] = useState(false);
  const [locationAutoFilled, setLocationAutoFilled] = useState({
    state: false,
    district: false
  });

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
      const response = await fetch(`${API_URL}/yield/states`);
      const data = await response.json();
      
      if (data.success) {
        setOptions(prev => ({
          ...prev,
          states: data.states || []
        }));
        console.log('✅ Loaded states:', data.states?.length);
      }

      // Load crops and seasons separately
      const optionsResponse = await fetch(`${API_URL}/api/yield/options`);
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
      const response = await fetch(`${API_URL}/yield/districts/${encodeURIComponent(state)}`);
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
    
    // Clear auto-fill indicator when user manually changes
    if (name === 'state' || name === 'district') {
      setLocationAutoFilled(prev => ({ ...prev, [name]: false }));
    }
    
    // If state changes, clear district selection
    if (name === 'state') {
      setFormData({
        ...formData,
        state: value,
        district: '' // Clear district when state changes
      });
      setLocationAutoFilled(prev => ({ ...prev, district: false }));
    } else {
      setFormData({
        ...formData,
        [name]: value
      });
    }
  };

  const handleMapLocationSelect = async (lat, lng) => {
    setShowMap(false); // Close map immediately for better UX
    
    try {
      // Reverse geocoding using Nominatim (OpenStreetMap) with English language preference
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&addressdetails=1&accept-language=en`,
        {
          headers: {
            'Accept-Language': 'en-US,en;q=0.9'
          }
        }
      );
      const data = await response.json();
      
      if (data.address) {
        // Try multiple possible fields for state and district
        const geocodedState = data.address.state || data.address.region || '';
        const geocodedDistrict = data.address.state_district || 
                                 data.address.county || 
                                 data.address.district || 
                                 data.address.city_district || '';
        
        console.log('📍 Location from map:', { 
          state: geocodedState, 
          district: geocodedDistrict, 
          lat, 
          lng,
          fullAddress: data.display_name 
        });
        
        // Helper function for fuzzy matching
        const fuzzyMatch = (str1, str2) => {
          if (!str1 || !str2) return false;
          const s1 = str1.toLowerCase().trim();
          const s2 = str2.toLowerCase().trim();
          
          // Exact match
          if (s1 === s2) return true;
          
          // Contains match
          if (s1.includes(s2) || s2.includes(s1)) return true;
          
          // Word match (at least one word matches)
          const words1 = s1.split(/\s+/);
          const words2 = s2.split(/\s+/);
          return words1.some(w1 => words2.some(w2 => w1 === w2 && w1.length > 3));
        };
        
        // Find matching state from available options (fuzzy matching)
        const matchingState = options.states.find(s => fuzzyMatch(s, geocodedState));
        
        if (matchingState) {
          console.log('✅ Found matching state:', matchingState);
          
          // First set the state - this will trigger district loading via useEffect
          setFormData(prev => ({
            ...prev,
            state: matchingState,
            district: '' // Clear district initially
          }));
          setLocationAutoFilled(prev => ({ ...prev, state: true, district: false }));
          
          // Load districts for this state and then find matching district
          try {
            const districtResponse = await fetch(`${API_URL}/yield/districts/${encodeURIComponent(matchingState)}`);
            const districtData = await districtResponse.json();
            
            if (districtData.success && districtData.districts) {
              // Find matching district using fuzzy matching
              const matchingDistrict = districtData.districts.find(d => fuzzyMatch(d, geocodedDistrict));
              
              if (matchingDistrict) {
                console.log('✅ Found matching district:', matchingDistrict);
                // Set the district after a short delay to ensure state update is processed
                setTimeout(() => {
                  setFormData(prev => ({
                    ...prev,
                    district: matchingDistrict
                  }));
                  setLocationAutoFilled(prev => ({ ...prev, district: true }));
                }, 200);
                
                alert(`✅ Location Auto-filled Successfully!\n\nState: ${matchingState}\nDistrict: ${matchingDistrict}\n\nPlease verify the values in the form below and complete other fields.`);
              } else {
                console.warn('⚠️ No matching district found. Available:', districtData.districts.slice(0, 3));
                alert(`✅ State auto-filled: ${matchingState}\n⚠️ District "${geocodedDistrict}" not found in database.\n\nPlease select the district manually from the dropdown.`);
              }
            }
          } catch (districtErr) {
            console.error('Error loading districts:', districtErr);
            alert(`✅ State auto-filled: ${matchingState}\n\nPlease select the district manually.`);
          }
        } else {
          console.warn('⚠️ No matching state found. Received:', geocodedState, 'Available:', options.states.slice(0, 5));
          alert(`❌ Location Not Found in Database\n\nDetected: ${geocodedState}, ${geocodedDistrict}\n\nThis state is not available in the yield prediction database.\nPlease select State and District manually from the dropdowns.`);
        }
      } else {
        throw new Error('No address data received from geocoding service');
      }
    } catch (err) {
      console.error('Error in reverse geocoding:', err);
      alert('❌ Could Not Determine Location\n\nPlease select State and District manually from the dropdowns.');
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
      
      const response = await fetch(`${API_URL}/predict-yield`, {
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

  // Simple Map Selector Component
  const MapSelector = ({ onLocationSelect }) => {
    const [selectedLocation, setSelectedLocation] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);

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

    const handleLocationConfirm = async () => {
      setIsProcessing(true);
      await onLocationSelect(selectedLocation.lat, selectedLocation.lng);
      setIsProcessing(false);
    };

    return (
      <div>
        <div className="mb-2 p-2 bg-blue-100 border border-blue-300 rounded text-sm text-blue-800">
          📍 <strong>Click on the map</strong> to select your location. State and District will be auto-filled.
        </div>
        <div id="yield-map" style={{ height: '300px', width: '100%', borderRadius: '8px', border: '2px solid #e5e7eb' }}></div>
        {selectedLocation && (
          <div className="mt-3 flex items-center justify-between bg-green-50 p-3 rounded-lg border border-green-200">
            <p className="text-sm text-gray-700">
              📌 Selected: <span className="font-mono font-semibold">{selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}</span>
            </p>
            <button
              type="button"
              onClick={handleLocationConfirm}
              disabled={isProcessing}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? '🔄 Processing...' : '✓ Use This Location'}
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

                      {showMap && (
                        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <MapSelector onLocationSelect={handleMapLocationSelect} />
                        </div>
                      )}

                      {(locationAutoFilled.state || locationAutoFilled.district) && (
                        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-start">
                          <div className="flex-shrink-0">
                            <svg className="w-5 h-5 text-green-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="ml-3 flex-1">
                            <p className="text-sm font-medium text-green-800">
                              Location auto-filled from map
                            </p>
                            <p className="text-xs text-green-700 mt-1">
                              {locationAutoFilled.state && `State: ${formData.state}`}
                              {locationAutoFilled.state && locationAutoFilled.district && ' • '}
                              {locationAutoFilled.district && `District: ${formData.district}`}
                            </p>
                          </div>
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                            State <span className="text-red-500">*</span>
                            {locationAutoFilled.state && (
                              <span className="ml-2 text-xs text-green-600 bg-green-100 px-2 py-0.5 rounded-full flex items-center">
                                ✓ Auto-filled
                              </span>
                            )}
                          </label>
                          <select
                            name="state"
                            value={formData.state}
                            onChange={handleChange}
                            required
                            className={`input-field ${locationAutoFilled.state ? 'bg-green-50 border-green-300' : ''}`}
                          >
                            <option value="">Select State</option>
                            {options.states.map(state => (
                              <option key={state} value={state}>{state}</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                            District <span className="text-red-500">*</span>
                            {locationAutoFilled.district && (
                              <span className="ml-2 text-xs text-green-600 bg-green-100 px-2 py-0.5 rounded-full flex items-center">
                                ✓ Auto-filled
                              </span>
                            )}
                          </label>
                          <select
                            name="district"
                            value={formData.district}
                            onChange={handleChange}
                            required
                            disabled={!formData.state || loadingDistricts}
                            className={`input-field ${locationAutoFilled.district ? 'bg-green-50 border-green-300' : ''}`}
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
                          setLocationAutoFilled({ state: false, district: false });
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
                    <div className="p-3 bg-blue-100 border border-blue-300 rounded text-xs text-blue-800 space-y-2">
                      <div className="flex items-start">
                        <span className="mr-1">📍</span>
                        <p><strong>Quick Start:</strong> Click "Select from Map" or "Use My Location" to auto-fill State and District</p>
                      </div>
                      <div className="flex items-start">
                        <span className="mr-1">💡</span>
                        <p>Predictions are based on historical agricultural data using machine learning</p>
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

export default YieldPrediction;
