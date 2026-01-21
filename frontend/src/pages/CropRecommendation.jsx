import React, { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService } from '../services/services';
import { Leaf, MapPin, Edit3 } from 'lucide-react';

const CropRecommendation = () => {
  const [mode, setMode] = useState('manual'); // 'manual' or 'map'
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: '',
    ozone: ''
  });
  const [locationData, setLocationData] = useState({
    latitude: '',
    longitude: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fetchingLocation, setFetchingLocation] = useState(false);
  
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  // Initialize Leaflet map
  useEffect(() => {
    if (mode === 'map' && mapRef.current && !mapInstanceRef.current) {
      // Load Leaflet CSS
      if (!document.getElementById('leaflet-css')) {
        const link = document.createElement('link');
        link.id = 'leaflet-css';
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);
      }

      // Load Leaflet JS
      if (!window.L) {
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        script.onload = initializeMap;
        document.body.appendChild(script);
      } else {
        initializeMap();
      }
    }
  }, [mode]);

  const initializeMap = () => {
    if (mapInstanceRef.current) return;

    const L = window.L;
    const map = L.map(mapRef.current).setView([20.5937, 78.9629], 5);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    map.on('click', async (e) => {
      const { lat, lng } = e.latlng;
      selectLocation(lat, lng, L, map);
    });

    mapInstanceRef.current = map;
  };

  const selectLocation = async (lat, lng, L, map) => {
    if (markerRef.current) {
      map.removeLayer(markerRef.current);
    }

    markerRef.current = L.marker([lat, lng]).addTo(map);

    setLocationData({
      latitude: lat.toFixed(4),
      longitude: lng.toFixed(4)
    });

    setFetchingLocation(true);
    try {
      const data = await cropService.fetchLocationData(lat, lng);
      if (data.success) {
        setFormData({
          nitrogen: data.nitrogen?.toFixed(2) || '',
          phosphorus: data.phosphorus?.toFixed(2) || '',
          potassium: data.potassium?.toFixed(2) || '',
          temperature: data.temperature?.toFixed(2) || '',
          humidity: data.humidity?.toFixed(2) || '',
          ph: data.ph?.toFixed(2) || '',
          rainfall: data.rainfall?.toFixed(2) || '',
          ozone: data.ozone?.toFixed(2) || ''
        });
      }
    } catch (err) {
      console.error('Error fetching location data:', err);
      setError('Failed to fetch location data. Please enter values manually.');
    }
    setFetchingLocation(false);
  };

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    setFetchingLocation(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const L = window.L;
        const map = mapInstanceRef.current;
        
        if (map && L) {
          map.setView([latitude, longitude], 12);
          selectLocation(latitude, longitude, L, map);
        }
      },
      (error) => {
        setFetchingLocation(false);
        alert('Unable to get your location. Please select manually on the map.');
      }
    );
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      let data;
      if (mode === 'manual') {
        data = await cropService.recommendCrop(formData);
      } else {
        if (!locationData.latitude || !locationData.longitude) {
          setError('Please select a location on the map first');
          setLoading(false);
          return;
        }
        data = await cropService.recommendCropByLocation({
          ...locationData,
          ...formData
        });
      }
      
      if (data.success) {
        setResult(data);
      } else {
        setError(data.message || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.message || 'Failed to get recommendation');
    }
    setLoading(false);
  };

  const handleReset = () => {
    setFormData({
      nitrogen: '',
      phosphorus: '',
      potassium: '',
      temperature: '',
      humidity: '',
      ph: '',
      rainfall: '',
      ozone: ''
    });
    setLocationData({
      latitude: '',
      longitude: ''
    });
    setResult(null);
    setError('');
    
    if (markerRef.current && mapInstanceRef.current) {
      mapInstanceRef.current.removeLayer(markerRef.current);
      markerRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8 text-center">
            <h1 className="text-4xl font-bold text-gray-800 mb-3 flex items-center justify-center">
              <Leaf className="w-10 h-10 mr-3 text-green-600" />
              Crop Recommendation System
            </h1>
            <p className="text-gray-600 text-lg">
              Get personalized crop recommendations based on soil and weather conditions
            </p>
          </div>

          <div className="flex justify-center gap-4 mb-8">
            <button
              type="button"
              onClick={() => {
                setMode('manual');
                handleReset();
              }}
              className={`px-8 py-3 rounded-lg font-semibold text-lg transition-all ${
                mode === 'manual'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white text-green-600 border-2 border-green-600 hover:bg-green-50'
              }`}
            >
              <Edit3 className="inline w-5 h-5 mr-2" />
              Manual Input
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('map');
                handleReset();
              }}
              className={`px-8 py-3 rounded-lg font-semibold text-lg transition-all ${
                mode === 'map'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white text-green-600 border-2 border-green-600 hover:bg-green-50'
              }`}
            >
              <MapPin className="inline w-5 h-5 mr-2" />
              Select from Map
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-2xl shadow-xl p-6">
                {mode === 'map' && (
                  <div className="mb-6">
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-xl font-semibold text-gray-800">Select Location</h2>
                      <button
                        type="button"
                        onClick={getCurrentLocation}
                        disabled={fetchingLocation}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
                      >
                        {fetchingLocation ? 'Locating...' : '📍 Use My Location'}
                      </button>
                    </div>
                    
                    <div 
                      ref={mapRef} 
                      className="w-full h-96 rounded-lg shadow-md mb-4"
                      style={{ minHeight: '400px' }}
                    />
                    
                    {locationData.latitude && locationData.longitude && (
                      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                        <p className="font-semibold text-blue-900">Selected Location:</p>
                        <p className="text-blue-800">
                          Latitude: {locationData.latitude}, Longitude: {locationData.longitude}
                        </p>
                      </div>
                    )}
                  </div>
                )}
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-6">
                    {mode === 'map' ? 'Review & Edit Parameters' : 'Input Parameters'}
                  </h2>
                  
                  {error && (
                    <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
                      {error}
                    </div>
                  )}

                  <form onSubmit={handleSubmit}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                      <InputField
                        label="Nitrogen (N) - kg/ha"
                        name="nitrogen"
                        type="number"
                        value={formData.nitrogen}
                        onChange={handleChange}
                        placeholder="0-200"
                        required
                        min="0"
                        max="200"
                        step="0.01"
                      />
                      <InputField
                        label="Phosphorus (P) - kg/ha"
                        name="phosphorus"
                        type="number"
                        value={formData.phosphorus}
                        onChange={handleChange}
                        placeholder="0-200"
                        required
                        min="0"
                        max="200"
                        step="0.01"
                      />
                      <InputField
                        label="Potassium (K) - kg/ha"
                        name="potassium"
                        type="number"
                        value={formData.potassium}
                        onChange={handleChange}
                        placeholder="0-200"
                        required
                        min="0"
                        max="200"
                        step="0.01"
                      />
                      <InputField
                        label="Temperature - °C"
                        name="temperature"
                        type="number"
                        value={formData.temperature}
                        onChange={handleChange}
                        placeholder="-10 to 60"
                        required
                        min="-10"
                        max="60"
                        step="0.01"
                      />
                      <InputField
                        label="Humidity - %"
                        name="humidity"
                        type="number"
                        value={formData.humidity}
                        onChange={handleChange}
                        placeholder="0-100"
                        required
                        min="0"
                        max="100"
                        step="0.01"
                      />
                      <InputField
                        label="pH Value"
                        name="ph"
                        type="number"
                        value={formData.ph}
                        onChange={handleChange}
                        placeholder="3-10"
                        required
                        min="3"
                        max="10"
                        step="0.01"
                      />
                      <InputField
                        label="Rainfall - mm"
                        name="rainfall"
                        type="number"
                        value={formData.rainfall}
                        onChange={handleChange}
                        placeholder="0-500"
                        required
                        min="0"
                        max="500"
                        step="0.01"
                      />
                      <InputField
                        label="Ozone (O3) - ppb"
                        name="ozone"
                        type="number"
                        value={formData.ozone}
                        onChange={handleChange}
                        placeholder="0-100"
                        required
                        min="0"
                        max="100"
                        step="0.01"
                      />
                    </div>

                    <div className="flex gap-4">
                      <button
                        type="submit"
                        disabled={loading}
                        className="flex-1 bg-gradient-to-r from-green-600 to-green-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-700 hover:to-green-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                      >
                        {loading ? 'Predicting...' : '🌱 Get Crop Recommendation'}
                      </button>
                      <button
                        type="button"
                        onClick={handleReset}
                        className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
                      >
                        Reset
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="bg-white rounded-2xl shadow-xl p-6 sticky top-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Recommendation Result
                </h2>

                {loading && (
                  <div className="text-center py-12">
                    <LoadingSpinner />
                    <p className="text-gray-600 mt-4">Analyzing data...</p>
                  </div>
                )}

                {!loading && !result && (
                  <div className="text-center py-12 text-gray-500">
                    <Leaf className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                    <p>Enter parameters and submit to get recommendation</p>
                  </div>
                )}

                {!loading && result && (
                  <div className="space-y-4">
                    <div className="bg-gradient-to-br from-green-100 to-green-50 border-2 border-green-500 rounded-xl p-6 text-center">
                      <p className="text-sm font-semibold text-green-700 mb-2">
                        RECOMMENDED CROP
                      </p>
                      <h3 className="text-4xl font-bold text-green-900 mb-2">
                        {result.crop}
                      </h3>
                      {result.confidence && (
                        <div className="inline-block bg-green-600 text-white px-4 py-2 rounded-full font-semibold">
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>

                    {result.input_values && (
                      <div className="bg-gray-50 rounded-xl p-4">
                        <p className="font-semibold text-gray-800 mb-3">Input Summary:</p>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          {Object.entries(result.input_values).map(([key, value]) => (
                            <div key={key} className="bg-white rounded p-2">
                              <span className="text-gray-600 capitalize">{key}:</span>
                              <span className="font-semibold text-gray-800 ml-1">
                                {typeof value === 'number' ? value.toFixed(2) : value}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.message && (
                      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                        <p className="text-sm text-blue-800">{result.message}</p>
                      </div>
                    )}
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

export default CropRecommendation;
