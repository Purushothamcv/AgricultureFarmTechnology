import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { fertilizerService } from '../services/services';
import { Droplet, Cloud, RefreshCw } from 'lucide-react';

const FertilizerRecommendation = () => {
  const [manualWeather, setManualWeather] = useState(false);
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    crop: '',
    soilMoisture: '',
    temperature: '',
    humidity: '',
    rainfall: '',
    lat: '',
    lng: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weatherSource, setWeatherSource] = useState('auto');

  // Load weather from localStorage
  useEffect(() => {
    const storedWeather = localStorage.getItem('currentWeather');
    if (storedWeather) {
      try {
        const weather = JSON.parse(storedWeather);
        setFormData(prev => ({
          ...prev,
          temperature: weather.temperature?.toString() || '',
          humidity: weather.humidity?.toString() || '',
          rainfall: weather.rainfall?.toString() || '',
          lat: weather.lat?.toString() || '',
          lng: weather.lng?.toString() || ''
        }));
        setWeatherSource('auto');
        console.log('✅ Auto-filled weather for Fertilizer Recommendation:', weather);
      } catch (e) {
        console.error('Failed to parse stored weather', e);
      }
    }
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleToggleManual = () => {
    setManualWeather(!manualWeather);
    setWeatherSource(manualWeather ? 'auto' : 'manual');
  };

  const handleReset = () => {
    const storedWeather = localStorage.getItem('currentWeather');
    const weather = storedWeather ? JSON.parse(storedWeather) : {};
    
    setFormData({
      nitrogen: '',
      phosphorus: '',
      potassium: '',
      crop: '',
      soilMoisture: '',
      temperature: weather.temperature?.toString() || '',
      humidity: weather.humidity?.toString() || '',
      rainfall: weather.rainfall?.toString() || '',
      lat: weather.lat?.toString() || '',
      lng: weather.lng?.toString() || ''
    });
    setResult(null);
    setManualWeather(false);
    setWeatherSource('auto');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        N: parseFloat(formData.nitrogen),
        P: parseFloat(formData.phosphorus),
        K: parseFloat(formData.potassium),
        crop: formData.crop,
        soilMoisture: parseFloat(formData.soilMoisture),
        lat: parseFloat(formData.lat) || 20.5937,
        lng: parseFloat(formData.lng) || 78.9629
      };

      // Include weather only if manually editing
      if (manualWeather) {
        payload.temperature = parseFloat(formData.temperature);
        payload.humidity = parseFloat(formData.humidity);
        payload.rainfall = parseFloat(formData.rainfall);
      }

      console.log('📤 Sending fertilizer recommendation request:', payload);
      const data = await fertilizerService.recommendFertilizer(payload);
      console.log('📥 Received fertilizer recommendation:', data);
      setResult(data);
    } catch (err) {
      console.error('Error:', err);
      // Demo fallback
      const fertilizers = ['Urea (N)', 'DAP (P)', 'MOP (K)'];
      setResult({
        fertilizer: fertilizers.join(', '),
        recommendations: [
          'Apply in split doses for better efficiency',
          'Irrigate after application for nutrient absorption'
        ],
        npk_status: {
          nitrogen: formData.nitrogen,
          phosphorus: formData.phosphorus,
          potassium: formData.potassium
        }
      });
    }
    setLoading(false);
  };

  const crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato'];
  const soilTypes = ['Alluvial', 'Clay', 'Loamy', 'Sandy', 'Red', 'Black'];

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <Droplet className="w-8 h-8 mr-3 text-primary-600" />
              Fertilizer Recommendation
            </h1>
            <p className="text-gray-600">Get optimal fertilizer recommendations for your crops</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  {/* Crop Selection */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Crop Type <span className="text-red-500">*</span>
                    </label>
                    <select
                      name="crop"
                      value={formData.crop}
                      onChange={handleChange}
                      required
                      className="input-field"
                    >
                      <option value="">Select a crop</option>
                      {crops.map(crop => (
                        <option key={crop} value={crop}>{crop}</option>
                      ))}
                    </select>
                  </div>

                  {/* NPK Levels */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <InputField
                      label="Nitrogen (N)"
                      name="nitrogen"
                      type="number"
                      value={formData.nitrogen}
                      onChange={handleChange}
                      placeholder="0-140"
                      required
                      min="0"
                      max="200"
                      step="0.1"
                    />
                    <InputField
                      label="Phosphorus (P)"
                      name="phosphorus"
                      type="number"
                      value={formData.phosphorus}
                      onChange={handleChange}
                      placeholder="0-150"
                      required
                      min="0"
                      max="200"
                      step="0.1"
                    />
                    <InputField
                      label="Potassium (K)"
                      name="potassium"
                      type="number"
                      value={formData.potassium}
                      onChange={handleChange}
                      placeholder="0-210"
                      required
                      min="0"
                      max="250"
                      step="0.1"
                    />
                  </div>

                  {/* Soil Moisture */}
                  <div className="mb-4">
                    <InputField
                      label="Soil Moisture (0-1)"
                      name="soilMoisture"
                      type="number"
                      value={formData.soilMoisture}
                      onChange={handleChange}
                      placeholder="0.5"
                      required
                      min="0"
                      max="1"
                      step="0.01"
                    />
                  </div>

                  {/* Weather Section with Auto-fill */}
                  <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center">
                        <Cloud className="w-5 h-5 mr-2 text-blue-600" />
                        <h3 className="text-sm font-semibold text-gray-800">
                          Weather Parameters
                        </h3>
                      </div>
                      <label className="flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={manualWeather}
                          onChange={handleToggleManual}
                          className="mr-2"
                        />
                        <span className="text-sm text-gray-700">Manual Edit</span>
                      </label>
                    </div>

                    {weatherSource === 'auto' && !manualWeather && (
                      <div className="mb-2 text-xs text-blue-700 flex items-center">
                        <RefreshCw className="w-3 h-3 mr-1" />
                        Auto-filled from Dashboard. Enable "Manual Edit" to customize.
                      </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <InputField
                        label="Temperature (°C)"
                        name="temperature"
                        type="number"
                        value={formData.temperature}
                        onChange={handleChange}
                        placeholder="Auto-filled"
                        disabled={!manualWeather}
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
                        disabled={!manualWeather}
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
                        disabled={!manualWeather}
                        required
                        step="0.1"
                      />
                    </div>
                  </div>

                  {/* Reset Button */}
                  <button
                    type="button"
                    onClick={handleReset}
                    className="btn-secondary w-full mb-2"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Reset & Reload Weather
                  </button>

                  <button type="submit" disabled={loading} className="btn-primary w-full">
                    {loading ? 'Processing...' : 'Get Recommendation'}
                  </button>
                </form>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Analyzing soil..." />
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    {/* Fertilizer Recommendation */}
                    <div className="card bg-gradient-to-br from-green-50 to-emerald-50">
                      <div className="flex items-center mb-4">
                        <div className="p-2 bg-green-500 rounded-lg mr-3">
                          <Droplet className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-gray-800">Recommended Fertilizer</h3>
                          <p className="text-2xl font-bold text-green-600 mt-1">{result.fertilizer}</p>
                        </div>
                      </div>

                      {result.recommendations && result.recommendations.length > 0 && (
                        <div className="mt-4">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Application Guidelines:</h4>
                          <ul className="space-y-2">
                            {result.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start text-sm text-gray-700">
                                <span className="text-green-600 mr-2">•</span>
                                <span>{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {result.npk_status && (
                        <div className="mt-4 pt-4 border-t border-green-200">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">NPK Status:</h4>
                          <div className="grid grid-cols-3 gap-2">
                            <div className="bg-white p-2 rounded">
                              <p className="text-xs text-gray-600">Nitrogen</p>
                              <p className="text-sm font-semibold text-gray-800">{result.npk_status.N}</p>
                            </div>
                            <div className="bg-white p-2 rounded">
                              <p className="text-xs text-gray-600">Phosphorus</p>
                              <p className="text-sm font-semibold text-gray-800">{result.npk_status.P}</p>
                            </div>
                            <div className="bg-white p-2 rounded">
                              <p className="text-xs text-gray-600">Potassium</p>
                              <p className="text-sm font-semibold text-gray-800">{result.npk_status.K}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Weather Used */}
                    {result.weather_used && (
                      <div className="card bg-blue-50">
                        <div className="flex items-center mb-3">
                          <Cloud className="w-5 h-5 mr-2 text-blue-600" />
                          <h4 className="text-sm font-semibold text-gray-800">Weather Conditions Used</h4>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Temperature:</span>
                            <span className="font-medium">{result.weather_used.temperature}°C</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Humidity:</span>
                            <span className="font-medium">{result.weather_used.humidity}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Rainfall:</span>
                            <span className="font-medium">{result.weather_used.rainfall}mm</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Soil Moisture:</span>
                            <span className="font-medium">{result.weather_used.soilMoisture}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center">
                      Enter soil and crop details to get fertilizer recommendations
                    </p>
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

export default FertilizerRecommendation;
