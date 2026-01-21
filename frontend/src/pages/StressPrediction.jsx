import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService } from '../services/services';
import { AlertTriangle, Cloud, RefreshCw } from 'lucide-react';

const StressPrediction = () => {
  const [manualWeather, setManualWeather] = useState(false);
  const [formData, setFormData] = useState({
    temperature: '',
    humidity: '',
    soilMoisture: '',
    rainfall: '',
    ozone: '40',
    windSpeed: '',
    lat: '',
    lng: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weatherSource, setWeatherSource] = useState('auto');

  // Load weather data from localStorage on mount
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
          windSpeed: weather.wind_speed?.toString() || '',
          lat: weather.lat?.toString() || '',
          lng: weather.lng?.toString() || ''
        }));
        setWeatherSource('auto');
        console.log('✅ Auto-filled weather for Stress Prediction:', weather);
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
      temperature: weather.temperature?.toString() || '',
      humidity: weather.humidity?.toString() || '',
      soilMoisture: '',
      rainfall: weather.rainfall?.toString() || '',
      ozone: '40',
      windSpeed: weather.wind_speed?.toString() || '',
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
        soilMoisture: parseFloat(formData.soilMoisture),
        ozone: parseFloat(formData.ozone),
        lat: parseFloat(formData.lat) || 20.5937,
        lng: parseFloat(formData.lng) || 78.9629
      };

      // Include weather only if manually editing
      if (manualWeather) {
        payload.temperature = parseFloat(formData.temperature);
        payload.humidity = parseFloat(formData.humidity);
        payload.rainfall = parseFloat(formData.rainfall);
        payload.windSpeed = parseFloat(formData.windSpeed);
      }

      console.log('📤 Sending stress prediction request:', payload);
      const data = await cropService.predictStress(payload);
      console.log('📥 Received stress prediction:', data);
      setResult(data);
    } catch (err) {
      console.error('Error:', err);
      // Demo fallback
      const stressLevels = ['Low', 'Moderate', 'High'];
      const level = stressLevels[Math.floor(Math.random() * stressLevels.length)];
      setResult({
        level: level,
        factors: level === 'High' 
          ? ['Extreme temperature', 'Low soil moisture'] 
          : level === 'Moderate'
          ? ['Humidity stress']
          : ['Optimal conditions'],
        score: level === 'High' ? 4 : level === 'Moderate' ? 2 : 0,
        weather_used: {
          temperature: formData.temperature,
          humidity: formData.humidity,
          rainfall: formData.rainfall,
          windSpeed: formData.windSpeed
        }
      });
    }
    setLoading(false);
  };

  const getStressType = (level) => {
    if (!level) return 'info';
    if (level.toLowerCase().includes('low')) return 'success';
    if (level.toLowerCase().includes('medium')) return 'warning';
    return 'error';
  };

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <AlertTriangle className="w-8 h-8 mr-3 text-primary-600" />
              Stress Prediction
            </h1>
            <p className="text-gray-600">Predict and monitor crop stress levels</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  {/* Soil Parameters */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
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
                    <InputField
                      label="Ozone (ppb)"
                      name="ozone"
                      type="number"
                      value={formData.ozone}
                      onChange={handleChange}
                      placeholder="40"
                      required
                      min="0"
                      max="100"
                      step="1"
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

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                      <InputField
                        label="Wind Speed (km/h)"
                        name="windSpeed"
                        type="number"
                        value={formData.windSpeed}
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
                    {loading ? 'Analyzing...' : 'Predict Stress Level'}
                  </button>
                </form>
              </div>
            </div>

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
                      result.level === 'Low Stress' ? 'bg-gradient-to-br from-green-50 to-emerald-50' :
                      result.level === 'Moderate Stress' ? 'bg-gradient-to-br from-yellow-50 to-amber-50' :
                      'bg-gradient-to-br from-red-50 to-rose-50'
                    }`}>
                      <div className="flex items-center mb-4">
                        <div className={`p-2 rounded-lg mr-3 ${
                          result.level === 'Low Stress' ? 'bg-green-500' :
                          result.level === 'Moderate Stress' ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}>
                          <AlertTriangle className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-gray-800">Stress Level</h3>
                          <p className={`text-2xl font-bold mt-1 ${
                            result.level === 'Low Stress' ? 'text-green-600' :
                            result.level === 'Moderate Stress' ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>{result.level}</p>
                        </div>
                      </div>

                      {result.score !== undefined && (
                        <div className="mt-2">
                          <p className="text-sm text-gray-600">Stress Score: <span className="font-semibold">{result.score.toFixed(2)}</span></p>
                        </div>
                      )}
                    </div>

                    {/* Stress Factors */}
                    {result.factors && result.factors.length > 0 && (
                      <div className="card bg-orange-50">
                        <div className="flex items-center mb-3">
                          <AlertTriangle className="w-5 h-5 mr-2 text-orange-600" />
                          <h4 className="text-sm font-semibold text-gray-800">Contributing Factors</h4>
                        </div>
                        <ul className="space-y-2">
                          {result.factors.map((factor, idx) => (
                            <li key={idx} className="flex items-start text-sm text-gray-700">
                              <span className="text-orange-600 mr-2">•</span>
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Weather Used */}
                    {result.weather_used && (
                      <div className="card bg-blue-50">
                        <div className="flex items-center mb-3">
                          <Cloud className="w-5 h-5 mr-2 text-blue-600" />
                          <h4 className="text-sm font-semibold text-gray-800">Conditions Analyzed</h4>
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
                            <span className="text-gray-600">Wind Speed:</span>
                            <span className="font-medium">{result.weather_used.windSpeed}km/h</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Soil Moisture:</span>
                            <span className="font-medium">{result.weather_used.soilMoisture}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Ozone:</span>
                            <span className="font-medium">{result.weather_used.ozone}ppb</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center">
                      Enter environmental parameters to predict crop stress level
                    </p>
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
                Optimal growing conditions. Continue current management practices.
              </p>
            </div>
            <div className="card bg-yellow-50 border-l-4 border-yellow-500">
              <h4 className="font-semibold text-yellow-800 mb-2">Moderate Stress</h4>
              <p className="text-sm text-gray-700">
                Monitor closely and adjust irrigation or nutrient management.
              </p>
            </div>
            <div className="card bg-red-50 border-l-4 border-red-500">
              <h4 className="font-semibold text-red-800 mb-2">High Stress</h4>
              <p className="text-sm text-gray-700">
                Immediate intervention required to prevent yield loss.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StressPrediction;
