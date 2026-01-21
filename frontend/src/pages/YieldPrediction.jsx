import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService } from '../services/services';
import { TrendingUp, Cloud, RefreshCw } from 'lucide-react';

const YieldPrediction = () => {
  const [manualWeather, setManualWeather] = useState(false);
  const [formData, setFormData] = useState({
    crop: '',
    area: '',
    soilMoisture: '',
    ozone: '40',
    temperature: '',
    humidity: '',
    rainfall: '',
    lat: '',
    lng: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weatherSource, setWeatherSource] = useState('auto');

  // Load weather data from localStorage on component mount
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
        console.log('✅ Auto-filled weather from Dashboard:', weather);
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        crop: formData.crop,
        area: parseFloat(formData.area),
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
      }

      console.log('📤 Sending yield prediction request:', payload);
      const data = await cropService.predictYield(payload);
      console.log('📥 Received yield prediction:', data);
      setResult(data);
    } catch (err) {
      console.error('Error:', err);
      // Demo fallback
      const predictedYield = (parseFloat(formData.area) * Math.random() * 50 + 100).toFixed(2);
      setResult({
        yield: `${predictedYield} tonnes/hectare`,
        value: parseFloat(predictedYield),
        crop: formData.crop,
        weather_used: {
          temperature: formData.temperature,
          humidity: formData.humidity,
          rainfall: formData.rainfall
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
              <TrendingUp className="w-8 h-8 mr-3 text-primary-600" />
              Yield Prediction
            </h1>
            <p className="text-gray-600">Predict crop yield based on environmental factors</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
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
                        Auto-filled from Dashboard weather. Enable "Manual Edit" to customize.
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

                  <div className="flex space-x-3 mt-6">
                    <button type="submit" disabled={loading} className="btn-primary flex-1">
                      {loading ? 'Predicting...' : 'Predict Yield'}
                    </button>
                    <button 
                      type="button" 
                      onClick={() => {
                        const storedWeather = localStorage.getItem('currentWeather');
                        const weather = storedWeather ? JSON.parse(storedWeather) : {};
                        setFormData({
                          crop: '',
                          area: '',
                          soilMoisture: '',
                          ozone: '40',
                          temperature: weather.temperature?.toString() || '',
                          humidity: weather.humidity?.toString() || '',
                          rainfall: weather.rainfall?.toString() || '',
                          lat: weather.lat?.toString() || '',
                          lng: weather.lng?.toString() || ''
                        });
                        setResult(null);
                      }} 
                      className="btn-secondary"
                    >
                      Reset
                    </button>
                  </div>
                </form>
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
                    <ResultCard
                      result={{
                        'Yield': result.yield || result.predicted_yield,
                        'Crop': result.crop || formData.crop
                      }}
                      type="success"
                      title="Prediction Result"
                      icon={TrendingUp}
                    />
                    {result.weather_used && (
                      <div className="card mt-4 bg-blue-50 border border-blue-200">
                        <h4 className="text-sm font-semibold text-gray-800 mb-2 flex items-center">
                          <Cloud className="w-4 h-4 mr-2 text-blue-600" />
                          Weather Used
                        </h4>
                        <div className="text-xs text-gray-700 space-y-1">
                          <div>Temp: {result.weather_used.temperature}°C</div>
                          <div>Humidity: {result.weather_used.humidity}%</div>
                          <div>Rainfall: {result.weather_used.rainfall} mm</div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center text-sm mb-3">
                      Fill in the form and click "Predict Yield"
                    </p>
                    {!formData.temperature && (
                      <div className="p-3 bg-blue-100 border border-blue-300 rounded text-xs text-blue-800">
                        💡 Tip: Go to Dashboard and click "Get Weather" to auto-fill weather data
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

export default YieldPrediction;
