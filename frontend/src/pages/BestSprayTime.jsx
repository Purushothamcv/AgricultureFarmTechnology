import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import InputField from '../components/InputField';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { cropService, weatherService } from '../services/services';
import { Sprout, CheckCircle, XCircle, RefreshCw, MapPin } from 'lucide-react';

const BestSprayTime = () => {
  const [formData, setFormData] = useState({
    temperature: '',
    humidity: '',
    windSpeed: '',
    rainfall: '',
    timeOfDay: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weatherLoaded, setWeatherLoaded] = useState(false);

  // Load weather data from localStorage on component mount
  useEffect(() => {
    const storedWeather = localStorage.getItem('currentWeather');
    if (storedWeather) {
      try {
        const weather = JSON.parse(storedWeather);
        setFormData(prev => ({
          ...prev,
          temperature: weather.temperature || '',
          humidity: weather.humidity || '',
          windSpeed: weather.wind_speed || '',
          rainfall: weather.rainfall || ''
        }));
        setWeatherLoaded(true);
      } catch (error) {
        console.error('Error loading stored weather:', error);
      }
    }
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Fetch current weather from user's location
  const fetchCurrentWeather = async () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        try {
          const data = await weatherService.getWeather(latitude, longitude);
          const updatedFormData = {
            ...formData,
            temperature: data.temp || '',
            humidity: data.humidity || '',
            windSpeed: data.wind || '',
            rainfall: data.rain || ''
          };
          setFormData(updatedFormData);
          setWeatherLoaded(true);
          
          // Store in localStorage for other modules
          const weatherData = {
            temperature: data.temp,
            humidity: data.humidity,
            wind_speed: data.wind,
            rainfall: data.rain,
            lat: latitude,
            lng: longitude
          };
          localStorage.setItem('currentWeather', JSON.stringify(weatherData));
          setLoading(false);
        } catch (error) {
          console.error('Error fetching weather:', error);
          alert('Failed to fetch weather data. Please enter manually.');
          setLoading(false);
        }
      },
      (error) => {
        setLoading(false);
        alert('Unable to get your location. Please enter weather data manually.');
        console.error('Geolocation error:', error);
      }
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const data = await cropService.getBestSprayTime(formData);
      setResult(data);
    } catch (err) {
      // Demo fallback
      const isSafe = parseFloat(formData.windSpeed) < 15 && 
                     parseFloat(formData.temperature) < 35 &&
                     parseFloat(formData.rainfall) < 1;
      
      setResult({
        is_safe: isSafe,
        recommendation: isSafe 
          ? 'Safe to spray - conditions are favorable' 
          : 'Not recommended - wait for better conditions',
        best_time: isSafe ? formData.timeOfDay : 'Early morning (6-8 AM) or late evening (5-7 PM)',
        factors: {
          wind: parseFloat(formData.windSpeed) < 15 ? 'Favorable' : 'Too high',
          temperature: parseFloat(formData.temperature) < 35 ? 'Optimal' : 'Too high',
          rainfall: parseFloat(formData.rainfall) < 1 ? 'No rain' : 'Rain expected'
        }
      });
    }
    setLoading(false);
  };

  const timeSlots = [
    'Early Morning (6-8 AM)',
    'Morning (8-10 AM)',
    'Late Morning (10-12 PM)',
    'Afternoon (12-3 PM)',
    'Late Afternoon (3-5 PM)',
    'Evening (5-7 PM)'
  ];

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <Sprout className="w-8 h-8 mr-3 text-primary-600" />
              Best Time to Spray
            </h1>
            <p className="text-gray-600">Get optimal spraying schedule based on weather conditions</p>
            
            {/* Weather Data Status */}
            <div className="mt-4 flex items-center gap-3">
              {weatherLoaded ? (
                <div className="flex items-center gap-2 text-green-600 text-sm">
                  <CheckCircle className="w-4 h-4" />
                  <span>Weather data loaded</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-amber-600 text-sm">
                  <MapPin className="w-4 h-4" />
                  <span>No weather data - enter manually or load from dashboard</span>
                </div>
              )}
              <button
                onClick={fetchCurrentWeather}
                disabled={loading}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 text-sm flex items-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                {loading ? 'Loading...' : 'Use Current Weather'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center justify-between">
                  <span>Weather Conditions</span>
                  {weatherLoaded && (
                    <span className="text-xs text-green-600 font-normal">Auto-filled</span>
                  )}
                </h3>
                <form onSubmit={handleSubmit}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <InputField
                      label="Temperature (°C)"
                      name="temperature"
                      type="number"
                      value={formData.temperature}
                      onChange={handleChange}
                      placeholder="Current/forecast temperature (or use button above)"
                      required
                      min="0"
                      step="0.1"
                    />
                    <InputField
                      label="Humidity (%)"
                      name="humidity"
                      type="number"
                      value={formData.humidity}
                      onChange={handleChange}
                      placeholder="Current/forecast humidity"
                      required
                      min="0"
                      max="100"
                      step="0.1"
                    />
                    <InputField
                      label="Wind Speed (km/h)"
                      name="windSpeed"
                      type="number"
                      value={formData.windSpeed}
                      onChange={handleChange}
                      placeholder="Current/forecast wind speed"
                      required
                      min="0"
                      step="0.1"
                    />
                    <InputField
                      label="Expected Rainfall (mm)"
                      name="rainfall"
                      type="number"
                      value={formData.rainfall}
                      onChange={handleChange}
                      placeholder="Forecast rainfall"
                      required
                      min="0"
                      step="0.1"
                    />
                    <div className="md:col-span-2">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Preferred Time Slot <span className="text-red-500">*</span>
                      </label>
                      <select
                        name="timeOfDay"
                        value={formData.timeOfDay}
                        onChange={handleChange}
                        required
                        className="input-field"
                      >
                        <option value="">Select time slot</option>
                        {timeSlots.map(slot => (
                          <option key={slot} value={slot}>{slot}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <button type="submit" disabled={loading} className="btn-primary w-full mt-6">
                    {loading ? 'Analyzing...' : 'Check Spray Conditions'}
                  </button>
                </form>
              </div>

              <div className="card mt-6 bg-amber-50 border border-amber-200">
                <h3 className="text-lg font-semibold text-amber-800 mb-3">💡 How to Use</h3>
                <ul className="space-y-2 text-sm text-gray-700 mb-4">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Option 1:</strong> Click "Use Current Weather" to auto-fill from your location</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Option 2:</strong> Select location on Dashboard first, then visit this page</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Option 3:</strong> Manually enter weather forecast data</span>
                  </li>
                </ul>
                
                <h3 className="text-lg font-semibold text-amber-800 mb-3 mt-4">Ideal Conditions</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Temperature:</strong> 15-25°C (optimal)</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Wind Speed:</strong> Below 15 km/h</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Humidity:</strong> 50-70%</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
                    <span><strong>Avoid:</strong> Rain within 6 hours</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Checking conditions..." />
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    {/* Safety Status Card */}
                    <div className={`card ${result.is_safe ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200' : 'bg-gradient-to-br from-orange-50 to-amber-50 border-2 border-orange-200'}`}>
                      <div className="flex items-start mb-4">
                        {result.is_safe ? (
                          <div className="p-2 bg-green-500 rounded-lg mr-3">
                            <CheckCircle className="w-6 h-6 text-white" />
                          </div>
                        ) : (
                          <div className="p-2 bg-orange-500 rounded-lg mr-3">
                            <XCircle className="w-6 h-6 text-white" />
                          </div>
                        )}
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-800">Spray Assessment</h3>
                          <p className={`text-xl font-bold mt-1 ${result.is_safe ? 'text-green-600' : 'text-orange-600'}`}>
                            {result.is_safe ? 'Safe to Spray' : 'Not Recommended'}
                          </p>
                        </div>
                      </div>

                      {/* Recommendation */}
                      <div className={`p-3 rounded-lg mb-3 ${result.is_safe ? 'bg-green-100' : 'bg-orange-100'}`}>
                        <p className={`text-sm font-medium ${result.is_safe ? 'text-green-800' : 'text-orange-800'}`}>
                          {result.recommendation}
                        </p>
                      </div>

                      {/* Best Time */}
                      <div className="border-t pt-3">
                        <p className="text-sm text-gray-600 mb-1">Best Time:</p>
                        <p className="text-base font-semibold text-gray-800">
                          {result.best_time}
                        </p>
                      </div>
                    </div>

                    {/* Factors Card */}
                    {result.factors && (
                      <div className="card bg-blue-50">
                        <h4 className="text-sm font-semibold text-gray-800 mb-3">Condition Analysis</h4>
                        <div className="space-y-2">
                          {Object.entries(result.factors).map(([key, value]) => {
                            const isFavorable = typeof value === 'string' && 
                              (value.toLowerCase().includes('favorable') || 
                               value.toLowerCase().includes('optimal') || 
                               value.toLowerCase().includes('no rain'));
                            
                            return (
                              <div key={key} className="flex items-center justify-between py-2 border-b last:border-b-0">
                                <span className="text-sm font-medium text-gray-700 capitalize">
                                  {key}:
                                </span>
                                <span className={`text-sm font-semibold px-2 py-1 rounded ${
                                  isFavorable 
                                    ? 'bg-green-100 text-green-700' 
                                    : 'bg-red-100 text-red-700'
                                }`}>
                                  {value}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center">
                      Enter weather conditions to get spraying recommendations
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

export default BestSprayTime;
