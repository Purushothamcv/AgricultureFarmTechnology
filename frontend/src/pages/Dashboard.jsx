import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from 'react-leaflet';
import { 
  Leaf, 
  TrendingUp, 
  Droplet, 
  AlertTriangle, 
  Sprout,
  ImageIcon,
  MessageSquare,
  MapPin,
  Search,
  X
} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import Navbar from '../components/Navbar';
import WeatherCard from '../components/WeatherCard';
import LoadingSpinner from '../components/LoadingSpinner';
import InputField from '../components/InputField';
import { weatherService } from '../services/services';

// Fix for default marker icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to handle map view changes
function MapViewController({ center, zoom }) {
  const map = useMap();
  
  useEffect(() => {
    if (center) {
      map.setView([center.lat, center.lng], zoom || 13, {
        animate: true,
        duration: 1
      });
    }
  }, [center, zoom, map]);
  
  return null;
}

function LocationMarker({ position, setPosition }) {
  useMapEvents({
    click(e) {
      setPosition(e.latlng);
    },
  });

  return position === null ? null : <Marker position={position} />;
}

const Dashboard = () => {
  const navigate = useNavigate();
  const mapRef = useRef(null);
  const [position, setPosition] = useState({ lat: 28.6139, lng: 77.2090 }); // Default: New Delhi
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Search for locations using Nominatim API
  const searchLocation = async (query) => {
    if (!query || query.trim().length < 3) {
      setSearchResults([]);
      return;
    }

    setSearching(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`,
        {
          headers: {
            'User-Agent': 'SmartAgri-AI'
          }
        }
      );
      const data = await response.json();
      setSearchResults(data);
      setShowResults(true);
    } catch (error) {
      console.error('Error searching location:', error);
      setSearchResults([]);
    }
    setSearching(false);
  };

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchQuery) {
        searchLocation(searchQuery);
      } else {
        setSearchResults([]);
        setShowResults(false);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Handle location selection from search results
  const handleLocationSelect = async (result) => {
    const newPosition = {
      lat: parseFloat(result.lat),
      lng: parseFloat(result.lon)
    };
    setPosition(newPosition);
    setSearchQuery(result.display_name);
    setShowResults(false);
    
    // Automatically fetch weather for selected location
    setLoading(true);
    try {
      const data = await weatherService.getWeather(newPosition.lat, newPosition.lng);
      const mappedWeather = {
        temperature: data.temp !== undefined && data.temp !== null ? data.temp : 0,
        humidity: data.humidity !== undefined && data.humidity !== null ? data.humidity : 0,
        rainfall: data.rain !== undefined && data.rain !== null ? data.rain : 0,
        wind_speed: data.wind !== undefined && data.wind !== null ? data.wind : 0,
        lat: newPosition.lat,
        lng: newPosition.lng,
        description: result.display_name
      };
      
      setWeather(mappedWeather);
      localStorage.setItem('currentWeather', JSON.stringify(mappedWeather));
      console.log('✅ Weather fetched for:', result.display_name);
    } catch (error) {
      console.error('❌ Error fetching weather:', error);
    }
    setLoading(false);
  };

  const handleGetWeather = async () => {
    setLoading(true);
    try {
      const data = await weatherService.getWeather(position.lat, position.lng);
      console.log('✅ Weather API Response:', data); // Debug log
      
      // Map backend response keys to frontend expected keys with null safety
      const mappedWeather = {
        temperature: data.temp !== undefined && data.temp !== null ? data.temp : 0,
        humidity: data.humidity !== undefined && data.humidity !== null ? data.humidity : 0,
        rainfall: data.rain !== undefined && data.rain !== null ? data.rain : 0,
        wind_speed: data.wind !== undefined && data.wind !== null ? data.wind : 0,
        lat: position.lat,
        lng: position.lng,
        description: `Current conditions at ${position.lat.toFixed(2)}, ${position.lng.toFixed(2)}`
      };
      
      console.log('✅ Mapped Weather Data:', mappedWeather); // Debug log
      console.log('Rainfall value:', data.rain, '→', mappedWeather.rainfall); // Debug rainfall specifically
      setWeather(mappedWeather);
      
      // Store weather in localStorage for other modules to use
      localStorage.setItem('currentWeather', JSON.stringify(mappedWeather));
      console.log('✅ Weather stored in localStorage for auto-fill');
    } catch (error) {
      console.error('❌ Error fetching weather:', error);
      console.error('Error details:', error.response?.data || error.message);
      
      // Fallback to dummy data for demo
      const fallbackWeather = {
        temperature: 28,
        humidity: 65,
        rainfall: 2.5,
        wind_speed: 12,
        lat: position.lat,
        lng: position.lng,
        description: 'Demo data - API unavailable'
      };
      setWeather(fallbackWeather);
      localStorage.setItem('currentWeather', JSON.stringify(fallbackWeather));
    }
    setLoading(false);
  };

  // Get user's current location
  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (geoPosition) => {
        const { latitude, longitude } = geoPosition.coords;
        const newPosition = { lat: latitude, lng: longitude };
        setPosition(newPosition);
        
        // Automatically fetch weather for current location
        try {
          const data = await weatherService.getWeather(latitude, longitude);
          const mappedWeather = {
            temperature: data.temp !== undefined && data.temp !== null ? data.temp : 0,
            humidity: data.humidity !== undefined && data.humidity !== null ? data.humidity : 0,
            rainfall: data.rain !== undefined && data.rain !== null ? data.rain : 0,
            wind_speed: data.wind !== undefined && data.wind !== null ? data.wind : 0,
            lat: latitude,
            lng: longitude,
            description: 'Current location'
          };
          
          setWeather(mappedWeather);
          localStorage.setItem('currentWeather', JSON.stringify(mappedWeather));
          console.log('✅ Weather fetched for current location');
        } catch (error) {
          console.error('❌ Error fetching weather for current location:', error);
        }
        setLoading(false);
      },
      (error) => {
        setLoading(false);
        alert('Unable to get your location. Please select manually on the map or search.');
        console.error('Geolocation error:', error);
      }
    );
  };

  const modules = [
    {
      title: 'Crop Recommendation',
      description: 'Get AI-powered crop suggestions',
      icon: Leaf,
      color: 'bg-green-500',
      path: '/crop-recommendation'
    },
    {
      title: 'Yield Prediction',
      description: 'Predict crop yield accurately',
      icon: TrendingUp,
      color: 'bg-blue-500',
      path: '/yield-prediction'
    },
    {
      title: 'Fertilizer Guide',
      description: 'Optimize fertilizer usage',
      icon: Droplet,
      color: 'bg-purple-500',
      path: '/fertilizer'
    },
    {
      title: 'Stress Prediction',
      description: 'Monitor crop stress levels',
      icon: AlertTriangle,
      color: 'bg-yellow-500',
      path: '/stress-prediction'
    },
    {
      title: 'Best Spray Time',
      description: 'Optimal spraying schedule',
      icon: Sprout,
      color: 'bg-teal-500',
      path: '/spray-time'
    },
    {
      title: 'Fruit Disease',
      description: 'Detect fruit diseases',
      icon: ImageIcon,
      color: 'bg-red-500',
      path: '/fruit-disease'
    },
    {
      title: 'Leaf Disease',
      description: 'Identify leaf diseases',
      icon: ImageIcon,
      color: 'bg-orange-500',
      path: '/leaf-disease'
    },
    {
      title: 'AI Chatbot',
      description: 'Agriculture assistant',
      icon: MessageSquare,
      color: 'bg-indigo-500',
      path: '/chatbot'
    }
  ];

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Dashboard</h1>
          <p className="text-gray-600">Select your location and explore agricultural insights</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Map Section */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <MapPin className="w-6 h-6 mr-2 text-primary-600" />
              Select Location
            </h2>
            
            {/* Search Bar */}
            <div className="mb-4 relative">
              <div className="flex gap-2 mb-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search for a location (e.g., Mumbai, Delhi, Bangalore...)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onFocus={() => searchResults.length > 0 && setShowResults(true)}
                    className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setSearchResults([]);
                        setShowResults(false);
                      }}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                  {searching && (
                    <div className="absolute right-10 top-1/2 transform -translate-y-1/2">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
                    </div>
                  )}
                </div>
                <button
                  onClick={getCurrentLocation}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 flex items-center gap-2"
                  title="Use my current location"
                >
                  📍 Use My Location
                </button>
              </div>

              {/* Search Results Dropdown */}
              {showResults && searchResults.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      onClick={() => handleLocationSelect(result)}
                      className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                    >
                      <div className="flex items-start">
                        <MapPin className="w-4 h-4 mr-2 mt-1 text-primary-600 flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-800">{result.display_name}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            Lat: {parseFloat(result.lat).toFixed(4)}, Lng: {parseFloat(result.lon).toFixed(4)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="mb-4">
              <MapContainer
                center={[position.lat, position.lng]}
                zoom={10}
                style={{ height: '300px', width: '100%' }}
                className="rounded-lg"
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <MapViewController center={position} zoom={13} />
                <LocationMarker position={position} setPosition={setPosition} />
              </MapContainer>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <InputField
                label="Latitude"
                name="latitude"
                type="number"
                value={position.lat.toFixed(4)}
                onChange={(e) => setPosition({ ...position, lat: parseFloat(e.target.value) || 0 })}
                step="0.0001"
              />
              <InputField
                label="Longitude"
                name="longitude"
                type="number"
                value={position.lng.toFixed(4)}
                onChange={(e) => setPosition({ ...position, lng: parseFloat(e.target.value) || 0 })}
                step="0.0001"
              />
            </div>

            <button
              onClick={handleGetWeather}
              disabled={loading}
              className="w-full btn-primary"
            >
              {loading ? 'Loading...' : 'Get Weather & Recommendations'}
            </button>
          </div>

          {/* Weather Section */}
          <div>
            {loading ? (
              <div className="card h-full flex items-center justify-center">
                <LoadingSpinner text="Fetching weather data..." />
              </div>
            ) : (
              <WeatherCard weather={weather} />
            )}
          </div>
        </div>

        {/* Modules Grid */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Agricultural Tools</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {modules.map((module, index) => {
              const Icon = module.icon;
              return (
                <div
                  key={index}
                  onClick={() => navigate(module.path)}
                  className="card cursor-pointer hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200"
                >
                  <div className={`${module.color} w-12 h-12 rounded-lg flex items-center justify-center mb-3`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-1">{module.title}</h3>
                  <p className="text-sm text-gray-600">{module.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
