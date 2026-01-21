import React from 'react';
import { Cloud, Droplets, Wind, Thermometer } from 'lucide-react';

const WeatherCard = ({ weather }) => {
  if (!weather) {
    return (
      <div className="card">
        <p className="text-gray-500 text-center">Select a location on the map to view weather</p>
      </div>
    );
  }

  const weatherItems = [
    {
      icon: Thermometer,
      label: 'Temperature',
      value: `${weather.temperature !== undefined && weather.temperature !== null ? weather.temperature : 'N/A'}°C`,
      color: 'text-red-500'
    },
    {
      icon: Droplets,
      label: 'Humidity',
      value: `${weather.humidity !== undefined && weather.humidity !== null ? weather.humidity : 'N/A'}%`,
      color: 'text-blue-500'
    },
    {
      icon: Cloud,
      label: 'Rainfall',
      value: `${weather.rainfall !== undefined && weather.rainfall !== null ? weather.rainfall : 'N/A'} mm`,
      color: 'text-indigo-500'
    },
    {
      icon: Wind,
      label: 'Wind Speed',
      value: `${weather.wind_speed !== undefined && weather.wind_speed !== null ? weather.wind_speed : 'N/A'} km/h`,
      color: 'text-teal-500'
    }
  ];

  return (
    <div className="card">
      <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
        <Cloud className="w-6 h-6 mr-2 text-primary-600" />
        Current Weather
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {weatherItems.map((item, index) => {
          const Icon = item.icon;
          return (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <Icon className={`w-5 h-5 mr-2 ${item.color}`} />
                <span className="text-sm text-gray-600">{item.label}</span>
              </div>
              <p className="text-2xl font-bold text-gray-800">{item.value}</p>
            </div>
          );
        })}
      </div>
      {weather.description && (
        <div className="mt-4 p-3 bg-primary-50 rounded-lg">
          <p className="text-sm text-primary-700 font-medium">
            {weather.description}
          </p>
        </div>
      )}
    </div>
  );
};

export default WeatherCard;
