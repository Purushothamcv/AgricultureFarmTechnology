import React from 'react';
import { CheckCircle, AlertCircle, Info, XCircle } from 'lucide-react';

const ResultCard = ({ result, type = 'success', title = 'Result', icon: CustomIcon }) => {
  const getConfig = () => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-green-50',
          border: 'border-green-200',
          text: 'text-green-800',
          icon: CustomIcon || CheckCircle,
          iconColor: 'text-green-600'
        };
      case 'warning':
        return {
          bg: 'bg-yellow-50',
          border: 'border-yellow-200',
          text: 'text-yellow-800',
          icon: CustomIcon || AlertCircle,
          iconColor: 'text-yellow-600'
        };
      case 'error':
        return {
          bg: 'bg-red-50',
          border: 'border-red-200',
          text: 'text-red-800',
          icon: CustomIcon || XCircle,
          iconColor: 'text-red-600'
        };
      case 'info':
      default:
        return {
          bg: 'bg-blue-50',
          border: 'border-blue-200',
          text: 'text-blue-800',
          icon: CustomIcon || Info,
          iconColor: 'text-blue-600'
        };
    }
  };

  const config = getConfig();
  const Icon = config.icon;

  if (!result) return null;

  return (
    <div className={`card border-2 ${config.border} ${config.bg}`}>
      <div className="flex items-start space-x-3">
        <Icon className={`w-6 h-6 ${config.iconColor} flex-shrink-0 mt-1`} />
        <div className="flex-1">
          <h3 className={`text-lg font-semibold ${config.text} mb-2`}>{title}</h3>
          {typeof result === 'string' ? (
            <p className={`${config.text} text-base`}>{result}</p>
          ) : typeof result === 'object' ? (
            <div className="space-y-2">
              {Object.entries(result).map(([key, value]) => {
                // Skip rendering if value is an object or array
                if (typeof value === 'object' && value !== null) {
                  return null;
                }
                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className={`${config.text} font-medium capitalize`}>
                      {key.replace(/_/g, ' ')}:
                    </span>
                    <span className={`${config.text} font-semibold`}>
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <pre className={`${config.text} text-sm whitespace-pre-wrap`}>
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
