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
    <div className={`border-2 ${config.border} ${config.bg} rounded-lg p-5 w-full max-w-2xl`}>
      {/* Header with Icon and Title */}
      <div className="flex items-center gap-3 mb-4">
        <Icon className={`w-6 h-6 ${config.iconColor} flex-shrink-0`} />
        <h3 className={`text-lg font-semibold ${config.text}`}>{title}</h3>
      </div>

      {/* Content */}
      {typeof result === 'string' ? (
        <p className={`${config.text} text-base whitespace-normal`}>{result}</p>
      ) : typeof result === 'object' ? (
        <div className="flex flex-col gap-3">
          {Object.entries(result).map(([key, value]) => {
            // Skip fields that are internal or should be displayed separately
            const skipFields = [
              'warnings', 'hasWarnings', 'actionRequired', 'interpretation', 
              'top3', 'alternatives', 'fullClass', 'class', 'prediction'
            ];
            if (skipFields.includes(key)) {
              return null;
            }
            // Skip rendering if value is an object or array
            if (typeof value === 'object' && value !== null) {
              return null;
            }
            
            // Format the display value (already clean from backend)
            let displayValue = value;
            
            return (
              <div key={key} className="flex justify-between items-flex-start gap-3">
                <span className={`${config.text} font-semibold capitalize flex-shrink-0 min-w-[120px]`}>
                  {key.replace(/_/g, ' ')}:
                </span>
                <span className={`${config.text} text-right flex-1 whitespace-normal break-words leading-relaxed`}>
                  {typeof displayValue === 'number' ? displayValue.toFixed(2) : displayValue}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <pre className={`${config.text} text-sm whitespace-pre-wrap overflow-auto max-h-96`}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
};

export default ResultCard;
