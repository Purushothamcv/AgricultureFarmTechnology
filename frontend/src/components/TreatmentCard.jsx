import React from 'react';
import { AlertCircle, CheckCircle, Droplet, Shield, Zap } from 'lucide-react';

/**
 * TreatmentCard - Displays detailed treatment recommendations for plant diseases
 * Shows remedy, pesticide, prevention, and action guidance
 */
const TreatmentCard = ({ disease, crop, remedy, pesticide, prevention, action, confidence, isHealthy, source }) => {
  
  if (!remedy || !pesticide || !prevention || !action) {
    return null;
  }

  return (
    <div className="mt-8 space-y-6">
      {/* Header */}
      <div className="border-b-2 border-green-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          {isHealthy ? (
            <>
              <CheckCircle className="w-6 h-6 text-green-500" />
              Plant Health Status
            </>
          ) : (
            <>
              <AlertCircle className="w-6 h-6 text-orange-500" />
              Recommended Treatment
            </>
          )}
        </h2>
        {!isHealthy && confidence && (
          <p className="text-sm text-gray-600 mt-1">
            Detection Confidence: <span className="font-semibold">{(confidence * 100).toFixed(1)}%</span>
          </p>
        )}
        {source && (
          <p className="text-xs text-gray-500 mt-1">
            Information source: <span className="font-medium capitalize">{source}</span>
          </p>
        )}
      </div>

      {/* Treatment Cards Grid - Equal Heights with Responsive Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 auto-rows-fr">
        {/* Remedy Card */}
        <div className="flex flex-col h-full bg-gradient-to-br from-blue-50 to-blue-100 border-l-4 border-blue-500 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-blue-600 flex-shrink-0" />
            <h3 className="text-lg font-semibold text-blue-900">Remedy</h3>
          </div>
          <p className="text-blue-800 leading-relaxed text-sm flex-1">
            {remedy}
          </p>
        </div>

        {/* Pesticide Card */}
        <div className="flex flex-col h-full bg-gradient-to-br from-amber-50 to-amber-100 border-l-4 border-amber-500 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center gap-3 mb-4">
            <Droplet className="w-6 h-6 text-amber-600 flex-shrink-0" />
            <h3 className="text-lg font-semibold text-amber-900">Suggested Pesticide / Fungicide</h3>
          </div>
          <p className="text-amber-800 leading-relaxed text-sm flex-1">
            {pesticide}
          </p>
        </div>

        {/* Prevention Card */}
        <div className="flex flex-col h-full bg-gradient-to-br from-green-50 to-green-100 border-l-4 border-green-500 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-green-600 flex-shrink-0" />
            <h3 className="text-lg font-semibold text-green-900">Prevention</h3>
          </div>
          <p className="text-green-800 leading-relaxed text-sm flex-1">
            {prevention}
          </p>
        </div>

        {/* Action Card */}
        <div className="flex flex-col h-full bg-gradient-to-br from-purple-50 to-purple-100 border-l-4 border-purple-500 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-6 h-6 text-purple-600 flex-shrink-0" />
            <h3 className="text-lg font-semibold text-purple-900">Action to Take</h3>
          </div>
          <p className="text-purple-800 leading-relaxed text-sm flex-1">
            {action}
          </p>
        </div>
      </div>

      {/* Additional Info */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <p className="text-xs text-gray-600">
          <strong>Note:</strong> These recommendations are based on {source === 'database' ? 'comprehensive agricultural databases' : source === 'llm' ? 'AI-powered agricultural expertise' : 'general agricultural practices'}. For specific region-specific guidance or if symptoms persist, contact your local agricultural extension office.
        </p>
      </div>
    </div>
  );
};

export default TreatmentCard;
