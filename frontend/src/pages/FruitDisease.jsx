import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { diseaseService } from '../services/services';
import { ImageIcon } from 'lucide-react';

const FruitDisease = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResult(null);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedImage) {
      setError('Please upload an image');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);  // V2 API expects 'file' not 'image'

      const response = await diseaseService.classifyFruitDisease(formData);
      
      // Handle V2 API response structure
      if (response.success && response.data) {
        const data = response.data;
        const diseaseInfo = data.disease_info || {};
        
        // Format action required as user-friendly text
        const formatActionRequired = (action) => {
          const actionMap = {
            'EXPERT_REVIEW_RECOMMENDED': 'Expert review recommended',
            'FOLLOW_TREATMENT': 'Follow treatment plan',
            'MONITOR': 'Monitor condition',
            'IMMEDIATE_ACTION': 'Immediate action required',
            'NONE': 'No special action required'
          };
          return actionMap[action] || action;
        };
        
        // Transform to expected format for ResultCard
        setResult({
          disease: data.prediction || 'Unknown',
          confidence: `${(data.confidence * 100).toFixed(1)}%`,
          severity: diseaseInfo.severity || 'Unknown',
          treatment: diseaseInfo.treatment || 'No treatment information available',
          fruit: diseaseInfo.fruit || 'Unknown',
          interpretation: data.interpretation || '',
          warnings: data.warnings || [],
          hasWarnings: data.has_warnings || false,
          actionRequired: formatActionRequired(data.action_required || 'NONE'),
          top3: data.top_3 || []
        });
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Fruit disease detection error:', err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Failed to detect fruit disease. Please ensure the backend is running and try again.'
      );
    }
    setLoading(false);
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
    setError('');
  };

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
              <ImageIcon className="w-8 h-8 mr-3 text-primary-600" />
              Fruit Disease Classification
            </h1>
            <p className="text-gray-600">Upload fruit images to detect diseases</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:items-start">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  <ImageUploader
                    onImageSelect={handleImageSelect}
                    label="Upload Fruit Image"
                    accept="image/png, image/jpeg, image/jpg"
                  />

                  {error && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-600">{error}</p>
                    </div>
                  )}

                  <div className="flex space-x-3 mt-6">
                    <button 
                      type="submit" 
                      disabled={loading || !selectedImage} 
                      className="btn-primary flex-1"
                    >
                      {loading ? 'Analyzing...' : 'Classify Disease'}
                    </button>
                    <button 
                      type="button" 
                      onClick={handleReset} 
                      className="btn-secondary"
                    >
                      Reset
                    </button>
                  </div>
                </form>
              </div>

              <div className="card mt-6 bg-blue-50 border border-blue-200">
                <h3 className="text-lg font-semibold text-blue-800 mb-3">Tips for Best Results</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Use clear, well-lit images</li>
                  <li>• Focus on the affected area</li>
                  <li>• Avoid blurry or distant shots</li>
                  <li>• Capture different angles if unsure</li>
                  <li>• Supported fruits: Apple, Guava, Mango, Pomegranate</li>
                  <li>• Single fruit per image for best accuracy</li>
                </ul>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="lg:sticky lg:top-24 self-start">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Analyzing image..." />
                  </div>
                ) : result ? (
                  <>
                    <ResultCard
                      result={result}
                      type={result.disease?.includes('Healthy') ? 'success' : 'warning'}
                      title="Disease Detection"
                      icon={ImageIcon}
                    />
                    
                    {result.hasWarnings && result.warnings && result.warnings.length > 0 && (
                      <div className="card mt-4 bg-yellow-50 border border-yellow-200">
                        <h4 className="text-sm font-semibold text-yellow-800 mb-2">⚠️ Warnings</h4>
                        <ul className="space-y-1">
                          {result.warnings.map((warning, idx) => (
                            <li key={idx} className="text-xs text-yellow-700">• {warning}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {result.interpretation && (
                      <div className="card mt-4 bg-blue-50 border border-blue-200">
                        <h4 className="text-sm font-semibold text-blue-800 mb-2">📊 Interpretation</h4>
                        <p className="text-xs text-blue-700">{result.interpretation}</p>
                      </div>
                    )}
                    
                    {result.actionRequired && result.actionRequired !== 'No special action required' && (
                      <div className="card mt-4 bg-purple-50 border border-purple-200">
                        <h4 className="text-sm font-semibold text-purple-800 mb-2">🎯 Action Required</h4>
                        <p className="text-xs text-purple-700 font-medium">{result.actionRequired}</p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center">
                      Upload an image to detect fruit diseases
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

export default FruitDisease;
