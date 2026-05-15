import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { diseaseService } from '../services/services';
import { ImageIcon, AlertCircle } from 'lucide-react';

// Supported fruits in trained model
const SUPPORTED_FRUITS = ['Apple', 'Mango', 'Guava', 'Pomegranate'];

const FruitDisease = () => {
  const [selectedFruit, setSelectedFruit] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResult(null);
    setError('');
  };

  // Extract fruit and disease from prediction label
  // Format: "Disease_Fruit" e.g., "Anthracnose_Mango", "Healthy_Apple"
  const extractFruitAndDisease = (label) => {
    if (!label) return { fruit: 'Unknown', disease: 'Unknown' };
    
    const parts = label.split('_');
    if (parts.length === 2) {
      const disease = parts[0];
      const fruit = parts[1];
      return { fruit, disease };
    }
    
    // Fallback: if format is different, try to extract fruit from end
    const lastPart = parts[parts.length - 1];
    const disease = parts.slice(0, -1).join('_');
    return { fruit: lastPart, disease };
  };

  // Check if predicted fruit matches selected fruit
  const doesFruitMatch = (predictedLabel, selectedFruit) => {
    const { fruit: predictedFruit } = extractFruitAndDisease(predictedLabel);
    return predictedFruit.toLowerCase() === selectedFruit.toLowerCase();
  };

  // Check if label contains any valid fruit
  const isValidFruitLabel = (label) => {
    const { fruit } = extractFruitAndDisease(label);
    return SUPPORTED_FRUITS.some(f => f.toLowerCase() === fruit.toLowerCase());
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation: Check if fruit is selected
    if (!selectedFruit || selectedFruit.trim() === '') {
      setError('Please select a fruit type.');
      return;
    }

    // Validation: Check if fruit is supported
    if (!SUPPORTED_FRUITS.includes(selectedFruit)) {
      setError('This fruit is currently not supported by the trained model.');
      return;
    }

    if (!selectedImage) {
      setError('Please upload an image');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await diseaseService.classifyFruitDisease(formData);
      
      // Handle V2 API response structure
      if (response.success && response.data) {
        const data = response.data;
        
        // Check for low confidence prediction
        if (data.is_low_confidence === true) {
          setResult({
            isLowConfidence: true,
            confidence: `${(data.confidence * 100).toFixed(1)}%`,
            confidenceValue: data.confidence,
            message: data.message || "Low confidence detected. Please upload a clearer and valid fruit image."
          });
          return;
        }
        
        const predictionLabel = data.prediction || 'Unknown';
        
        // Validate: Check if predicted label contains a valid fruit
        if (!isValidFruitLabel(predictionLabel)) {
          setError('Invalid image. Please upload a valid fruit image.');
          return;
        }

        // Validate: Check if predicted fruit matches selected fruit
        if (!doesFruitMatch(predictionLabel, selectedFruit)) {
          setError('The uploaded image does not match the selected fruit.');
          return;
        }

        // Extract fruit and disease from prediction
        const { fruit: predictedFruit, disease: predictedDisease } = extractFruitAndDisease(predictionLabel);
        
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
        
        // Create friendly disease display name
        const getFriendlyDiseaseName = (disease) => {
          if (disease === 'Healthy') {
            return 'Healthy (No Disease Detected)';
          }
          return disease.replace(/_/g, ' ');
        };
        
        // Transform to expected format for ResultCard
        setResult({
          isLowConfidence: false,
          fruit: predictedFruit,
          disease: predictedDisease,
          diseaseName: getFriendlyDiseaseName(predictedDisease),
          isHealthy: predictedDisease === 'Healthy',
          confidence: `${(data.confidence * 100).toFixed(1)}%`,
          confidenceValue: data.confidence,
          severity: diseaseInfo.severity || 'Unknown',
          treatment: diseaseInfo.treatment || 'No treatment information available',
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
      
      // Check if error is from unclear image
      if (err.response?.data?.detail?.includes('unclear')) {
        setError('Unable to detect disease clearly. Try another image.');
      } else {
        setError(
          err.response?.data?.detail || 
          err.message || 
          'Failed to detect fruit disease. Please ensure the backend is running and try again.'
        );
      }
    }
    setLoading(false);
  };

  const handleReset = () => {
    setSelectedFruit('');
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
                  {/* Fruit Selection Dropdown */}
                  <div className="mb-6">
                    <label htmlFor="fruitSelect" className="block text-sm font-semibold text-gray-700 mb-2">
                      Select Fruit Type
                    </label>
                    <select
                      id="fruitSelect"
                      value={selectedFruit}
                      onChange={(e) => {
                        setSelectedFruit(e.target.value);
                        setError('');
                      }}
                      disabled={loading}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      <option value="">-- Select a fruit --</option>
                      {SUPPORTED_FRUITS.map((fruit) => (
                        <option key={fruit} value={fruit}>
                          {fruit}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Supported fruits: {SUPPORTED_FRUITS.join(', ')}
                    </p>
                  </div>

                  <ImageUploader
                    onImageSelect={handleImageSelect}
                    label="Upload Fruit Image"
                    accept="image/png, image/jpeg, image/jpg"
                  />

                  {error && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start">
                      <AlertCircle className="w-5 h-5 text-red-600 mr-2 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-red-600">{error}</p>
                    </div>
                  )}

                  <div className="flex space-x-3 mt-6">
                    <button 
                      type="submit" 
                      disabled={loading || !selectedImage || !selectedFruit} 
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
                  <li>• Select the fruit type first</li>
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
                    {/* Low Confidence Warning */}
                    {result.isLowConfidence ? (
                      <div className="card bg-yellow-50 border-2 border-yellow-400">
                        <div className="flex items-start">
                          <AlertCircle className="w-6 h-6 text-yellow-600 mr-3 flex-shrink-0 mt-0.5" />
                          <div className="flex-1">
                            <h3 className="text-sm font-bold text-yellow-800 mb-2">
                              ⚠️ Low Confidence Detection
                            </h3>
                            <p className="text-sm text-yellow-700 mb-3">
                              {result.message}
                            </p>
                            <div className="bg-yellow-100 rounded p-2">
                              <p className="text-xs text-yellow-700">
                                <strong>Confidence:</strong> {result.confidence}
                              </p>
                              <p className="text-xs text-yellow-700 mt-1">
                                The model is not confident enough to make a reliable prediction. Please try uploading a clearer image.
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        {/* Result Summary Card */}
                        <div className={`card ${result.isHealthy ? 'bg-green-50 border border-green-200' : 'bg-orange-50 border border-orange-200'}`}>
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <h3 className={`text-sm font-semibold ${result.isHealthy ? 'text-green-800' : 'text-orange-800'}`}>
                                {result.isHealthy ? '✓ Healthy' : '⚠ Disease Detected'}
                              </h3>
                            </div>
                          </div>
                          
                          <div className="space-y-3">
                            {/* Fruit Name */}
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Fruit:</span>
                              <span className="text-sm font-semibold text-gray-900">{result.fruit}</span>
                            </div>
                            
                            {/* Disease Name */}
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-600">Status:</span>
                              <span className={`text-sm font-semibold ${result.isHealthy ? 'text-green-700' : 'text-orange-700'}`}>
                                {result.diseaseName}
                              </span>
                            </div>
                            
                            {/* Confidence Score */}
                            <div className="flex justify-between items-center pt-2 border-t">
                              <span className="text-sm text-gray-600">Confidence:</span>
                              <div className="flex items-center space-x-2">
                                <span className="text-sm font-semibold text-gray-900">{result.confidence}</span>
                                <div className="w-16 bg-gray-200 rounded-full h-2">
                                  <div
                                    className={`h-2 rounded-full transition-all ${
                                      result.confidenceValue >= 0.8 
                                        ? 'bg-green-500' 
                                        : result.confidenceValue >= 0.6 
                                        ? 'bg-yellow-500' 
                                        : 'bg-orange-500'
                                    }`}
                                    style={{ width: `${result.confidenceValue * 100}%` }}
                                  />
                                </div>
                              </div>
                            </div>

                            {/* Treatment Info */}
                            {!result.isHealthy && result.treatment && (
                              <div className="pt-2 border-t">
                                <p className="text-xs font-semibold text-gray-700 mb-1">Treatment:</p>
                                <p className="text-xs text-gray-600 line-clamp-3">{result.treatment}</p>
                              </div>
                            )}

                            {result.isHealthy && (
                              <div className="pt-2 border-t">
                                <p className="text-xs text-green-700 font-medium">The fruit is healthy.</p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Interpretation Card */}
                        {result.interpretation && (
                          <div className="card mt-4 bg-blue-50 border border-blue-200">
                            <h4 className="text-sm font-semibold text-blue-800 mb-2">📊 Analysis</h4>
                            <p className="text-xs text-blue-700">{result.interpretation}</p>
                          </div>
                        )}

                        {/* Warnings Card */}
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

                        {/* Action Required Card */}
                        {result.actionRequired && result.actionRequired !== 'No special action required' && (
                          <div className="card mt-4 bg-purple-50 border border-purple-200">
                            <h4 className="text-sm font-semibold text-purple-800 mb-2">🎯 Action Required</h4>
                            <p className="text-xs text-purple-700 font-medium">{result.actionRequired}</p>
                          </div>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center text-sm">
                      Select a fruit type and upload an image to detect diseases
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
