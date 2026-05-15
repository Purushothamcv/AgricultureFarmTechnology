import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import ImageUploader from '../components/ImageUploader';
import LoadingSpinner from '../components/LoadingSpinner';
import { fruitDetectionService } from '../services/services';
import { ImageIcon, CheckCircle, AlertCircle, XCircle } from 'lucide-react';

const FruitDetectionDetailed = () => {
  const [supportedFruits, setSupportedFruits] = useState([]);
  const [selectedFruit, setSelectedFruit] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [loadingFruits, setLoadingFruits] = useState(true);

  // Load supported fruits on component mount
  useEffect(() => {
    fetchSupportedFruits();
  }, []);

  const fetchSupportedFruits = async () => {
    try {
      setLoadingFruits(true);
      const response = await fruitDetectionService.getSupportedFruits();
      if (response.success && response.data?.fruits) {
        setSupportedFruits(response.data.fruits);
        setError('');
      }
    } catch (err) {
      console.error('Error fetching supported fruits:', err);
      setError('Failed to load supported fruits. Please refresh the page.');
    } finally {
      setLoadingFruits(false);
    }
  };

  const handleImageSelect = (file) => {
    if (file) {
      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      setResult(null);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    if (!selectedFruit || selectedFruit.trim() === '') {
      setError('Please select a fruit type from the dropdown');
      setResult(null);
      return;
    }

    if (!selectedImage) {
      setError('Please upload an image');
      setResult(null);
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('fruit_type', selectedFruit);
      formData.append('file', selectedImage);
      formData.append('confidence_threshold', 0.50);

      const response = await fruitDetectionService.predictWithSelection(formData);

      if (response.success && response.data) {
        const data = response.data;
        
        // Transform to display format
        setResult({
          selectedFruit: data.selected_fruit,
          disease: data.prediction,
          confidence: `${(data.confidence * 100).toFixed(1)}%`,
          confidenceDecimal: data.confidence,
          interpretation: data.interpretation,
          warnings: data.warnings || [],
          hasWarnings: data.has_warnings,
          actionRequired: data.action_required,
          imagePreview: imagePreview,
          success: true
        });
        setError('');
      } else if (response.error) {
        setError(response.error);
        if (response.data?.supported_fruits) {
          console.log('Supported fruits:', response.data.supported_fruits);
        }
        setResult(null);
      }
    } catch (err) {
      console.error('Fruit disease detection error:', err);
      const errorMessage = err.response?.data?.error || 
                          err.message || 
                          'Failed to detect fruit disease. Please ensure the backend is running and try again.';
      setError(errorMessage);
      setResult(null);
    }
    setLoading(false);
  };

  const handleReset = () => {
    setSelectedFruit('');
    setSelectedImage(null);
    setImagePreview(null);
    setResult(null);
    setError('');
  };

  // Confidence color based on value
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-orange-600';
  };

  // Get severity indicator
  const getSeverityDisplay = (disease) => {
    if (disease.includes('Healthy')) {
      return { label: 'Healthy', color: 'bg-green-100 text-green-800', icon: CheckCircle };
    }
    return { label: 'Disease Detected', color: 'bg-red-100 text-red-800', icon: AlertCircle };
  };

  const severityInfo = result ? getSeverityDisplay(result.disease) : null;
  const SeverityIcon = severityInfo?.icon;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <Navbar />
      
      <div className="page-content py-8">
        <div className="max-w-5xl mx-auto px-4">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center">
              <ImageIcon className="w-10 h-10 mr-3 text-indigo-600" />
              Fruit Disease Detection
            </h1>
            <p className="text-gray-600 text-lg">
              Select a fruit type and upload an image to detect diseases
            </p>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Input Form */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-6">Disease Detection</h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Fruit Selection Dropdown */}
                <div>
                  <label htmlFor="fruitSelect" className="block text-sm font-medium text-gray-700 mb-2">
                    Select Fruit Type *
                  </label>
                  <select
                    id="fruitSelect"
                    value={selectedFruit}
                    onChange={(e) => {
                      setSelectedFruit(e.target.value);
                      setError('');
                    }}
                    disabled={loadingFruits}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 bg-white text-gray-700 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="">-- Select a fruit --</option>
                    {supportedFruits.map((fruit) => (
                      <option key={fruit} value={fruit}>
                        {fruit}
                      </option>
                    ))}
                  </select>
                  {loadingFruits && (
                    <p className="text-sm text-gray-500 mt-2">Loading available fruits...</p>
                  )}
                  <p className="text-xs text-gray-500 mt-2">
                    Supported fruits: {supportedFruits.join(', ')}
                  </p>
                </div>

                {/* Image Upload */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Fruit Image *
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-indigo-400 transition">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => {
                        if (e.target.files?.[0]) {
                          handleImageSelect(e.target.files[0]);
                        }
                      }}
                      className="hidden"
                      id="imageInput"
                      disabled={loading}
                    />
                    <label htmlFor="imageInput" className="cursor-pointer">
                      <div className="text-indigo-600 text-3xl mb-2">📸</div>
                      <p className="text-sm text-gray-600">
                        {selectedImage ? selectedImage.name : 'Click to upload or drag and drop'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">PNG, JPG, JPEG up to 10MB</p>
                    </label>
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
                    <XCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-red-800 font-medium">Error</p>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3">
                  <button
                    type="submit"
                    disabled={loading || !selectedFruit || !selectedImage}
                    className="flex-1 bg-indigo-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
                  >
                    {loading ? 'Detecting...' : 'Detect Disease'}
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    disabled={loading}
                    className="flex-1 bg-gray-200 text-gray-800 py-3 px-4 rounded-lg font-medium hover:bg-gray-300 disabled:cursor-not-allowed transition"
                  >
                    Reset
                  </button>
                </div>
              </form>

              {/* Loading Spinner */}
              {loading && (
                <div className="mt-6 flex justify-center">
                  <LoadingSpinner />
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div className="space-y-6">
              {/* Image Preview */}
              {imagePreview && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Image Preview</h3>
                  <div className="aspect-square rounded-lg overflow-hidden border border-gray-200">
                    <img 
                      src={imagePreview} 
                      alt="Selected fruit" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              )}

              {/* Results Display */}
              {result && result.success && (
                <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-gray-800">Detection Results</h3>
                    {SeverityIcon && (
                      <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${severityInfo.color}`}>
                        <SeverityIcon className="w-4 h-4" />
                        <span className="text-sm font-medium">{severityInfo.label}</span>
                      </div>
                    )}
                  </div>

                  <div className="space-y-3 border-t pt-4">
                    {/* Selected Fruit */}
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Selected Fruit:</span>
                      <span className="font-semibold text-gray-900">{result.selectedFruit}</span>
                    </div>

                    {/* Disease Name */}
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Detection Result:</span>
                      <span className="font-semibold text-indigo-600">{result.disease}</span>
                    </div>

                    {/* Confidence Score */}
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Confidence:</span>
                      <div className="flex items-center gap-2">
                        <span className={`font-semibold text-lg ${getConfidenceColor(result.confidenceDecimal)}`}>
                          {result.confidence}
                        </span>
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              result.confidenceDecimal >= 0.8 
                                ? 'bg-green-500' 
                                : result.confidenceDecimal >= 0.6 
                                ? 'bg-yellow-500' 
                                : 'bg-orange-500'
                            }`}
                            style={{ width: `${result.confidenceDecimal * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Interpretation */}
                    {result.interpretation && (
                      <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
                        <p className="text-sm text-blue-900">
                          <strong>Analysis:</strong> {result.interpretation}
                        </p>
                      </div>
                    )}

                    {/* Warnings */}
                    {result.hasWarnings && result.warnings.length > 0 && (
                      <div className="mt-3 p-3 bg-yellow-50 rounded border border-yellow-200">
                        <p className="text-sm font-medium text-yellow-900 mb-2">⚠️ Warnings:</p>
                        <ul className="text-sm text-yellow-800 space-y-1">
                          {result.warnings.map((warning, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="mr-2">•</span>
                              <span>{warning}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Action Required */}
                    {result.actionRequired && result.actionRequired !== 'NONE' && (
                      <div className="mt-3 p-3 bg-red-50 rounded border border-red-200">
                        <p className="text-sm font-medium text-red-900">
                          📋 Action Required: {result.actionRequired}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* No Results Message */}
              {!result && !loading && !imagePreview && (
                <div className="bg-gray-50 rounded-lg border border-dashed border-gray-300 p-8 text-center">
                  <p className="text-gray-500">
                    Upload an image and select a fruit type to see detection results
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Info Section */}
          <div className="mt-12 bg-indigo-50 rounded-lg border border-indigo-200 p-6">
            <h3 className="text-lg font-semibold text-indigo-900 mb-3">ℹ️ How to Use</h3>
            <ol className="space-y-2 text-indigo-800 text-sm">
              <li><strong>1.</strong> Select the fruit type from the dropdown</li>
              <li><strong>2.</strong> Upload a clear image of the fruit</li>
              <li><strong>3.</strong> Click "Detect Disease" to analyze</li>
              <li><strong>4.</strong> Review the results and recommendations</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FruitDetectionDetailed;
