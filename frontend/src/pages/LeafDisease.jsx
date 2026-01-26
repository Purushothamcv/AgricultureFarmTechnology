import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { diseaseService } from '../services/services';
import { ImageIcon } from 'lucide-react';

const LeafDisease = () => {
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
      formData.append('file', selectedImage);

      const data = await diseaseService.detectLeafDisease(formData);
      
      // Backend now returns cleaned names directly
      // { crop, disease, confidence, severity, warning, top_3 }
      setResult({
        crop: data.crop,
        disease: data.disease,
        confidence: `${(data.confidence * 100).toFixed(1)}%`,
        severity: data.severity,
        warning: data.warning || null,
        alternatives: data.top_3 ? data.top_3.slice(1).map(p => ({
          name: `${p.crop} - ${p.disease}`,
          confidence: `${(p.confidence * 100).toFixed(1)}%`
        })) : []
      });
    } catch (err) {
      console.error('Error detecting disease:', err);
      setError('Failed to analyze image. Please ensure the backend server is running on port 8000.');
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
              Plant Leaf Disease Detection
            </h1>
            <p className="text-gray-600">Upload leaf images to identify diseases</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="card">
                <form onSubmit={handleSubmit}>
                  <ImageUploader
                    onImageSelect={handleImageSelect}
                    label="Upload Plant Leaf Image"
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
                      {loading ? 'Detecting...' : 'Detect Disease'}
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

              <div className="card mt-6 bg-green-50 border border-green-200">
                <h3 className="text-lg font-semibold text-green-800 mb-3">Supported Crops</h3>
                <div className="grid grid-cols-2 gap-2 text-sm text-gray-700">
                  <div>• Apple</div>
                  <div>• Blueberry</div>
                  <div>• Cherry</div>
                  <div>• Corn (Maize)</div>
                  <div>• Grape</div>
                  <div>• Orange</div>
                  <div>• Peach</div>
                  <div>• Pepper (Bell)</div>
                  <div>• Potato</div>
                  <div>• Raspberry</div>
                  <div>• Soybean</div>
                  <div>• Squash</div>
                  <div>• Strawberry</div>
                  <div>• Tomato</div>
                </div>
              </div>

              <div className="card mt-6 bg-blue-50 border border-blue-200">
                <h3 className="text-lg font-semibold text-blue-800 mb-3">Image Guidelines</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Capture leaf against plain background</li>
                  <li>• Ensure good lighting (natural light preferred)</li>
                  <li>• Include the entire leaf or affected area</li>
                  <li>• Avoid shadows and reflections</li>
                  <li>• Image should be in focus</li>
                </ul>
              </div>
            </div>

            <div className="lg:col-span-1">
              <div className="sticky top-24">
                {loading ? (
                  <div className="card">
                    <LoadingSpinner text="Analyzing leaf..." />
                  </div>
                ) : result ? (
                  <ResultCard
                    result={result}
                    type={result.disease?.toLowerCase().includes('healthy') ? 'success' : 'warning'}
                    title="Disease Detection"
                    icon={ImageIcon}
                  />
                ) : (
                  <div className="card bg-gray-50">
                    <p className="text-gray-500 text-center">
                      Upload a leaf image to detect diseases
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

export default LeafDisease;
