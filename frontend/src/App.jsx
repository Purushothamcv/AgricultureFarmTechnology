import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import CropRecommendation from './pages/CropRecommendation';
import YieldPrediction from './pages/YieldPrediction';
import FertilizerRecommendation from './pages/FertilizerRecommendation';
import StressPrediction from './pages/StressPrediction';
import BestSprayTime from './pages/BestSprayTime';
import FruitDisease from './pages/FruitDisease';
import LeafDisease from './pages/LeafDisease';
import Chatbot from './pages/Chatbot';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          {/* Protected Routes */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/crop-recommendation" 
            element={
              <ProtectedRoute>
                <CropRecommendation />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/yield-prediction" 
            element={
              <ProtectedRoute>
                <YieldPrediction />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/fertilizer" 
            element={
              <ProtectedRoute>
                <FertilizerRecommendation />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/stress-prediction" 
            element={
              <ProtectedRoute>
                <StressPrediction />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/spray-time" 
            element={
              <ProtectedRoute>
                <BestSprayTime />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/fruit-disease" 
            element={
              <ProtectedRoute>
                <FruitDisease />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/leaf-disease" 
            element={
              <ProtectedRoute>
                <LeafDisease />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/chatbot" 
            element={
              <ProtectedRoute>
                <Chatbot />
              </ProtectedRoute>
            } 
          />
          
          {/* Default Route - Redirect to login */}
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
