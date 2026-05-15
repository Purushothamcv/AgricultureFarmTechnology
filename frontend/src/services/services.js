import api from './api';

export const authService = {
  async login(credentials) {
    try {
      console.log('🔐 Attempting login for:', credentials.email);
      const response = await api.post('/auth/login', credentials);
      console.log('✅ Login response:', response.data);
      
      // Store JWT token and user info
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        console.log('🔑 JWT token stored');
      }
      if (response.data.user) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
        console.log('💾 User stored in localStorage:', response.data.user);
      }
      return response.data;
    } catch (error) {
      console.error('❌ Login failed:', error);
      console.error('   Error response:', error.response?.data);
      console.error('   Status code:', error.response?.status);
      throw error;
    }
  },

  async googleLogin(credential) {
    try {
      console.log('🔐 Attempting Google OAuth login...');
      const response = await api.post('/auth/google', { credential });
      console.log('✅ Google login response:', response.data);
      
      // Store JWT token and user info
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        console.log('🔑 JWT token stored');
      }
      if (response.data.user) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
        console.log('💾 User stored in localStorage:', response.data.user);
      }
      return response.data;
    } catch (error) {
      console.error('❌ Google login failed:', error);
      console.error('   Error response:', error.response?.data);
      console.error('   Status code:', error.response?.status);
      throw error;
    }
  },

  async register(userData) {
    try {
      console.log('📝 Attempting registration for:', userData.email);
      console.log('   User data:', { name: userData.name, email: userData.email });
      const response = await api.post('/auth/register', userData);
      console.log('✅ Registration response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Registration failed:', error);
      console.error('   Error response:', error.response?.data);
      console.error('   Status code:', error.response?.status);
      throw error;
    }
  },

  logout() {
    console.log('🚪 Logging out user');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  getCurrentUser() {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        console.log('👤 Current user from storage:', user.email);
        return user;
      } catch (e) {
        console.error('❌ Failed to parse user from localStorage:', e);
        localStorage.removeItem('user');
        return null;
      }
    }
    return null;
  },

  getToken() {
    return localStorage.getItem('token');
  },

  isAuthenticated() {
    return !!this.getCurrentUser();
  }
};

export const weatherService = {
  async getWeather(lat, lon) {
    const response = await api.get(`/api/weather?lat=${lat}&lon=${lon}`);
    return response.data;
  }
};

export const cropService = {
  async recommendCrop(data) {
    const payload = {
      nitrogen: parseFloat(data.nitrogen) || 0,
      phosphorus: parseFloat(data.phosphorus) || 0,
      potassium: parseFloat(data.potassium) || 0,
      temperature: parseFloat(data.temperature) || 0,
      humidity: parseFloat(data.humidity) || 0,
      ph: parseFloat(data.ph) || 0,
      rainfall: parseFloat(data.rainfall) || 0,
      ozone: parseFloat(data.ozone) || 0
    };
    const response = await api.post('/predict/manual', payload);
    return response.data;
  },

  async recommendCropByLocation(data) {
    const payload = {
      latitude: parseFloat(data.latitude),
      longitude: parseFloat(data.longitude),
      nitrogen: data.nitrogen ? parseFloat(data.nitrogen) : null,
      phosphorus: data.phosphorus ? parseFloat(data.phosphorus) : null,
      potassium: data.potassium ? parseFloat(data.potassium) : null,
      temperature: data.temperature ? parseFloat(data.temperature) : null,
      humidity: data.humidity ? parseFloat(data.humidity) : null,
      ph: data.ph ? parseFloat(data.ph) : null,
      rainfall: data.rainfall ? parseFloat(data.rainfall) : null,
      ozone: data.ozone ? parseFloat(data.ozone) : null
    };
    const response = await api.post('/predict/location', payload);
    return response.data;
  },

  async fetchLocationData(latitude, longitude) {
    const response = await api.get(`/api/location-data?latitude=${latitude}&longitude=${longitude}`);
    return response.data;
  },

  async predictYield(data) {
    const response = await api.post('/yield/predict', data);
    return response.data;
  },

  async predictStress(data) {
    const response = await api.post('/stress/predict', data);
    return response.data;
  },

  async getBestSprayTime(data) {
    const response = await api.post('/spray/recommend', data);
    return response.data;
  }
};

export const fertilizerService = {
  async recommendFertilizer(data) {
    const response = await api.post('/fertilizer/recommend', data);
    return response.data;
  }
};

export const diseaseService = {
  async classifyFruitDisease(formData) {
    // CORRECTED: Use V2 endpoint with biologically correct predictions
    const response = await api.post('/api/v2/fruit-disease/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    return response.data;
  },

  async detectLeafDisease(formData) {
    const response = await api.post('/predict/plant-disease', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    return response.data;
  },

  async generateRemedy(diseaseInfo) {
    const response = await api.post('/api/generate-disease-remedy', diseaseInfo);
    return response.data;
  }
};

export const chatbotService = {
  async sendMessage(payload) {
    const response = await api.post('/chat/send', payload);
    return response.data;
  }
};

export const fruitDetectionService = {
  async getSupportedFruits() {
    try {
      console.log('🍎 Fetching supported fruits...');
      const response = await api.get('/api/fruit-disease-detection/supported-fruits');
      console.log('✅ Supported fruits loaded:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to fetch supported fruits:', error);
      throw error;
    }
  },

  async predictWithSelection(formData) {
    try {
      console.log('🔍 Sending fruit disease detection request...');
      const response = await api.post('/api/fruit-disease-detection/predict-with-selection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      console.log('✅ Prediction response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Fruit disease prediction failed:', error);
      
      // Handle error response
      if (error.response?.data) {
        return {
          success: false,
          error: error.response.data.error || 'Failed to predict disease',
          data: error.response.data.data || null
        };
      }
      
      throw error;
    }
  },

  async checkHealth() {
    try {
      const response = await api.get('/api/fruit-disease-detection/health');
      return response.data;
    } catch (error) {
      console.error('❌ Health check failed:', error);
      throw error;
    }
  }
};
