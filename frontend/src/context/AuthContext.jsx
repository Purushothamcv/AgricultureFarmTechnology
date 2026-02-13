import React, { createContext, useState, useContext, useEffect } from 'react';
import { authService } from '../services/services';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on mount
    const currentUser = authService.getCurrentUser();
    setUser(currentUser);
    setLoading(false);
  }, []);

  const login = async (credentials) => {
    try {
      console.log('🔐 AuthContext: Starting login process...');
      const data = await authService.login(credentials);
      console.log('✅ AuthContext: Login successful, setting user state');
      setUser(data.user);
      return { success: true };
    } catch (error) {
      console.error('❌ AuthContext: Login failed');
      console.error('   Error details:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Login failed. Please check your credentials.';
      console.error('   Error message:', errorMessage);
      return { 
        success: false, 
        error: errorMessage
      };
    }
  };

  const register = async (userData) => {
    try {
      console.log('📝 AuthContext: Starting registration process...');
      const data = await authService.register(userData);
      console.log('✅ AuthContext: Registration successful');
      // After registration, user needs to login
      return { success: true, message: data.message };
    } catch (error) {
      console.error('❌ AuthContext: Registration failed');
      console.error('   Error details:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Registration failed. Please try again.';
      console.error('   Error message:', errorMessage);
      return { 
        success: false, 
        error: errorMessage
      };
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
  };

  const value = {
    user,
    login,
    register,
    logout,
    isAuthenticated: !!user,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
