import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const AuthCallback = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();
  const [error, setError] = useState('');

  useEffect(() => {
    const processCallback = async () => {
      try {
        // Get URL parameters
        const params = new URLSearchParams(location.search);
        const token = params.get('token');
        const email = params.get('email');
        const name = params.get('name');
        const success = params.get('success');
        const errorMsg = params.get('error');

        console.log('🔐 Auth Callback - Processing...');
        console.log('   Token:', token ? '✓ Present' : '✗ Missing');
        console.log('   Email:', email);
        console.log('   Success:', success);
        console.log('   Error:', errorMsg);

        // Check for errors
        if (errorMsg) {
          console.error('❌ OAuth Error:', errorMsg);
          setError(`Authentication failed: ${decodeURIComponent(errorMsg)}`);
          
          // Redirect to login after 3 seconds
          setTimeout(() => {
            navigate('/login', { 
              state: { message: `Failed to authenticate: ${decodeURIComponent(errorMsg)}` } 
            });
          }, 3000);
          return;
        }

        if (!token || success !== 'true') {
          console.error('❌ Invalid callback parameters');
          setError('Invalid authentication response');
          
          setTimeout(() => {
            navigate('/login');
          }, 3000);
          return;
        }

        // Store credentials in localStorage
        console.log('💾 Storing credentials...');
        localStorage.setItem('token', token);
        
        if (email || name) {
          const user = {
            email: email || 'User',
            name: name || 'User'
          };
          localStorage.setItem('user', JSON.stringify(user));
          console.log('✅ User stored:', user.email);
        }

        console.log('✅ Auth Callback - Success! Redirecting to dashboard...');
        
        // Redirect to dashboard
        setTimeout(() => {
          navigate('/dashboard', { replace: true });
        }, 500);

      } catch (err) {
        console.error('❌ Auth Callback Error:', err);
        setError('An error occurred during authentication');
        
        setTimeout(() => {
          navigate('/login');
        }, 3000);
      }
    };

    processCallback();
  }, [location, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-primary-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full text-center">
        {error ? (
          <>
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 font-semibold">⚠️ {error}</p>
              <p className="text-red-600 text-sm mt-2">Redirecting to login...</p>
            </div>
          </>
        ) : (
          <>
            <Loader className="w-12 h-12 animate-spin mx-auto text-primary-600 mb-4" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Completing Sign-In</h2>
            <p className="text-gray-600">Verifying your authentication...</p>
          </>
        )}
      </div>
    </div>
  );
};

export default AuthCallback;
