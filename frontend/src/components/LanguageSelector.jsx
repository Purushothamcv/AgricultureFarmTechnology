/**
 * Language Selector Component
 * ============================
 * 
 * Allows users to select their preferred language for the chatbot
 * Supports: English, Hindi (हिंदी), and Kannada (ಕನ್ನಡ)
 */

import React, { useState, useRef, useEffect } from 'react';
import { Languages, Check } from 'lucide-react';

const LanguageSelector = ({ currentLanguage, onLanguageChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const languages = [
    {
      code: 'english',
      name: 'English',
      nativeName: 'English',
      flag: '🇬🇧',
      voiceCode: 'en-US'
    },
    {
      code: 'hindi',
      name: 'Hindi',
      nativeName: 'हिंदी',
      flag: '🇮🇳',
      voiceCode: 'hi-IN'
    },
    {
      code: 'kannada',
      name: 'Kannada',
      nativeName: 'ಕನ್ನಡ',
      flag: '🇮🇳',
      voiceCode: 'kn-IN'
    }
  ];

  const currentLang = languages.find(lang => lang.code === currentLanguage) || languages[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLanguageSelect = (languageCode) => {
    onLanguageChange(languageCode);
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Language Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-300 shadow-md hover:shadow-lg"
        title="Select Language"
      >
        <Languages className="w-5 h-5" />
        <span className="font-medium">{currentLang.flag} {currentLang.nativeName}</span>
        <svg
          className={`w-4 h-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-xl border border-gray-200 z-50 animate-fadeIn">
          <div className="p-2">
            <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Choose Language
            </div>
            {languages.map((lang) => (
              <button
                key={lang.code}
                onClick={() => handleLanguageSelect(lang.code)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg transition-colors duration-200 ${
                  currentLanguage === lang.code
                    ? 'bg-blue-50 text-blue-700'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{lang.flag}</span>
                  <div className="text-left">
                    <div className="font-medium">{lang.nativeName}</div>
                    <div className="text-xs text-gray-500">{lang.name}</div>
                  </div>
                </div>
                {currentLanguage === lang.code && (
                  <Check className="w-5 h-5 text-blue-600" />
                )}
              </button>
            ))}
          </div>
          
          {/* Voice Feature Info */}
          <div className="border-t border-gray-200 p-3">
            <div className="flex items-center space-x-2 text-xs text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Voice input & output available in all languages</span>
            </div>
          </div>
        </div>
      )}

      {/* CSS Animation */}
      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
        }
      `}</style>
    </div>
  );
};

export default LanguageSelector;
