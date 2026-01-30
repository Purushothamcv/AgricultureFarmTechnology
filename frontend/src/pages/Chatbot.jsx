/**
 * AI Chatbot Component with Voice Assistance
 * ===========================================
 * 
 * Features:
 * - Voice input using Web Speech API
 * - Voice output using Text-to-Speech
 * - Multilingual support (English, Hindi & Kannada)
 * - Real-time chat with Groq AI
 * - Agricultural expertise
 */

import React, { useState, useRef, useEffect } from 'react';
import Navbar from '../components/Navbar';
import ChatMessage from '../components/ChatMessage';
import LanguageSelector from '../components/LanguageSelector';
import { MessageCircle, Mic, MicOff, Volume2, VolumeX, Send, Trash2, Loader, Bot } from 'lucide-react';

const Chatbot = () => {
  // ============================================================================
  // STATE MANAGEMENT
  // ============================================================================
  
  const [messages, setMessages] = useState([
    {
      text: "Hello! I'm SmartAgri AI Assistant. I can help you with crops, diseases, fertilizers, and farming advice. How can I assist you today?",
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('english'); // 'english', 'hindi', or 'kannada'
  
  // Voice states
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const synthesisRef = useRef(null);

  // ============================================================================
  // VOICE RECOGNITION SETUP (Web Speech API)
  // ============================================================================
  
  useEffect(() => {
    // Check if browser supports Speech Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = language === 'hindi' ? 'hi-IN' : (language === 'kannada' ? 'kn-IN' : 'en-US');
      
      recognitionRef.current.onstart = () => {
        setIsListening(true);
      };
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(transcript);
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
    
    // Text-to-Speech setup
    synthesisRef.current = window.speechSynthesis;
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (synthesisRef.current) {
        synthesisRef.current.cancel();
      }
    };
  }, [language]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // ============================================================================
  // VOICE FUNCTIONS
  // ============================================================================
  
  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      try {
        recognitionRef.current.lang = language === 'hindi' ? 'hi-IN' : (language === 'kannada' ? 'kn-IN' : 'en-US');
        recognitionRef.current.start();
      } catch (error) {
        console.error('Error starting recognition:', error);
      }
    }
  };
  
  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  };
  
  const speakText = (text) => {
    if (!voiceEnabled || !synthesisRef.current) return;
    
    synthesisRef.current.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === 'hindi' ? 'hi-IN' : (language === 'kannada' ? 'kn-IN' : 'en-US');
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    synthesisRef.current.speak(utterance);
  };
  
  const toggleVoice = () => {
    if (isSpeaking) {
      synthesisRef.current.cancel();
      setIsSpeaking(false);
    }
    setVoiceEnabled(!voiceEnabled);
  };
  
  const handleLanguageChange = (newLanguage) => {
    setLanguage(newLanguage);
  };
  
  const clearChat = () => {
    setMessages([
      {
        text: "Chat cleared. How can I assist you?",
        isUser: false,
        timestamp: new Date()
      }
    ]);
    setInputMessage('');
    if (synthesisRef.current) {
      synthesisRef.current.cancel();
    }
  };

  const handleSend = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      text: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      // Call Groq AI backend
      const response = await fetch('http://localhost:8000/chatbot/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.text,
          language: language,
          conversation_history: messages.map(m => ({
            role: m.isUser ? 'user' : 'assistant',
            content: m.text
          })),
          context: null
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      const botMessage = {
        text: data.response,
        isUser: false,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Speak the response if voice is enabled
      if (voiceEnabled) {
        speakText(data.response);
      }
      
    } catch (error) {
      console.error('Error:', error);
      
      const botMessage = {
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
    }
    
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickQuestions = [
    "What crop is best for clay soil?",
    "How to control pest infestation?",
    "What's the ideal pH for rice?",
    "When to apply fertilizer?"
  ];

  const handleQuickQuestion = (question) => {
    setInputMessage(question);
    inputRef.current?.focus();
  };

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="page-content">
        <div className="max-w-4xl mx-auto">
          {/* Header with Controls */}
          <div className="mb-6 card">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-r from-green-500 to-blue-500 p-3 rounded-full">
                  <MessageCircle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-800">SmartAgri AI Assistant</h1>
                  <p className="text-sm text-gray-600">
                    {language === 'english' ? '🇬🇧 English' : (language === 'hindi' ? '🇮🇳 हिंदी' : '🇮🇳 ಕನ್ನಡ')} • 
                    {voiceEnabled ? ' 🔊 Voice On' : ' 🔇 Voice Off'}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                {/* Language Selector */}
                <LanguageSelector 
                  currentLanguage={language}
                  onLanguageChange={handleLanguageChange}
                />
                
                {/* Voice Toggle */}
                <button
                  onClick={toggleVoice}
                  className={`p-2 rounded-lg transition-colors ${
                    voiceEnabled 
                      ? 'bg-green-100 text-green-600 hover:bg-green-200' 
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                  title="Toggle Voice Output"
                >
                  {voiceEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
                </button>
                
                {/* Clear Chat */}
                <button
                  onClick={clearChat}
                  className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-colors"
                  title="Clear Chat"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Chat Container */}
          <div className="card h-[500px] flex flex-col">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message, index) => (
                <ChatMessage
                  key={index}
                  message={message.text}
                  isUser={message.isUser}
                />
              ))}
              
              {loading && (
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-gray-700" />
                  </div>
                  <div className="bg-gray-200 rounded-2xl rounded-tl-none px-4 py-2">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Quick Questions */}
            {messages.length === 1 && (
              <div className="px-4 pb-4">
                <p className="text-sm text-gray-600 mb-2">Quick questions:</p>
                <div className="flex flex-wrap gap-2">
                  {quickQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleQuickQuestion(question)}
                      className="text-xs bg-primary-100 hover:bg-primary-200 text-primary-700 px-3 py-2 rounded-full transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Area with Voice */}
            <div className="border-t border-gray-200 p-4">
              <div className="flex items-end space-x-2">
                {/* Voice Input Button */}
                <button
                  onClick={isListening ? stopListening : startListening}
                  disabled={loading}
                  className={`flex-shrink-0 p-3 rounded-full transition-all ${
                    isListening
                      ? 'bg-red-500 text-white animate-pulse'
                      : 'bg-green-500 text-white hover:bg-green-600'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                  title={isListening ? 'Stop Listening' : 'Start Voice Input'}
                >
                  {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                </button>
                
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    language === 'english'
                      ? 'Ask about crops, diseases, fertilizers...'
                      : 'ಬೆಳೆಗಳು, ರೋಗಗಳು, ಗೊಬ್ಬರಗಳ ಬಗ್ಗೆ ಕೇಳಿ...'
                  }
                  disabled={loading}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none disabled:bg-gray-50"
                />
                <button
                  onClick={handleSend}
                  disabled={loading || !inputMessage.trim()}
                  className="btn-primary px-6 flex items-center space-x-2"
                >
                  {loading ? (
                    <Loader className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      <span>Send</span>
                    </>
                  )}
                </button>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                <span>
                  {isSpeaking && '🔊 Speaking...'}
                  {isListening && '🎤 Listening...'}
                  {!isSpeaking && !isListening && 'Press Enter to send'}
                </span>
                <span>{messages.length} messages</span>
              </div>
            </div>
          </div>

{/* Feature Info */}
          <div className="mt-4 card">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              {language === 'english' ? '✨ Features' : '✨ ವೈಶಿಷ್ಟ್ಯಗಳು'}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
              <div>
                <p className="font-medium text-green-600">🎤 Voice Input</p>
                <p>{language === 'english' ? 'Speak your questions' : 'ನಿಮ್ಮ ಪ್ರಶ್ನೆಗಳನ್ನು ಮಾತನಾಡಿ'}</p>
              </div>
              <div>
                <p className="font-medium text-blue-600">🔊 Voice Output</p>
                <p>{language === 'english' ? 'Hear responses aloud' : 'ಪ್ರತಿಕ್ರಿಯೆಗಳನ್ನು ಕೇಳಿ'}</p>
              </div>
              <div>
                <p className="font-medium text-purple-600">🌍 Bilingual</p>
                <p>{language === 'english' ? 'English & Kannada' : 'ಇಂಗ್ಲಿಷ್ ಮತ್ತು ಕನ್ನಡ'}</p>
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-xs text-yellow-800">
              <strong>Note:</strong> This AI assistant provides general agricultural guidance. 
              Always consult with local agricultural experts for specific recommendations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
