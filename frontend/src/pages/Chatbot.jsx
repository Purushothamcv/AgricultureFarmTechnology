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
import api from '../services/api';
import { MessageCircle, Mic, MicOff, Volume2, VolumeX, Send, Plus, Loader, Bot } from 'lucide-react';

// Use the same backend base URL used by other modules/services.
const API_URL = (api.defaults.baseURL || import.meta.env.VITE_BACKEND_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const API_FALLBACK_URL = API_URL.includes('localhost:8001')
  ? API_URL.replace('localhost:8001', 'localhost:8000')
  : (API_URL.includes('localhost:8000') ? API_URL.replace('localhost:8000', 'localhost:8001') : null);
const API_CANDIDATES = API_FALLBACK_URL ? [API_URL, API_FALLBACK_URL] : [API_URL];
const CHAT_USER_STORAGE_KEY = 'smartagri_chat_user_id';
const CHAT_SESSION_STORAGE_KEY = 'smartagri_chat_session_id';

const DEFAULT_WELCOME_MESSAGE = {
  text: "Hello! I'm SmartAgri AI Assistant. I can help you with crops, diseases, fertilizers, and farming advice. How can I assist you today?",
  isUser: false,
  timestamp: new Date().toISOString()
};

const createClientId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `u_${Date.now()}_${Math.floor(Math.random() * 1000000)}`;
};

const chatbotFetch = async (path, options = {}) => {
  let lastError = null;

  for (const baseUrl of API_CANDIDATES) {
    try {
      return await fetch(`${baseUrl}${path}`, options);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error('Failed to connect to backend');
};

const Chatbot = () => {
  // ============================================================================
  // STATE MANAGEMENT
  // ============================================================================
  
  const [messages, setMessages] = useState([DEFAULT_WELCOME_MESSAGE]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('english'); // 'english', 'hindi', or 'kannada'
  const [userId, setUserId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [sessionList, setSessionList] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  
  // Voice states
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const synthesisRef = useRef(null);

  const mapApiMessagesToUi = (history = []) => {
    const mapped = history
      .filter((msg) => msg?.content && (msg.role === 'user' || msg.role === 'assistant'))
      .map((msg) => ({
        text: msg.content,
        isUser: msg.role === 'user',
        timestamp: msg.timestamp || new Date().toISOString()
      }));

    return mapped.length > 0 ? mapped : [DEFAULT_WELCOME_MESSAGE];
  };

  const formatSessionTime = (isoString) => {
    if (!isoString) return '';
    const time = new Date(isoString);
    if (Number.isNaN(time.getTime())) return '';
    return time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const loadSessionList = async (activeUserId, preferredSessionId = null) => {
    if (!activeUserId) return;

    const response = await chatbotFetch(`/chat/sessions/${encodeURIComponent(activeUserId)}`);
    if (!response.ok) {
      throw new Error(`Failed to load sessions (${response.status})`);
    }

    const data = await response.json();
    const sessions = Array.isArray(data.sessions) ? data.sessions : [];
    setSessionList(sessions);

    if (preferredSessionId && sessions.some((session) => session.session_id === preferredSessionId)) {
      setSessionId(preferredSessionId);
    }
  };

  const fetchAndRenderHistory = async (activeSessionId, activeUserId, showLoadingState = false) => {
    if (showLoadingState) {
      setHistoryLoading(true);
    }

    try {
      const response = await chatbotFetch(`/chat/history/${activeSessionId}?user_id=${encodeURIComponent(activeUserId)}`);
      if (!response.ok) {
        throw new Error(`Failed to load chat history (${response.status})`);
      }

      const data = await response.json();
      setMessages(mapApiMessagesToUi(data.messages));
    } finally {
      if (showLoadingState) {
        setHistoryLoading(false);
      }
    }
  };

  const createNewSession = async (activeUserId, shouldResetUi = true) => {
    const response = await chatbotFetch('/chat/new-session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ user_id: activeUserId })
    });

    if (!response.ok) {
      throw new Error(`Failed to create new session (${response.status})`);
    }

    const data = await response.json();
    const nextSessionId = data.session_id;
    localStorage.setItem(CHAT_SESSION_STORAGE_KEY, nextSessionId);
    setSessionId(nextSessionId);

    if (shouldResetUi) {
      setMessages([DEFAULT_WELCOME_MESSAGE]);
      setInputMessage('');
      if (synthesisRef.current) {
        synthesisRef.current.cancel();
      }
    }

    return nextSessionId;
  };

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

  useEffect(() => {
    const initializeChatSession = async () => {
      let activeUserId = localStorage.getItem(CHAT_USER_STORAGE_KEY);
      if (!activeUserId) {
        activeUserId = createClientId();
        localStorage.setItem(CHAT_USER_STORAGE_KEY, activeUserId);
      }
      setUserId(activeUserId);

      try {
        const existingSessionId = localStorage.getItem(CHAT_SESSION_STORAGE_KEY);
        if (existingSessionId) {
          setSessionId(existingSessionId);
          await fetchAndRenderHistory(existingSessionId, activeUserId);
          await loadSessionList(activeUserId, existingSessionId);
        } else {
          const createdSessionId = await createNewSession(activeUserId, true);
          await loadSessionList(activeUserId, createdSessionId);
        }
      } catch (error) {
        console.error('Failed to initialize chat session:', error);
        setMessages([DEFAULT_WELCOME_MESSAGE]);
      }
    };

    initializeChatSession();
  }, []);
  
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
  
  const startNewChat = async () => {
    if (!userId || loading) return;

    try {
      setLoading(true);
      const createdSessionId = await createNewSession(userId, true);
      await loadSessionList(userId, createdSessionId);
    } catch (error) {
      console.error('Failed to create new chat session:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: 'Unable to create a new chat right now. Please try again.',
          isUser: false,
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const switchToSession = async (targetSessionId) => {
    if (!targetSessionId || !userId || targetSessionId === sessionId || loading) return;

    try {
      setSessionId(targetSessionId);
      localStorage.setItem(CHAT_SESSION_STORAGE_KEY, targetSessionId);
      await fetchAndRenderHistory(targetSessionId, userId, true);
    } catch (error) {
      console.error('Failed to switch chat session:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: 'Unable to load selected chat. Please try again.',
          isUser: false,
          timestamp: new Date().toISOString()
        }
      ]);
    }
  };

  const sendLegacyChatMessage = async (messageText) => {
    const response = await chatbotFetch('/chatbot/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: messageText,
        language,
        conversation_history: messages.map((m) => ({
          role: m.isUser ? 'user' : 'assistant',
          content: m.text,
          timestamp: m.timestamp,
        })),
        context: null,
      }),
    });

    if (!response.ok) {
      throw new Error(`Legacy chat failed: ${response.status}`);
    }

    const data = await response.json();
    return data.response;
  };

  const handleSend = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      text: inputMessage,
      isUser: true,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      let assistantResponse = '';
      let activeSessionId = sessionId;

      if (userId && !activeSessionId) {
        try {
          activeSessionId = await createNewSession(userId, false);
          await loadSessionList(userId, activeSessionId);
        } catch (sessionError) {
          console.warn('Session creation failed, falling back to legacy chat:', sessionError);
        }
      }

      if (userId && activeSessionId) {
        try {
          const response = await chatbotFetch('/chat/send', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              user_id: userId,
              session_id: activeSessionId,
              message: userMessage.text,
              language,
              context: null
            })
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          assistantResponse = data.response;
          await loadSessionList(userId, activeSessionId);
        } catch (persistentError) {
          console.warn('Persistent chat failed, falling back to legacy chat:', persistentError);
          assistantResponse = await sendLegacyChatMessage(userMessage.text);
        }
      } else {
        assistantResponse = await sendLegacyChatMessage(userMessage.text);
      }

      const botMessage = {
        text: assistantResponse,
        isUser: false,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Speak the response if voice is enabled
      if (voiceEnabled) {
        speakText(assistantResponse);
      }
      
    } catch (error) {
      console.error('Error:', error);
      
      const botMessage = {
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date().toISOString()
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
                
                {/* New Chat */}
                <button
                  onClick={startNewChat}
                  disabled={loading || !userId}
                  className="p-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="New Chat"
                >
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            {/* Chat History Sidebar */}
            <div className="card lg:col-span-1 h-[500px] flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-700">Chat History</h2>
                <button
                  onClick={startNewChat}
                  disabled={loading || !userId}
                  className="p-1.5 bg-blue-100 text-blue-600 rounded-md hover:bg-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="New Chat"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {sessionList.length === 0 && (
                  <p className="text-xs text-gray-500">No previous chats yet.</p>
                )}

                {sessionList.map((session) => {
                  const isActiveSession = session.session_id === sessionId;
                  return (
                    <button
                      key={session.session_id}
                      onClick={() => switchToSession(session.session_id)}
                      className={`w-full text-left p-2 rounded-lg border transition-colors ${
                        isActiveSession
                          ? 'border-blue-300 bg-blue-50'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      <p className="text-xs font-semibold text-gray-800 truncate">{session.title || 'New Chat'}</p>
                      <p className="text-xs text-gray-500 mt-1 line-clamp-2">{session.preview || 'No messages yet'}</p>
                      <p className="text-[11px] text-gray-400 mt-1">{formatSessionTime(session.updated_at)}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Chat Container */}
            <div className="card h-[500px] flex flex-col lg:col-span-3">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {historyLoading && (
                <div className="text-sm text-gray-500">Loading chat history...</div>
              )}

              {messages.map((message, index) => (
                <ChatMessage
                  key={index}
                  message={message.text}
                  isUser={message.isUser}
                  timestamp={message.timestamp}
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
