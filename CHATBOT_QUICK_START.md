# 🚀 AI Chatbot - Quick Start Guide

## ⚡ 3-Step Setup

### Step 1: Backend Setup ✅
```bash
# Navigate to backend
cd backend

# Server should already be running!
# If not, start it:
python -m uvicorn main_fastapi:app --reload --port 8000
```

**✅ Expected Output**:
```
🤖 Initializing AI Chatbot Service...
✅ Groq AI client initialized successfully
✅ AI Chatbot Service initialized successfully!
✅ All services initialized
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Start development server
npm run dev
```

### Step 3: Use the Chatbot! 🎉
1. Open browser: `http://localhost:3000`
2. Login to SmartAgri
3. Click **"AI Chat"** in the navbar
4. Start chatting!

---

## 🎤 Voice Features

### Text Input
- Type your question
- Press **Enter** or click **Send**

### Voice Input
1. Click the **green microphone 🎤** button
2. Speak your question
3. Wait for text to appear
4. Click **Send**

### Voice Output
- Responses are automatically spoken
- Click **speaker icon 🔊** to toggle on/off

### Language Switch
- Click **globe icon 🌍** to toggle English ⇄ Kannada

---

## 💬 Example Questions

### English
- "What fertilizer should I use for tomato plants?"
- "How do I control aphids on my crops?"
- "What is the best time to plant rice?"
- "My potato plants have brown spots. What should I do?"
- "How much water does maize need?"

### Kannada (ಕನ್ನಡ)
- "ಟೊಮ್ಯಾಟೊ ಗಿಡಗಳಿಗೆ ಯಾವ ಗೊಬ್ಬರವನ್ನು ಬಳಸಬೇಕು?"
- "ಬೆಳೆಗಳ ಮೇಲಿನ ಕೀಟಗಳನ್ನು ಹೇಗೆ ನಿಯಂತ್ರಿಸುವುದು?"
- "ಅಕ್ಕಿಯನ್ನು ನೆಡುವ ಸೂಕ್ತ ಸಮಯ ಯಾವಾಗ?"

---

## 🔑 Key Features

✅ **Bilingual**: English + Kannada support  
✅ **Voice Input**: Speak your questions  
✅ **Voice Output**: Hear responses aloud  
✅ **Smart AI**: Agricultural expertise  
✅ **Context-Aware**: Remembers conversation  
✅ **Real-Time**: Instant responses  

---

## 🎯 Quick Troubleshooting

**Problem**: Chatbot not responding  
**Solution**: Check if backend server is running on port 8000

**Problem**: Voice not working  
**Solution**: Allow microphone permissions in browser

**Problem**: No sound output  
**Solution**: Check speaker icon is green (enabled)

---

## 📁 Important Files

### Backend
- `backend/chatbot_service.py` - Main AI service
- `backend/.env` - API key configuration
- `backend/main_fastapi.py` - FastAPI app

### Frontend
- `frontend/src/pages/Chatbot.jsx` - Chat UI

---

## 🌟 Pro Tips

1. **Use Voice for Hands-Free**: Perfect while working in the field
2. **Switch Languages Anytime**: No need to restart
3. **Ask Follow-Up Questions**: The bot remembers context
4. **Clear Chat**: Use trash icon to start fresh
5. **Check Status Bar**: See when AI is thinking/speaking

---

## ✅ Current Status

🟢 **FULLY OPERATIONAL**

- ✅ Backend running on port 8000
- ✅ AI service initialized
- ✅ Voice features enabled
- ✅ Bilingual support active
- ✅ Ready to use!

---

## 🆘 Need Help?

See full documentation: `AI_CHATBOT_DOCUMENTATION.md`

---

**🎉 You're all set! Start chatting with SmartAgri AI Assistant!**
