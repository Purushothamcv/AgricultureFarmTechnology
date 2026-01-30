"""
AI Chatbot Service with Groq Integration
=========================================
Multilingual voice-enabled chatbot for agricultural assistance.

Features:
- Groq AI integration for intelligent responses
- Multilingual support (English, Hindi & Kannada)
- Context-aware responses about crops, diseases, fertilizers
- Integration with existing agricultural data
"""

import os
import json
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"  # Updated to current Groq model (Jan 2026)

# Initialize Groq client
groq_client = None

# ============================================================================
# AGRICULTURAL KNOWLEDGE BASE
# ============================================================================

AGRICULTURAL_CONTEXT = """
You are SmartAgri AI Assistant, an expert agricultural advisor specializing in Indian farming practices. 
You have deep knowledge about:

1. CROPS: Rice, Wheat, Cotton, Maize, Sugarcane, Jute, Coffee, Tea, and more
2. PLANT DISEASES: Early Blight, Late Blight, Bacterial Spot, Leaf Curl, Powdery Mildew, etc.
3. FRUIT DISEASES: Anthracnose, Black Mold, Scab, Rot, Alternaria, Cercospora, etc.
4. FERTILIZERS: NPK ratios, Urea, DAP, Potash, organic fertilizers
5. WEATHER CONDITIONS: Temperature, humidity, rainfall requirements
6. SOIL TYPES: Black soil, Red soil, Alluvial soil, Laterite soil

You provide practical, farmer-friendly advice in English, Hindi, and Kannada.
Always be helpful, accurate, and consider the Indian agricultural context.
"""

HINDI_INSTRUCTIONS = """
When user requests Hindi language:
- Provide responses in Hindi (हिंदी)
- Use simple, farmer-friendly language (सरल भाषा)
- Include both Hindi and English technical terms in parentheses for clarity
- Use Devanagari script
- Be culturally appropriate for Hindi-speaking farmers across India
"""

KANNADA_INSTRUCTIONS = """
When user requests Kannada language:
- Provide responses in Kannada (ಕನ್ನಡ)
- Use simple, farmer-friendly language
- Include both Kannada and English technical terms in parentheses for clarity
- Be culturally appropriate for Karnataka farmers
"""

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    language: str = "english"  # 'english', 'hindi', or 'kannada'
    conversation_history: List[ChatMessage] = []
    context: Optional[str] = None  # Additional context (crop name, disease, etc.)

class ChatResponse(BaseModel):
    response: str
    language: str
    conversation_id: Optional[str] = None

# ============================================================================
# GROQ CLIENT INITIALIZATION
# ============================================================================

def initialize_groq_client():
    """Initialize Groq AI client"""
    global groq_client
    
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not found in environment variables")
        raise ValueError("GROQ_API_KEY not configured")
    
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq AI client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq client: {str(e)}")
        raise

# ============================================================================
# CHATBOT LOGIC
# ============================================================================

def build_system_prompt(language: str, context: Optional[str] = None) -> str:
    """
    Build system prompt based on language and context
    
    Args:
        language: 'english', 'hindi', or 'kannada'
        context: Additional context information
        
    Returns:
        System prompt string
    """
    base_prompt = AGRICULTURAL_CONTEXT
    
    if language == "hindi":
        base_prompt += "\n\n" + HINDI_INSTRUCTIONS
    elif language == "kannada":
        base_prompt += "\n\n" + KANNADA_INSTRUCTIONS
    
    if context:
        base_prompt += f"\n\nCurrent Context: {context}"
    
    return base_prompt

def format_conversation_history(history: List[ChatMessage]) -> List[Dict]:
    """
    Format conversation history for Groq API
    
    Args:
        history: List of ChatMessage objects
        
    Returns:
        List of message dictionaries
    """
    return [{"role": msg.role, "content": msg.content} for msg in history]

async def get_ai_response(
    message: str,
    language: str = "english",
    conversation_history: List[ChatMessage] = [],
    context: Optional[str] = None
) -> str:
    """
    Get AI response from Groq
    
    Args:
        message: User's message
        language: Response language
        conversation_history: Previous conversation
        context: Additional context
        
    Returns:
        AI-generated response
    """
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq AI client not initialized")
    
    try:
        # Build system prompt
        system_prompt = build_system_prompt(language, context)
        
        # Format messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(format_conversation_history(conversation_history))
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Log request
        logger.info(f"🤖 Sending request to Groq (language: {language})")
        
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        # Extract response
        response = chat_completion.choices[0].message.content
        
        logger.info(f"✅ Received response from Groq ({len(response)} chars)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# ============================================================================
# FASTAPI ROUTER
# ============================================================================

router = APIRouter(prefix="/chatbot", tags=["AI Chatbot"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with AI assistant
    
    **Features:**
    - Bilingual support (English/Kannada)
    - Context-aware responses
    - Conversation history tracking
    - Agricultural expertise
    
    **Example Request:**
    ```json
    {
        "message": "What fertilizer should I use for tomato plants?",
        "language": "english",
        "conversation_history": [],
        "context": "Tomato crop in Karnataka"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "response": "For tomato plants, I recommend...",
        "language": "english"
    }
    ```
    """
    try:
        # Get AI response
        response = await get_ai_response(
            message=request.message,
            language=request.language,
            conversation_history=request.conversation_history,
            context=request.context
        )
        
        return ChatResponse(
            response=response,
            language=request.language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if groq_client else "unhealthy",
        "groq_initialized": groq_client is not None,
        "model": MODEL_NAME,
        "supported_languages": ["english", "kannada"]
    }

@router.post("/translate")
async def translate_text(text: str, from_lang: str, to_lang: str):
    """
    Translate text between English and Kannada
    
    Args:
        text: Text to translate
        from_lang: Source language ('english' or 'kannada')
        to_lang: Target language ('english' or 'kannada')
    """
    if from_lang == to_lang:
        return {"translated_text": text}
    
    try:
        prompt = f"Translate the following text from {from_lang} to {to_lang}. Only provide the translation, no explanations:\n\n{text}"
        
        response = await get_ai_response(
            message=prompt,
            language=to_lang,
            conversation_history=[],
            context="Translation task"
        )
        
        return {"translated_text": response}
        
    except Exception as e:
        logger.error(f"❌ Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP EVENT
# ============================================================================

async def startup_event():
    """Initialize chatbot service on startup"""
    try:
        logger.info("🤖 Initializing AI Chatbot Service...")
        initialize_groq_client()
        logger.info("✅ AI Chatbot Service initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot service: {str(e)}")
        raise

# ============================================================================
# HELPER FUNCTIONS FOR AGRICULTURAL CONTEXT
# ============================================================================

def get_crop_context(crop_name: str) -> str:
    """Get context information about a specific crop"""
    return f"User is asking about {crop_name} crop. Provide relevant information about cultivation, diseases, fertilizers, and best practices."

def get_disease_context(disease_name: str, plant_type: str) -> str:
    """Get context information about a specific disease"""
    return f"User is asking about {disease_name} disease in {plant_type}. Provide information about symptoms, prevention, and treatment."

def get_fertilizer_context(crop_name: str, soil_type: str = None) -> str:
    """Get context information for fertilizer recommendations"""
    context = f"User needs fertilizer recommendations for {crop_name}"
    if soil_type:
        context += f" in {soil_type} soil"
    return context

if __name__ == "__main__":
    # For testing purposes
    import asyncio
    
    async def test_chatbot():
        """Test chatbot initialization"""
        await startup_event()
        print("\n" + "="*60)
        print("AI Chatbot Service - Test Results")
        print("="*60)
        print(f"\n✅ Service Status: {'Ready' if groq_client else 'Failed'}")
        print(f"🤖 Model: {MODEL_NAME}")
        print(f"🌍 Languages: English, Kannada")
        print("="*60)
    
    asyncio.run(test_chatbot())
