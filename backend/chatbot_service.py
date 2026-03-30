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

import logging
import os
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import get_database

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
CHAT_SESSIONS_COLLECTION = "chatbot"
MAX_CONTEXT_MESSAGES = 20
MAX_STORED_MESSAGES = 50

# Initialize Groq client
groq_client = None

# ============================================================================
# AGRICULTURAL KNOWLEDGE BASE
# ============================================================================

AGRICULTURAL_CONTEXT = """
You are SmartAgri AI Assistant, an advanced agricultural expert integrated into the Smart Agriculture Decision Support System.

Core responsibilities:
1) Crop Recommendation
    - Suggest suitable crops from soil, location, and climate context.
    - Explain clearly why a crop is suitable.
2) Yield Prediction
    - Explain predicted yield results in simple terms.
    - Highlight factors affecting yield and practical improvement steps.
3) Fertilizer Recommendation
    - Recommend fertilizers using N, P, K, pH, and moisture context.
    - Give both organic and chemical alternatives with usage guidance.
4) Plant Disease Detection
    - Identify likely diseases from symptoms.
    - Provide treatment options (organic + chemical) and prevention tips.
5) Crop Stress Analysis
    - Explain stress causes (water, weather, nutrients, pests).
    - Suggest corrective actions.
6) General Farming Advice
    - Seasonal practices, irrigation guidance, soil health, and pest management.

Context awareness rules:
- If user says "this result", "my crop", or "my prediction", treat it as system output context.
- If user provides state/district/crop/soil values, personalize advice using those details.
- Keep responses aligned to SmartAgri modules and practical farm actions.

Response style:
- Use simple, farmer-friendly language.
- Keep responses concise but helpful.
- Prefer short action lists over long theory.

Safety and reliability rules:
- Do not hallucinate unknown facts or fake measurements.
- If uncertain, say: "Based on available information..."
- Do not give harmful, unsafe, or illegal instructions.
- Prefer practical, low-risk, field-usable advice.

Smart behavior:
- If user asks "Which crop should I grow?" and key inputs are missing, ask for soil/location details.
- If user asks yield questions, explain rainfall, nutrients, and management factors.
- If user asks fertilizer questions and soil values are missing, ask for NPK/pH/moisture or location.
- If user question is vague, ask one clear follow-up question before giving a final recommendation.
"""

HINDI_INSTRUCTIONS = """
When user requests Hindi language:
- Provide responses in Hindi (हिंदी)
- Use simple, farmer-friendly language (सरल भाषा)
- IMPORTANT: Use only Hindi words and Devanagari script for the full response
- Do not mix English words, transliterations, or bilingual parentheses
- If an agricultural term is technical, explain it in pure Hindi instead of English
- Use Devanagari script
- Be culturally appropriate for Hindi-speaking farmers across India
"""

KANNADA_INSTRUCTIONS = """
When user requests Kannada language:
- Provide responses in Kannada (ಕನ್ನಡ)
- Use simple, farmer-friendly language
- IMPORTANT: Use only Kannada words and Kannada script for the full response
- Do not mix English words, transliterations, or bilingual parentheses
- If an agricultural term is technical, explain it in pure Kannada instead of English
- Be culturally appropriate for Karnataka farmers
"""

ENGLISH_INSTRUCTIONS = """
When user requests English language:
- Provide responses only in English
- Do not mix Hindi, Kannada, or other language words/scripts
- Keep wording simple, clear, and farmer-friendly
- Keep recommendations practical and actionable
"""

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StoredChatSession(BaseModel):
    user_id: str
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)

class ChatRequest(BaseModel):
    message: str
    language: str = "english"  # 'english', 'hindi', or 'kannada'
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    context: Optional[str] = None  # Additional context (crop name, disease, etc.)

class ChatResponse(BaseModel):
    response: str
    language: str
    conversation_id: Optional[str] = None


class NewSessionRequest(BaseModel):
    user_id: str


class NewSessionResponse(BaseModel):
    user_id: str
    session_id: str


class SendMessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    language: str = "english"
    context: Optional[str] = None


class SendMessageResponse(BaseModel):
    response: str


class SessionHistoryResponse(BaseModel):
    user_id: str
    session_id: str
    messages: List[ChatMessage]


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    preview: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class SessionListResponse(BaseModel):
    user_id: str
    sessions: List[ChatSessionSummary]

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
    else:
        base_prompt += "\n\n" + ENGLISH_INSTRUCTIONS
    
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


def get_chat_collection():
    """Get MongoDB collection for chatbot sessions."""
    db = get_database()
    return db[CHAT_SESSIONS_COLLECTION]


def trim_history_for_context(history: List[ChatMessage], max_messages: int = MAX_CONTEXT_MESSAGES) -> List[ChatMessage]:
    """Keep only the last N messages to avoid token overflow."""
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def trim_history_for_storage(history: List[ChatMessage], max_messages: int = MAX_STORED_MESSAGES) -> List[ChatMessage]:
    """Keep stored session history bounded for scalability."""
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def normalize_stored_messages(raw_messages: List[Dict]) -> List[ChatMessage]:
    """Convert MongoDB message documents to ChatMessage objects."""
    normalized: List[ChatMessage] = []
    for msg in raw_messages or []:
        role = msg.get("role")
        content = msg.get("content")
        timestamp = msg.get("timestamp")
        if role in {"user", "assistant"} and isinstance(content, str):
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now(timezone.utc)
            normalized.append(ChatMessage(role=role, content=content, timestamp=timestamp))
    return normalized


def build_session_title_from_messages(messages: List[ChatMessage]) -> str:
    """Generate a concise title from the first user message."""
    first_user_message = next((m.content.strip() for m in messages if m.role == "user" and m.content.strip()), "")
    if not first_user_message:
        return "New Chat"
    words = first_user_message.split()
    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


def build_session_preview(messages: List[ChatMessage]) -> str:
    """Build sidebar preview from the latest message."""
    if not messages:
        return "No messages yet"
    latest = messages[-1].content.strip()
    if len(latest) > 80:
        return latest[:80] + "..."
    return latest


def build_session_summary(session: Dict) -> ChatSessionSummary:
    """Convert a MongoDB session document into sidebar metadata."""
    messages = normalize_stored_messages(session.get("messages", []))
    created_at = session.get("created_at")
    updated_at = session.get("updated_at")
    if not isinstance(created_at, datetime):
        created_at = datetime.now(timezone.utc)
    if not isinstance(updated_at, datetime):
        updated_at = datetime.now(timezone.utc)

    title = session.get("title") or build_session_title_from_messages(messages)
    return ChatSessionSummary(
        session_id=session.get("session_id", ""),
        title=title,
        preview=build_session_preview(messages),
        created_at=created_at,
        updated_at=updated_at,
        message_count=len(messages),
    )


async def create_chat_session(user_id: str) -> str:
    """Create a new chat session for a user."""
    collection = get_chat_collection()
    session_id = str(uuid4())
    logger.debug(f"🆕 Creating new session - user_id: {user_id} | generated session_id: {session_id}")
    
    await collection.insert_one(
        {
            "user_id": user_id,
            "session_id": session_id,
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    
    logger.debug(f"💾 Session inserted to MongoDB - session_id: {session_id}")
    return session_id


async def get_chat_session(session_id: str) -> Optional[Dict]:
    """Fetch a session document by session_id."""
    collection = get_chat_collection()
    return await collection.find_one({"session_id": session_id}, {"_id": 0})


async def ensure_session_owner(session_id: str, user_id: str) -> Dict:
    """Ensure a session exists and belongs to the requesting user."""
    session = await get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")
    return session


async def append_messages_to_session(session_id: str, messages: List[ChatMessage]) -> None:
    """Append new chat messages and keep storage bounded."""
    collection = get_chat_collection()
    session = await get_chat_session(session_id)
    if not session:
        logger.error(f"❌ Session not found for append - session_id: {session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")

    current_messages = normalize_stored_messages(session.get("messages", []))
    logger.debug(f"📊 Before append - session_id: {session_id} | current messages: {len(current_messages)} | adding: {len(messages)}")
    
    updated = trim_history_for_storage(current_messages + messages)
    computed_title = session.get("title") or "New Chat"
    if computed_title == "New Chat":
        computed_title = build_session_title_from_messages(updated)

    await collection.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "title": computed_title,
                "messages": [m.model_dump() for m in updated],
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    
    logger.debug(f"📊 After append - session_id: {session_id} | total messages: {len(updated)} | new title: {computed_title}")

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
            temperature=0.5,
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

router = APIRouter(tags=["AI Chatbot"])


@router.post("/chat/new-session")
async def new_session(request: NewSessionRequest):
    """Create and return a fresh session_id for the given user."""
    try:
        # ===== USER ID VALIDATION =====
        if not request.user_id or not isinstance(request.user_id, str) or request.user_id.strip() == "":
            print("❌ Error: user_id is required and cannot be empty")
            logger.error("❌ user_id validation failed: empty or invalid user_id")
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        
        user_id = request.user_id.strip()
        print(f"Creating session for user: {user_id}")
        
        # ===== CREATE SESSION DIRECTLY =====
        import uuid
        from datetime import datetime
        
        session_id = str(uuid.uuid4())
        
        new_session_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "title": "New Chat"
        }
        
        # Get collection and insert
        try:
            collection = get_chat_collection()
            await collection.insert_one(new_session_doc)
            print(f"✅ Session created and inserted to MongoDB - session_id: {session_id}")
            logger.info(f"✅ NEW SESSION CREATED - user_id: {user_id} | session_id: {session_id}")
        except Exception as db_error:
            print(f"❌ Error inserting session to MongoDB: {str(db_error)}")
            logger.error(f"❌ MongoDB insert error: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id
        }
        
    except HTTPException as http_err:
        # Re-raise HTTP exceptions
        raise http_err
    except Exception as e:
        print(f"❌ Error creating session: {str(e)}")
        logger.error(f"❌ Unexpected error in new_session: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/chat/history/{session_id}", response_model=SessionHistoryResponse)
async def get_history(session_id: str, user_id: Optional[str] = Query(default=None)):
    """Return full chat history for a session."""
    try:
        logger.info(f"📖 HISTORY REQUESTED - session_id: {session_id} | user_id: {user_id}")
        
        session = await get_chat_session(session_id)
        if not session:
            logger.warning(f"⚠️ Session not found - session_id: {session_id}")
            raise HTTPException(status_code=404, detail="Chat session not found")

        if user_id and session.get("user_id") != user_id:
            logger.warning(f"⚠️ Session access denied - expected user_id: {user_id} | session owner: {session.get('user_id')}")
            raise HTTPException(status_code=403, detail="Session does not belong to this user")

        messages = normalize_stored_messages(session.get("messages", []))
        logger.info(f"✅ HISTORY RETRIEVED - session_id: {session_id} | message count: {len(messages)}")
        
        return SessionHistoryResponse(
            user_id=session.get("user_id", ""),
            session_id=session_id,
            messages=messages,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat history retrieval error - session_id: {session_id} | error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")


@router.get("/chat/sessions/{user_id}", response_model=SessionListResponse)
async def list_user_sessions(user_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """Return recent chat sessions for sidebar rendering."""
    try:
        logger.info(f"📋 SESSIONS LIST REQUESTED - user_id: {user_id}")
        
        collection = get_chat_collection()
        cursor = (
            collection.find({"user_id": user_id}, {"_id": 0})
            .sort("updated_at", -1)
            .limit(limit)
        )
        sessions = await cursor.to_list(length=limit)
        summaries = [build_session_summary(session) for session in sessions]
        
        session_ids = [s.session_id for s in summaries]
        logger.info(f"✅ SESSIONS LIST RETURNED - user_id: {user_id} | count: {len(summaries)} | session_ids: {session_ids}")
        
        return SessionListResponse(user_id=user_id, sessions=summaries)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat sessions list error - user_id: {user_id} | error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat sessions")


@router.post("/chat/send", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Send a user message using persistent conversation memory."""
    try:
        logger.info(f"📨 MESSAGE RECEIVED - user_id: {request.user_id} | session_id: {request.session_id}")
        logger.debug(f"📝 Message text: {request.message[:100]}...")
        
        session = await ensure_session_owner(request.session_id, request.user_id)
        full_history = normalize_stored_messages(session.get("messages", []))
        context_history = trim_history_for_context(full_history)
        
        logger.debug(f"📚 Message history length: {len(full_history)} (using last {len(context_history)} for context)")

        assistant_text = await get_ai_response(
            message=request.message,
            language=request.language,
            conversation_history=context_history,
            context=request.context,
        )

        now = datetime.now(timezone.utc)
        await append_messages_to_session(
            request.session_id,
            [
                ChatMessage(role="user", content=request.message, timestamp=now),
                ChatMessage(role="assistant", content=assistant_text, timestamp=now),
            ],
        )
        
        logger.info(f"✅ MESSAGE STORED - session_id: {request.session_id} | response length: {len(assistant_text)}")
        return SendMessageResponse(response=assistant_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Persistent chat send error - session_id: {request.session_id} | error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@router.post("/chatbot/chat", response_model=ChatResponse)
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

@router.get("/chat/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if groq_client else "unhealthy",
        "groq_initialized": groq_client is not None,
        "model": MODEL_NAME,
        "supported_languages": ["english", "hindi", "kannada"]
    }


@router.get("/chatbot/health")
async def legacy_health_check():
    """Backward-compatible chatbot health endpoint."""
    return await health_check()

@router.post("/chat/translate")
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


@router.post("/chatbot/translate")
async def legacy_translate_text(text: str, from_lang: str, to_lang: str):
    """Backward-compatible chatbot translation endpoint."""
    return await translate_text(text=text, from_lang=from_lang, to_lang=to_lang)

# ============================================================================
# STARTUP EVENT
# ============================================================================

async def startup_event():
    """Initialize chatbot service on startup"""
    try:
        logger.info("🤖 Initializing AI Chatbot Service...")
        initialize_groq_client()

        # Ensure session indexes exist for fast and scalable history lookups.
        collection = get_chat_collection()
        await collection.create_index("session_id", unique=True)
        await collection.create_index("user_id")
        await collection.create_index([("user_id", 1), ("updated_at", -1)])

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
