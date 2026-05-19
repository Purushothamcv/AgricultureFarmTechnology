"""
MongoDB Database Connection Module
Handles connection to MongoDB Atlas and local MongoDB using Motor (async driver for FastAPI)
Manages multiple databases: users, chatbot, and legacy data
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# MONGODB CONFIGURATION
# ============================================================================

# MongoDB Atlas is REQUIRED - no local fallback
MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is required. Please set it in .env file.")

USERS_DATABASE_NAME = os.getenv("USERS_DATABASE", "users")
CHATBOT_DATABASE_NAME = os.getenv("CHATBOT_DATABASE", "chatbot")
LEGACY_DATABASE_NAME = os.getenv("LEGACY_DATABASE", "FinalProject")

# Global MongoDB client and database instances
client: AsyncIOMotorClient = None
users_db: AsyncIOMotorDatabase = None
chatbot_db: AsyncIOMotorDatabase = None
legacy_db: AsyncIOMotorDatabase = None  # For backward compatibility

# Collection references
users_collection: AsyncIOMotorCollection = None
chat_sessions_collection: AsyncIOMotorCollection = None


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

async def connect_to_mongodb():
    """
    Establish connection to MongoDB Atlas with retry logic
    Called during application startup
    """
    global client, users_db, chatbot_db, legacy_db, users_collection, chat_sessions_collection
    
    logger.info(f"Connecting to MongoDB...")
    logger.info(f"MongoDB URL: {MONGODB_URL[:30]}..." if len(MONGODB_URL) > 30 else f"MongoDB URL: {MONGODB_URL}")
    
    max_retries = 3
    retry_delay = 1  # Reduced from 2 to 1 second
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries}...")
                await asyncio.sleep(retry_delay)
            
            # Connect to MongoDB Atlas with proper timeout settings
            client = AsyncIOMotorClient(
                MONGODB_URL, 
                serverSelectionTimeoutMS=30000,  # 30 seconds for server discovery
                connectTimeoutMS=30000,  # 30 seconds to establish connection
                socketTimeoutMS=60000,  # 60 seconds for socket operations
                retryWrites=True,  # Enable automatic write retries
                retryReads=True,  # Enable automatic read retries
                maxPoolSize=10,  # Connection pool size
                minPoolSize=2   # Minimum connection pool size
            )
            
            # Initialize database instances
            users_db = client[USERS_DATABASE_NAME]
            chatbot_db = client[CHATBOT_DATABASE_NAME]
            legacy_db = client[LEGACY_DATABASE_NAME]  # For backward compatibility
            
            # Initialize collection references
            users_collection = users_db["user_accounts"]
            chat_sessions_collection = chatbot_db["chat_sessions"]
            
            # Verify connection
            await asyncio.wait_for(
                client.admin.command('ping'),
                timeout=5.0
            )
            
            logger.info(f"[SUCCESS] Successfully connected to MongoDB")
            logger.info(f"[SUCCESS] Users Database: {USERS_DATABASE_NAME}")
            logger.info(f"[SUCCESS] Chatbot Database: {CHATBOT_DATABASE_NAME}")
            logger.info(f"[SUCCESS] Legacy Database: {LEGACY_DATABASE_NAME} (backward compatibility)")
            
            # Create indexes and collections
            await initialize_collections()
            
            return  # Success - exit function
            
        except asyncio.TimeoutError:
            logger.error(f"Connection attempt {attempt + 1} timed out")
        except Exception as e:
            logger.error(f"Connection attempt {attempt + 1} failed: {type(e).__name__}: {e}")
    
    # All retries failed
    logger.warning("="*60)
    logger.warning("[WARN]️  WARNING: Could not connect to MongoDB!")
    logger.warning("Please ensure:")
    logger.warning("1. MongoDB Atlas cluster is running")
    logger.warning("2. IP whitelist includes your machine")
    logger.warning("3. MONGODB_URL in .env is correct")
    logger.warning("4. Credentials are valid")
    logger.warning("="*60)


async def initialize_collections():
    """
    Initialize database collections with indexes
    """
    try:
        # Create unique index on email for users
        await users_collection.create_index("email", unique=True)
        logger.info("[SUCCESS] Created unique index on users collection (email)")
        
        # Create index on user_id for quick lookups
        await users_collection.create_index("user_id")
        logger.info("[SUCCESS] Created index on users collection (user_id)")
        
        # Create index on session_id for chatbot
        await chat_sessions_collection.create_index("session_id", unique=True)
        logger.info("[SUCCESS] Created unique index on chat_sessions collection (session_id)")
        
        # Create index on user_id for chatbot sessions
        await chat_sessions_collection.create_index("user_id")
        logger.info("[SUCCESS] Created index on chat_sessions collection (user_id)")
        
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")


async def close_mongodb_connection():
    """
    Close MongoDB connection
    Called during application shutdown
    """
    global client
    
    if client:
        client.close()
        logger.info("MongoDB connection closed")


def get_database(database_type: str = "legacy") -> AsyncIOMotorDatabase:
    """
    Get database instance by type
    
    Args:
        database_type: "users", "chatbot", or "legacy" (default for backward compatibility)
    
    Returns:
        AsyncIOMotorDatabase instance
        
    Raises:
        Exception: If database is not connected
    """
    if database_type == "users":
        if users_db is None:
            raise Exception("Users database not connected - MongoDB connection failed")
        return users_db
    elif database_type == "chatbot":
        if chatbot_db is None:
            raise Exception("Chatbot database not connected - MongoDB connection failed")
        return chatbot_db
    else:  # legacy
        if legacy_db is None:
            raise Exception("Database not connected - MongoDB connection failed")
        return legacy_db


# ============================================================================
# USER OPERATIONS (LOGIN / REGISTRATION)
# ============================================================================

async def create_user(name: str, email: str, password_hash: str) -> dict:
    """
    Create a new user account
    
    Args:
        name: User's full name
        email: User's email (must be unique)
        password_hash: Hashed password (use bcrypt)
    
    Returns:
        Created user document
        
    Raises:
        DuplicateKeyError: If email already exists
        Exception: Database error
    """
    if users_collection is None:
        raise Exception("Users collection not initialized")
    
    try:
        # Check if email already exists
        existing_user = await users_collection.find_one({"email": email})
        if existing_user:
            logger.warning(f"❌ User registration failed: email already exists - {email}")
            raise DuplicateKeyError(f"Email {email} already registered")
        
        user_doc = {
            "user_id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password": password_hash,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        result = await users_collection.insert_one(user_doc)
        logger.info(f"[SUCCESS] User created successfully - email: {email} | user_id: {user_doc['user_id']}")
        
        return user_doc
        
    except DuplicateKeyError as e:
        logger.error(f"❌ Duplicate email error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating user: {e}")
        raise


async def get_user_by_email(email: str) -> dict:
    """
    Fetch user by email
    
    Args:
        email: User's email
    
    Returns:
        User document or None
    """
    if users_collection is None:
        raise Exception("Users collection not initialized")
    
    try:
        user = await users_collection.find_one({"email": email})
        if user:
            logger.debug(f"[SUCCESS] User found by email: {email}")
        else:
            logger.debug(f"[WARN]️ User not found by email: {email}")
        return user
    except Exception as e:
        logger.error(f"❌ Error fetching user by email: {e}")
        raise


async def get_user_by_id(user_id: str) -> dict:
    """
    Fetch user by user_id
    
    Args:
        user_id: User's unique ID (UUID)
    
    Returns:
        User document or None
    """
    if users_collection is None:
        raise Exception("Users collection not initialized")
    
    try:
        user = await users_collection.find_one({"user_id": user_id})
        if user:
            logger.debug(f"[SUCCESS] User found by ID: {user_id}")
        else:
            logger.debug(f"[WARN]️ User not found by ID: {user_id}")
        return user
    except Exception as e:
        logger.error(f"❌ Error fetching user by ID: {e}")
        raise


async def update_user(user_id: str, update_data: dict) -> dict:
    """
    Update user information
    
    Args:
        user_id: User's unique ID
        update_data: Dictionary with fields to update
    
    Returns:
        Updated user document
    """
    if users_collection is None:
        raise Exception("Users collection not initialized")
    
    try:
        update_data["updated_at"] = datetime.utcnow()
        
        result = await users_collection.find_one_and_update(
            {"user_id": user_id},
            {"$set": update_data},
            return_document=True
        )
        
        if result:
            logger.info(f"[SUCCESS] User updated - user_id: {user_id}")
        else:
            logger.warning(f"[WARN]️ User not found for update - user_id: {user_id}")
            
        return result
    except Exception as e:
        logger.error(f"❌ Error updating user: {e}")
        raise


# ============================================================================
# CHATBOT SESSION OPERATIONS
# ============================================================================

async def create_chat_session(user_id: str) -> dict:
    """
    Create a new chatbot session
    
    Args:
        user_id: User's ID (from users collection)
    
    Returns:
        Created session document
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        session_doc = {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "messages": [],
            "title": "New Chat",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0
        }
        
        result = await chat_sessions_collection.insert_one(session_doc)
        logger.info(f"[SUCCESS] Chat session created - session_id: {session_doc['session_id']} | user_id: {user_id}")
        
        return session_doc
        
    except Exception as e:
        logger.error(f"❌ Error creating chat session: {e}")
        raise


async def get_chat_session(session_id: str) -> dict:
    """
    Fetch chat session by session_id
    
    Args:
        session_id: Session's unique ID
    
    Returns:
        Session document or None
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        session = await chat_sessions_collection.find_one({"session_id": session_id})
        if session:
            logger.debug(f"[SUCCESS] Chat session found: {session_id}")
        else:
            logger.debug(f"[WARN]️ Chat session not found: {session_id}")
        return session
    except Exception as e:
        logger.error(f"❌ Error fetching chat session: {e}")
        raise


async def save_message(session_id: str, role: str, content: str, metadata: dict = None) -> bool:
    """
    Add a message to a chat session
    
    Args:
        session_id: Session's unique ID
        role: Message role ('user' or 'assistant')
        content: Message content/text
        metadata: Optional metadata (context, language, etc.)
    
    Returns:
        True if successful
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
        
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
        
        if metadata:
            message.update(metadata)
        
        result = await chat_sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message},
                "$inc": {"message_count": 1},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count == 0:
            logger.warning(f"[WARN]️ Session not found for message insertion - session_id: {session_id}")
            return False
        
        logger.debug(f"[SUCCESS] Message saved to session - session_id: {session_id} | role: {role}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving message: {e}")
        raise


async def get_chat_history(session_id: str) -> dict:
    """
    Fetch full chat history for a session
    
    Args:
        session_id: Session's unique ID
    
    Returns:
        Session document with all messages
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        session = await chat_sessions_collection.find_one({"session_id": session_id})
        if session:
            logger.info(f"[SUCCESS] Chat history fetched - session_id: {session_id} | messages: {len(session.get('messages', []))}")
        else:
            logger.warning(f"[WARN]️ Session not found - session_id: {session_id}")
        return session
    except Exception as e:
        logger.error(f"❌ Error fetching chat history: {e}")
        raise


async def get_user_sessions(user_id: str, limit: int = 20) -> list:
    """
    Fetch all chat sessions for a user
    
    Args:
        user_id: User's ID
        limit: Maximum number of sessions to return
    
    Returns:
        List of session documents (sorted by most recent first)
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        cursor = chat_sessions_collection.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).limit(limit)
        
        sessions = await cursor.to_list(length=limit)
        logger.info(f"[SUCCESS] User sessions fetched - user_id: {user_id} | count: {len(sessions)}")
        
        return sessions
        
    except Exception as e:
        logger.error(f"❌ Error fetching user sessions: {e}")
        raise


async def delete_chat_session(session_id: str) -> bool:
    """
    Delete a chat session
    
    Args:
        session_id: Session's unique ID
    
    Returns:
        True if deleted, False if not found
    """
    if chat_sessions_collection is None:
        raise Exception("Chat sessions collection not initialized")
    
    try:
        result = await chat_sessions_collection.delete_one({"session_id": session_id})
        
        if result.deleted_count > 0:
            logger.info(f"[SUCCESS] Chat session deleted - session_id: {session_id}")
            return True
        else:
            logger.warning(f"[WARN]️ Session not found for deletion - session_id: {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error deleting chat session: {e}")
        raise


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def get_legacy_database():
    """
    Get legacy database (FinalProject) for backward compatibility
    This maintains compatibility with existing code
    """
    return get_database("legacy")
