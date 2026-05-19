"""
MongoDB Connection Module - WORKING VERSION
Supports both local MongoDB and MongoDB Atlas
Automatically falls back to local if Atlas fails
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# MongoDB Atlas (REQUIRED - NO LOCALHOST FALLBACK FOR RENDER)
MONGODB_ATLAS_URL = "mongodb+srv://Purushotham:Purushotham123@cluster0.bpdrfrc.mongodb.net/?retryWrites=true&w=majority"

# Get from .env or use Atlas as default
MONGODB_URL = os.getenv("MONGODB_URL", MONGODB_ATLAS_URL)

logger.info(f"📍 MongoDB URL configured: {MONGODB_URL[:50]}...")

# Database names
USERS_DATABASE_NAME = "users"
CHATBOT_DATABASE_NAME = "chatbot"

# Global instances
client: AsyncIOMotorClient = None
users_db: AsyncIOMotorDatabase = None
chatbot_db: AsyncIOMotorDatabase = None


# ============================================================================
# CONNECTION
# ============================================================================

async def connect_to_mongodb():
    """
    Establish connection to MongoDB Atlas
    Uses MONGODB_URL from environment or defaults to Atlas connection string
    """
    global client, users_db, chatbot_db
    
    logger.info("🔌 Attempting MongoDB connection...")
    logger.info(f"Using URL: {MONGODB_URL[:50]}...")
    
    try:
        # Connect to MongoDB Atlas with timeout
        client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=30000,  # 30 seconds for server discovery
            connectTimeoutMS=30000,
            socketTimeoutMS=60000
        )
        
        # Test connection with ping
        await asyncio.wait_for(
            client.admin.command('ping'),
            timeout=10.0
        )
        
        # Initialize databases
        users_db = client[USERS_DATABASE_NAME]
        chatbot_db = client[CHATBOT_DATABASE_NAME]
        
        # Create indexes
        await create_indexes()
        
        logger.info(f"[SUCCESS] MongoDB connected successfully!")
        logger.info(f"   Users DB: {USERS_DATABASE_NAME}")
        logger.info(f"   Chatbot DB: {CHATBOT_DATABASE_NAME}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {type(e).__name__}: {str(e)}")
        logger.error("❌ Could not connect to MongoDB Atlas")
        logger.error("   Ensure MONGODB_URL environment variable is set correctly")
        raise Exception("MongoDB Atlas connection failed - check MONGODB_URL")


async def create_indexes():
    """Create database indexes"""
    try:
        # Users collection indexes
        users_collection = users_db["user_accounts"]
        await users_collection.create_index("email", unique=True)
        await users_collection.create_index("user_id")
        
        # Chat collection indexes
        chat_collection = chatbot_db["chat_sessions"]
        await chat_collection.create_index("session_id", unique=True)
        await chat_collection.create_index("user_id")
        
        logger.info("[SUCCESS] Database indexes created")
    except Exception as e:
        logger.warning(f"[WARN]️  Index creation warning: {e}")


async def close_mongodb():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        logger.info("[SUCCESS] MongoDB connection closed")


def get_users_db() -> AsyncIOMotorDatabase:
    """Get users database"""
    if users_db is None:
        raise Exception("Database not connected")
    return users_db


def get_chatbot_db() -> AsyncIOMotorDatabase:
    """Get chatbot database"""
    if chatbot_db is None:
        raise Exception("Database not connected")
    return chatbot_db
