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

# Primary: MongoDB Atlas
MONGODB_ATLAS_URL = "mongodb+srv://Purushotham:Purushotham123@cluster0.bpdrfrc.mongodb.net/?retryWrites=true&w=majority"

# Fallback: Local MongoDB
MONGODB_LOCAL_URL = "mongodb://localhost:27017"

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
    Establish connection to MongoDB
    Tries Atlas first, then falls back to local MongoDB
    """
    global client, users_db, chatbot_db
    
    logger.info("🔌 Attempting MongoDB connection...")
    
    # Try original URL first
    urls_to_try = [MONGODB_URL]
    
    # If using Atlas and it fails, try local as fallback
    if "mongodb+srv" in MONGODB_URL:
        urls_to_try.append(MONGODB_LOCAL_URL)
    
    for attempt_url in urls_to_try:
        try:
            logger.info(f"⏳ Trying: {attempt_url[:30]}...")
            
            # Connect with timeout
            client = AsyncIOMotorClient(
                attempt_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection with ping
            await asyncio.wait_for(
                client.admin.command('ping'),
                timeout=5.0
            )
            
            # Initialize databases
            users_db = client[USERS_DATABASE_NAME]
            chatbot_db = client[CHATBOT_DATABASE_NAME]
            
            # Create indexes
            await create_indexes()
            
            logger.info(f"✅ MongoDB connected successfully!")
            logger.info(f"   URL: {attempt_url[:40]}...")
            logger.info(f"   Users DB: {USERS_DATABASE_NAME}")
            logger.info(f"   Chatbot DB: {CHATBOT_DATABASE_NAME}")
            
            return True
            
        except Exception as e:
            logger.warning(f"❌ Failed: {type(e).__name__}: {str(e)[:100]}")
            continue
    
    logger.error("❌ Could not connect to any MongoDB instance")
    logger.error("   Install local MongoDB and try again")
    raise Exception("MongoDB connection failed - install local MongoDB")


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
        
        logger.info("✅ Database indexes created")
    except Exception as e:
        logger.warning(f"⚠️  Index creation warning: {e}")


async def close_mongodb():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        logger.info("✅ MongoDB connection closed")


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
