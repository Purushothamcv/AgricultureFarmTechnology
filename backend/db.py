"""
MongoDB Atlas Connection Module
Central database connection for all authentication and chatbot operations
Uses synchronous PyMongo for simplicity and reliability
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# MONGODB ATLAS CONFIGURATION
# ============================================================================

# MongoDB Atlas connection string (REQUIRED - NO LOCALHOST FALLBACK)
MONGO_URI = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://Purushotham:Purushotham123@cluster0.bpdrfrc.mongodb.net/?retryWrites=true&w=majority"
)

print(f"\n{'='*70}")
print(f"[DB] MongoDB Atlas Connection")
print(f"{'='*70}")
print(f"Connection String: {MONGO_URI[:50]}...")

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

try:
    # Create MongoDB client with connection pooling
    # Increased timeouts for MongoDB Atlas (30s for initial connection, 60s for operations)
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=30000,  # 30 seconds for server discovery
        connectTimeoutMS=30000,  # 30 seconds to establish connection
        socketTimeoutMS=60000,  # 60 seconds for socket operations
        maxPoolSize=50,
        retryWrites=True,
        retryReads=True,
        w='majority'
    )
    
    # Test connection with ping command (non-critical - will retry on first use)
    try:
        client.admin.command('ping', timeoutMS=10000)
        print("[OK] MongoDB Atlas Connected Successfully!")
    except Exception as e:
        print(f"[WARN] MongoDB Initial Connection Check: {str(e)[:100]}")
        print("[WARN] Will retry on first database operation...")
    
    print(f"{'='*70}\n")
    
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"[ERROR] MongoDB Connection Failed: {e}")
    print(f"{'='*70}\n")
    raise

# ============================================================================
# DATABASE INSTANCES
# ============================================================================

# Users database for authentication
users_db = client["users"]

# Chatbot database for chat sessions and messages
chatbot_db = client["chatbot"]

# ============================================================================
# COLLECTIONS
# ============================================================================

# User accounts collection
users_collection = users_db["user_accounts"]

# Chat sessions collection
chat_sessions_collection = chatbot_db["chat_sessions"]

# Chat messages collection
chat_messages_collection = chatbot_db["chat_messages"]

# ============================================================================
# INDEXES & INITIALIZATION
# ============================================================================

try:
    # Create unique index on email for users
    users_collection.create_index("email", unique=True)
    print("[OK] Created unique index on users collection (email)")
    
    # Create index on user_id for quick lookups
    users_collection.create_index("user_id")
    print("[OK] Created index on users collection (user_id)")
    
    # Create unique index on session_id for chatbot
    chat_sessions_collection.create_index("session_id", unique=True)
    print("[OK] Created unique index on chat_sessions collection (session_id)")
    
    # Create index on user_id for chatbot sessions
    chat_sessions_collection.create_index("user_id")
    print("[OK] Created index on chat_sessions collection (user_id)")
    
    # Create index on timestamp for chat messages
    chat_messages_collection.create_index("timestamp")
    print("[OK] Created index on chat_messages collection (timestamp)")
    
except Exception as e:
    print(f"[WARN] Index creation info: {e}")

print()

# ============================================================================
# EXPORT CONNECTIONS
# ============================================================================
# All imports should use these objects:
# from db import users_collection, chat_sessions_collection, client

__all__ = [
    'client',
    'users_db',
    'chatbot_db',
    'users_collection',
    'chat_sessions_collection',
    'chat_messages_collection'
]
