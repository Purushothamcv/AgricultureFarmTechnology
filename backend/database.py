"""
MongoDB Database Connection Module
Handles connection to MongoDB using Motor (async driver for FastAPI)
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "FinalProject"

# Global MongoDB client and database instances
client = None
database = None


async def connect_to_mongodb():
    """
    Establish connection to MongoDB
    Called during application startup
    """
    global client, database
    
    try:
        client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        database = client[DATABASE_NAME]
        
        # Verify connection
        await client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB database: {DATABASE_NAME}")
        
        # Create unique index on email field for users collection
        await database.users.create_index("email", unique=True)
        print("✅ Email index created successfully")
        
        # Create additional collections (auto-created on first insert)
        # Collections: users, plant_disease_predictions, weather_logs
        print("📦 Collections available: users, plant_disease_predictions, weather_logs")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to MongoDB: {e}")
        print("⚠️  Please ensure MongoDB is running or configure MONGODB_URL in .env file")
        print("⚠️  The application will start but authentication features will not work.")
        # Don't raise - allow app to start


async def close_mongodb_connection():
    """
    Close MongoDB connection
    Called during application shutdown
    """
    global client
    
    if client:
        client.close()
        print("✅ MongoDB connection closed")


def get_database():
    """
    Get database instance
    Used for dependency injection in FastAPI routes
    """
    return database
