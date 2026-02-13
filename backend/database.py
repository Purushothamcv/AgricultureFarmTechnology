"""
MongoDB Database Connection Module
Handles connection to MongoDB using Motor (async driver for FastAPI)
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import os
import asyncio
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
    Establish connection to MongoDB with retry logic
    Called during application startup
    """
    global client, database
    
    print(f"Connecting to MongoDB...")
    print(f"MongoDB URL: {MONGODB_URL[:20]}..." if len(MONGODB_URL) > 20 else f"MongoDB URL: {MONGODB_URL}")
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt + 1}/{max_retries}...")
                await asyncio.sleep(retry_delay)
            
            client = AsyncIOMotorClient(
                MONGODB_URL, 
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            database = client[DATABASE_NAME]
            
            # Verify connection with timeout
            await asyncio.wait_for(
                client.admin.command('ping'),
                timeout=5.0
            )
            print(f"Successfully connected to MongoDB database: {DATABASE_NAME}")
            
            # Create unique index on email field for users collection
            try:
                await database.users.create_index("email", unique=True)
                print("Email index created successfully")
            except Exception as idx_error:
                print(f"Index creation warning: {idx_error}")
            
            # Create additional collections (auto-created on first insert)
            # Collections: users, plant_disease_predictions, weather_logs
            print("Collections available: users, plant_disease_predictions, weather_logs")
            return  # Success - exit function
            
        except asyncio.TimeoutError:
            print(f"Connection attempt {attempt + 1} timed out")
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {type(e).__name__}: {e}")
    
    # All retries failed
    print("========================================")
    print("WARNING: Could not connect to MongoDB!")
    print("Please ensure MongoDB is running:")
    print("1. Check if MongoDB service is started")
    print("2. Verify MongoDB_URL in .env file")
    print("3. Test connection: mongosh or mongo")
    print("The application will start but authentication features will not work.")
    print("========================================")
    # Don't raise - allow app to start without database


async def close_mongodb_connection():
    """
    Close MongoDB connection
    Called during application shutdown
    """
    global client
    
    if client:
        client.close()
        print("MongoDB connection closed")


def get_database():
    """
    Get database instance
    Used for dependency injection in FastAPI routes
    
    Raises:
        HTTPException: If database is not connected
    """
    if database is None:
        from fastapi import HTTPException
        print("Database not connected - cannot process request")
        raise HTTPException(
            status_code=503,
            detail="Database connection not available. Please check MongoDB configuration."
        )
    return database
