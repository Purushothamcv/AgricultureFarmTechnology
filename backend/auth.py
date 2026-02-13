"""
Authentication Routes Module
Handles user registration and login with secure password hashing
"""

from fastapi import APIRouter, HTTPException, status, Depends
import bcrypt
from datetime import datetime
from bson import ObjectId
from models import UserRegister, UserLogin, UserResponse, LoginResponse, MessageResponse
from database import get_database

# Initialize router
router = APIRouter(prefix="/auth", tags=["Authentication"])


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt
    
    Args:
        password: Plain-text password
    
    Returns:
        Hashed password string
    """
    # Convert password to bytes and hash it
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a hashed password
    
    Args:
        plain_password: Plain-text password to verify
        hashed_password: Stored hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister, db=Depends(get_database)):
    """
    Register a new user
    
    - **name**: User's full name (2-100 characters)
    - **email**: Valid email address (must be unique)
    - **password**: Password (minimum 6 characters)
    
    Returns:
        Success message upon registration
    
    Raises:
        HTTPException: 400 if email already exists
    """
    
    print(f"[REGISTER] Registration attempt - Name: {user_data.name}, Email: {user_data.email}")
    
    try:
        # Check if user with email already exists
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            print(f"[REGISTER] ERROR: Email {user_data.email} already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered. Please use a different email or login."
            )
        
        # Hash the password
        print(f"[REGISTER] Hashing password for {user_data.email}...")
        hashed_password = hash_password(user_data.password)
        
        # Prepare user document for MongoDB
        user_document = {
            "name": user_data.name,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "role": "user",
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        # Insert user into database
        print(f"[REGISTER] Inserting user {user_data.email} into database...")
        result = await db.users.insert_one(user_document)
        
        if result.inserted_id:
            print(f"[REGISTER] SUCCESS: User {user_data.email} registered with ID: {result.inserted_id}")
            return MessageResponse(
                message=f"User '{user_data.name}' registered successfully! Please login to continue."
            )
        else:
            print(f"[REGISTER] ERROR: No inserted_id returned")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register user. Please try again."
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[REGISTER] ERROR for {user_data.email}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@router.post("/login", response_model=LoginResponse)
async def login_user(user_credentials: UserLogin, db=Depends(get_database)):
    """
    Authenticate user and login
    
    - **email**: User's registered email address
    - **password**: User's password
    
    Returns:
        Success message with user information
    
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    
    print(f"\n[LOGIN] Login attempt for email: {user_credentials.email}")
    
    try:
        # Find user by email
        user = await db.users.find_one({"email": user_credentials.email})
        
        if not user:
            print(f"[LOGIN] ERROR: User not found for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password. Please check your credentials."
            )
        
        print(f"[LOGIN] User found: {user.get('name', 'N/A')}")
        
        # Verify password
        print(f"[LOGIN] Verifying password...")
        if not verify_password(user_credentials.password, user["hashed_password"]):
            print(f"[LOGIN] ERROR: Invalid password for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password. Please check your credentials."
            )
        
        print(f"[LOGIN] Password verified successfully")
        
        # Update last_login timestamp
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Prepare response (exclude sensitive data)
        user_info = {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "user")
        }
        
        print(f"[LOGIN] SUCCESS: Login successful for user: {user_info['email']}")
        print(f"   User info: {user_info}\n")
        
        return LoginResponse(
            message="Login successful",
            user=user_info
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[LOGIN] ERROR for {user_credentials.email}: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )


@router.get("/users/me", response_model=UserResponse)
async def get_current_user(email: str, db=Depends(get_database)):
    """
    Get current user information by email
    (This endpoint can be enhanced with JWT token authentication)
    
    Args:
        email: User's email address
    
    Returns:
        User information
    
    Raises:
        HTTPException: 404 if user not found
    """
    
    user = await db.users.find_one({"email": email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Convert ObjectId to string for response
    user["_id"] = str(user["_id"])
    
    return UserResponse(**user)


@router.get("/health")
async def health_check(db=Depends(get_database)):
    """
    Health check endpoint for authentication service
    Verifies database connectivity
    """
    try:
        # Test database connection
        await db.command('ping')
        user_count = await db.users.count_documents({})
        
        return {
            "status": "healthy",
            "database": "connected",
            "users_count": user_count,
            "message": "Authentication service is operational"
        }
    except Exception as e:
        print(f"[HEALTH] ERROR: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "message": "Database connection failed"
        }
