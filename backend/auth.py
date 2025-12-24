"""
Authentication Routes Module
Handles user registration and login with secure password hashing
"""

from fastapi import APIRouter, HTTPException, status, Depends
from passlib.context import CryptContext
from datetime import datetime
from bson import ObjectId
from models import UserRegister, UserLogin, UserResponse, LoginResponse, MessageResponse
from database import get_database

# Initialize router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Password hashing configuration using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt
    
    Args:
        password: Plain-text password
    
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a hashed password
    
    Args:
        plain_password: Plain-text password to verify
        hashed_password: Stored hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


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
    
    # Check if user with email already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered. Please use a different email or login."
        )
    
    # Hash the password
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
    try:
        result = await db.users.insert_one(user_document)
        
        if result.inserted_id:
            return MessageResponse(
                message=f"User '{user_data.name}' registered successfully! Please login to continue."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register user. Please try again."
            )
            
    except Exception as e:
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
    
    print(f"🔐 Login attempt for email: {user_credentials.email}")
    
    # Find user by email
    user = await db.users.find_one({"email": user_credentials.email})
    
    if not user:
        print(f"❌ Login failed: User not found for email {user_credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password. Please check your credentials."
        )
    
    # Verify password
    if not verify_password(user_credentials.password, user["hashed_password"]):
        print(f"❌ Login failed: Invalid password for email {user_credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password. Please check your credentials."
        )
    
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
    
    print(f"✅ Login successful for user: {user_info['email']}")
    
    return LoginResponse(
        message="Login successful",
        user=user_info
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
