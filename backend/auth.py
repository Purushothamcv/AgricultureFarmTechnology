"""
Authentication Routes Module
Handles user registration, login with JWT tokens, and Google OAuth
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from datetime import datetime, timedelta
from bson import ObjectId
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
from models import UserRegister, UserLogin, UserResponse, LoginResponse, MessageResponse, GoogleAuthRequest, TokenResponse
from database import get_database

# Initialize router
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Google OAuth Configuration (set in .env file)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token (usually user_id and email)
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token from request header
    
    Args:
        credentials: Authorization header with Bearer token
    
    Returns:
        Decoded token payload
    
    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        print(f"[AUTH] Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


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


@router.post("/login", response_model=TokenResponse)
async def login_user(user_credentials: UserLogin, db=Depends(get_database)):
    """
    Authenticate user and return JWT token
    
    - **email**: User's registered email address
    - **password**: User's password
    
    Returns:
        JWT access token and user information
    
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    
    print(f"\n[LOGIN] Login attempt for email: {user_credentials.email}")
    
    try:
        # Find user by email
        user = await db.users.find_one({"email": user_credentials.email})
        
        if not user:
            print(f"[LOGIN] FAILED: User not found for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"[LOGIN] User found: {user.get('name', 'N/A')}")
        
        # Verify password - CRITICAL SECURITY CHECK
        print(f"[LOGIN] Verifying password...")
        is_password_valid = verify_password(user_credentials.password, user["hashed_password"])
        
        if not is_password_valid:
            print(f"[LOGIN] FAILED: Invalid password for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"[LOGIN] Password verified successfully ✓")
        
        # Update last_login timestamp
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Create JWT token
        token_data = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"]
        }
        access_token = create_access_token(token_data)
        
        # Prepare user response (exclude sensitive data)
        user_info = {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "user")
        }
        
        print(f"[LOGIN] SUCCESS: Token generated for user: {user_info['email']}")
        print(f"   User info: {user_info}\n")
        
        return TokenResponse(
            message="Login successful",
            user=user_info,
            access_token=access_token,
            token_type="bearer"
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



@router.post("/google", response_model=TokenResponse)
async def google_auth(auth_data: GoogleAuthRequest, db=Depends(get_database)):
    """
    Authenticate user with Google OAuth
    
    - **credential**: Google ID token from Google Sign-In
    
    Returns:
        JWT access token and user information
    
    Raises:
        HTTPException: 401 if Google token is invalid
    """
    
    print(f"\n[GOOGLE AUTH] Google OAuth login attempt")
    
    try:
        # Verify Google ID token
        if not GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth is not configured. Please set GOOGLE_CLIENT_ID in environment variables."
            )
        
        print(f"[GOOGLE AUTH] Verifying Google token...")
        idinfo = id_token.verify_oauth2_token(
            auth_data.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        # Extract user info from Google token
        email = idinfo.get('email')
        name = idinfo.get('name')
        google_id = idinfo.get('sub')
        
        if not email:
            print(f"[GOOGLE AUTH] FAILED: No email in Google token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to retrieve email from Google account"
            )
        
        print(f"[GOOGLE AUTH] Google token verified for: {email}")
        
        # Check if user exists
        user = await db.users.find_one({"email": email})
        
        if not user:
            # Create new user with Google account
            print(f"[GOOGLE AUTH] Creating new user for Google account: {email}")
            user_document = {
                "name": name,
                "email": email,
                "hashed_password": None,  # No password for OAuth users
                "google_id": google_id,
                "auth_provider": "google",
                "role": "user",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            result = await db.users.insert_one(user_document)
            user = await db.users.find_one({"_id": result.inserted_id})
            print(f"[GOOGLE AUTH] New user created with ID: {result.inserted_id}")
        else:
            # Update existing user
            print(f"[GOOGLE AUTH] Existing user found: {user.get('name', 'N/A')}")
            await db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {
                    "last_login": datetime.utcnow(),
                    "google_id": google_id
                }}
            )
        
        # Create JWT token
        token_data = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"]
        }
        access_token = create_access_token(token_data)
        
        # Prepare user response
        user_info = {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "user")
        }
        
        print(f"[GOOGLE AUTH] SUCCESS: Token generated for user: {user_info['email']}\n")
        
        return TokenResponse(
            message="Google login successful",
            user=user_info,
            access_token=access_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Invalid token
        print(f"[GOOGLE AUTH] FAILED: Invalid Google token - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )
    except Exception as e:
        print(f"[GOOGLE AUTH] ERROR: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google authentication error: {str(e)}"
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
