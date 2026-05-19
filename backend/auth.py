"""
Authentication Routes Module
Handles user registration, login with JWT tokens, and Google OAuth
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
import traceback  # For detailed error logging
from datetime import datetime, timedelta
from bson import ObjectId
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
import urllib.parse
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import UserRegister, UserLogin, UserResponse, LoginResponse, MessageResponse, GoogleAuthRequest, TokenResponse
# Import from central MongoDB Atlas connection
from db import users_collection

# Initialize router
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Google OAuth Configuration (set in .env file)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Debug: Log Google OAuth configuration status
print(f"\n[AUTH] Google OAuth Configuration Status:")
print(f"  GOOGLE_CLIENT_ID: {'LOADED' if GOOGLE_CLIENT_ID else 'MISSING - OAuth will fail'}")
print(f"  GOOGLE_CLIENT_SECRET: {'LOADED' if GOOGLE_CLIENT_SECRET else 'MISSING - OAuth will fail'}")
print(f"  BACKEND_URL: {BACKEND_URL}")
print(f"  FRONTEND_URL: {FRONTEND_URL}\n")


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
def register_user(user_data: UserRegister):
    """
    Register a new user with MongoDB Atlas
    
    - **name**: User's full name (2-100 characters)
    - **email**: Valid email address (must be unique)
    - **password**: Password (minimum 6 characters)
    
    Returns:
        Success message upon registration
    
    Raises:
        HTTPException: 400 if email already exists
        HTTPException: 503 if MongoDB connection fails
    """
    
    print(f"\n[REGISTER] ═══════════════════════════════════════════════════════════════")
    print(f"[REGISTER] Registration attempt for: {user_data.email}")
    print(f"[REGISTER] User name: {user_data.name}")
    print(f"[REGISTER] ═══════════════════════════════════════════════════════════════")
    
    try:
        # Check if user with email already exists
        print(f"[REGISTER] Checking if email already registered...")
        existing_user = users_collection.find_one({"email": user_data.email})
        if existing_user:
            print(f"[REGISTER] [FAIL] FAILED: Email {user_data.email} already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered. Please use a different email or login."
            )
        
        print(f"[REGISTER] [OK] Email is available")
        
        # Hash the password
        print(f"[REGISTER] Hashing password...")
        hashed_password = hash_password(user_data.password)
        print(f"[REGISTER] [OK] Password hashed successfully")
        
        # Prepare user document for MongoDB
        user_document = {
            "name": user_data.name,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "role": "user",
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        # Insert user into MongoDB Atlas
        print(f"[REGISTER] Inserting user into MongoDB Atlas...")
        result = users_collection.insert_one(user_document)
        
        print(f"[REGISTER] [OK][OK] SUCCESS: User registered with ID: {result.inserted_id}")
        print(f"[REGISTER] ═══════════════════════════════════════════════════════════════\n")
        
        return MessageResponse(
            message=f"User '{user_data.name}' registered successfully! Please login to continue."
        )
            
    except HTTPException as e:
        # Re-raise HTTP exceptions
        print(f"[REGISTER] [FAIL] HTTP Exception: {e.status_code} - {e.detail}")
        print(f"[REGISTER] ═══════════════════════════════════════════════════════════════\n")
        raise
    except Exception as e:
        error_str = str(e).lower()
        print(f"[REGISTER] [FAIL] Exception occurred: {type(e).__name__}")
        print(f"[REGISTER] Error message: {str(e)}")
        print(f"[REGISTER] Full traceback:")
        traceback.print_exc()
        
        # Handle MongoDB connection errors gracefully
        if "ssl" in error_str or "connection" in error_str or "handshake" in error_str:
            print(f"[REGISTER] [FAIL] MongoDB connection failed")
            print(f"[REGISTER] ═══════════════════════════════════════════════════════════════\n")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service temporarily unavailable. Please check your network connection or try again later."
            )
        
        print(f"[REGISTER] ═══════════════════════════════════════════════════════════════\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service error. Please try again."
        )



@router.post("/login", response_model=TokenResponse)
def login_user(user_credentials: UserLogin):
    """
    Authenticate user and return JWT token using MongoDB Atlas
    
    - **email**: User's registered email address
    - **password**: User's password
    
    Returns:
        JWT access token and user information
    
    Raises:
        HTTPException: 401 if credentials are invalid
        HTTPException: 503 if MongoDB connection fails
    """
    
    print(f"\n[LOGIN] ═══════════════════════════════════════════════════════════════")
    print(f"[LOGIN] Login attempt for email: {user_credentials.email}")
    print(f"[LOGIN] Request method: POST")
    print(f"[LOGIN] Request path: /auth/login")
    print(f"[LOGIN] ═══════════════════════════════════════════════════════════════")
    
    try:
        # Find user by email in MongoDB Atlas
        print(f"[LOGIN] Querying MongoDB for user: {user_credentials.email}")
        user = users_collection.find_one({"email": user_credentials.email})
        
        if not user:
            print(f"[LOGIN] [FAIL] FAILED: User not found for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"[LOGIN] [OK] User found: {user.get('name', 'N/A')} ({user['email']})")
        
        # Verify password - CRITICAL SECURITY CHECK
        print(f"[LOGIN] Verifying password...")
        is_password_valid = verify_password(user_credentials.password, user["hashed_password"])
        
        if not is_password_valid:
            print(f"[LOGIN] [FAIL] FAILED: Invalid password for email {user_credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        print(f"[LOGIN] [OK] Password verified successfully")
        
        # Update last_login timestamp in MongoDB Atlas
        try:
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            print(f"[LOGIN] [OK] Last login timestamp updated")
        except Exception as e:
            print(f"[LOGIN] [WARN] Warning: Failed to update last_login: {e}")
        
        # Create JWT token
        print(f"[LOGIN] Creating JWT token...")
        token_data = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"]
        }
        access_token = create_access_token(token_data)
        print(f"[LOGIN] [OK] JWT token created successfully")
        
        # Prepare user response (exclude sensitive data)
        user_info = {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "user")
        }
        
        print(f"[LOGIN] [OK][OK] SUCCESS: Login completed for {user_info['email']}")
        print(f"[LOGIN] ═══════════════════════════════════════════════════════════════\n")
        
        return TokenResponse(
            message="Login successful",
            user=user_info,
            access_token=access_token,
            token_type="bearer"
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        print(f"[LOGIN] [FAIL] HTTP Exception: {e.status_code} - {e.detail}")
        print(f"[LOGIN] ═══════════════════════════════════════════════════════════════\n")
        raise
    except Exception as e:
        error_str = str(e).lower()
        print(f"[LOGIN] [FAIL] Exception occurred: {type(e).__name__}")
        print(f"[LOGIN] Error message: {str(e)}")
        print(f"[LOGIN] Full traceback:")
        traceback.print_exc()
        
        # Handle MongoDB connection errors gracefully
        if "ssl" in error_str or "connection" in error_str or "handshake" in error_str:
            print(f"[LOGIN] [FAIL] MongoDB connection failed")
            print(f"[LOGIN] ═══════════════════════════════════════════════════════════════\n")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service temporarily unavailable. Please check your network connection or try again later."
            )
        
        print(f"[LOGIN] ═══════════════════════════════════════════════════════════════\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error. Please try again."
        )




@router.post("/google", response_model=TokenResponse)
def google_auth(auth_data: GoogleAuthRequest):
    """
    Authenticate user with Google OAuth using MongoDB Atlas
    
    - **credential**: Google ID token from Google Sign-In
    
    Returns:
        JWT access token and user information
    
    Raises:
        HTTPException: 401 if Google token is invalid
        HTTPException: 503 if MongoDB connection fails
    """
    
    print(f"\n[GOOGLE AUTH] Google OAuth login attempt")
    
    try:
        # Verify Google OAuth is configured
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
        
        # Check if user exists in MongoDB Atlas
        user = users_collection.find_one({"email": email})
        
        if not user:
            # Create new user with Google account in MongoDB Atlas
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
            result = users_collection.insert_one(user_document)
            user = users_collection.find_one({"_id": result.inserted_id})
            print(f"[GOOGLE AUTH] [SUCCESS] New user created with ID: {result.inserted_id}")
        else:
            # Update existing user
            print(f"[GOOGLE AUTH] Existing user found: {user.get('name', 'N/A')}")
            users_collection.update_one(
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
        
        print(f"[GOOGLE AUTH] [SUCCESS] SUCCESS: Token generated for user: {user_info['email']}")
        
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
        error_str = str(e).lower()
        print(f"[GOOGLE AUTH] ERROR: {str(e)}")
        
        # Handle MongoDB connection errors gracefully
        if "ssl" in error_str or "connection" in error_str or "handshake" in error_str:
            print(f"[GOOGLE AUTH] ERROR: MongoDB connection failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service temporarily unavailable. Please check your network connection or try again later."
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google authentication error. Please try again."
        )


@router.get("/google/login")
def google_login_redirect():
    """
    Step 1: Initiate Google OAuth flow
    Redirects user to Google's consent screen
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        error_msg = []
        if not GOOGLE_CLIENT_ID:
            error_msg.append("GOOGLE_CLIENT_ID not set")
        if not GOOGLE_CLIENT_SECRET:
            error_msg.append("GOOGLE_CLIENT_SECRET not set")
        
        detail = f"Google OAuth not configured: {', '.join(error_msg)}. Check environment variables."
        print(f"[ERROR] {detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )
    
    # Google OAuth2 endpoint
    google_oauth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    
    # Query parameters
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"{BACKEND_URL}/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    
    # Build the full URL
    full_url = f"{google_oauth_url}?{urllib.parse.urlencode(params)}"
    print(f"[GOOGLE OAUTH] Redirecting to: {full_url}")
    
    return RedirectResponse(url=full_url)


@router.get("/google/callback")
async def google_callback(code: str = None, error: str = None, state: str = None):
    """
    Step 2: Handle Google OAuth callback
    Google redirects here after user authorizes
    """
    if error:
        print(f"[GOOGLE CALLBACK] Error from Google: {error}")
        error_msg = urllib.parse.quote(error)
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error={error_msg}",
            status_code=302
        )
    
    if not code:
        print(f"[GOOGLE CALLBACK] No authorization code received")
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=No_auth_code",
            status_code=302
        )
    
    try:
        print(f"[GOOGLE CALLBACK] Received auth code, exchanging for token...")
        
        # Exchange authorization code for tokens
        import requests
        token_url = "https://oauth2.googleapis.com/token"
        
        token_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": f"{BACKEND_URL}/auth/google/callback",
            "grant_type": "authorization_code"
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        tokens = token_response.json()
        
        print(f"[GOOGLE CALLBACK] Received tokens from Google")
        
        # Get user info from ID token
        id_token_str = tokens.get("id_token")
        if not id_token_str:
            raise ValueError("No ID token in response")
        
        # Verify and decode the ID token
        idinfo = id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        print(f"[GOOGLE CALLBACK] ID token verified")
        
        # Extract user information
        email = idinfo.get("email")
        name = idinfo.get("name")
        google_id = idinfo.get("sub")
        
        if not email:
            raise ValueError("No email in ID token")
        
        print(f"[GOOGLE CALLBACK] Processing user: {email}")
        
        # Check if user exists in MongoDB
        user = users_collection.find_one({"email": email})
        
        if not user:
            # Create new user
            print(f"[GOOGLE CALLBACK] Creating new user: {email}")
            user_document = {
                "name": name,
                "email": email,
                "hashed_password": None,
                "google_id": google_id,
                "auth_provider": "google",
                "role": "user",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            result = users_collection.insert_one(user_document)
            user = users_collection.find_one({"_id": result.inserted_id})
            print(f"[GOOGLE CALLBACK] New user created with ID: {result.inserted_id}")
        else:
            # Update existing user
            print(f"[GOOGLE CALLBACK] Updating existing user: {email}")
            users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "last_login": datetime.utcnow(),
                        "google_id": google_id,
                        "name": name
                    }
                }
            )
        
        # Create JWT token
        print(f"[GOOGLE CALLBACK] Creating JWT token")
        token_data = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"]
        }
        access_token = create_access_token(token_data)
        
        # Encode token for URL
        token_encoded = urllib.parse.quote(access_token, safe='')
        user_email = urllib.parse.quote(user["email"], safe='')
        user_name = urllib.parse.quote(user["name"], safe='')
        
        # Redirect to frontend with token
        redirect_url = f"{FRONTEND_URL}/auth/callback?token={token_encoded}&email={user_email}&name={user_name}&success=true"
        print(f"[GOOGLE CALLBACK] [SUCCESS] SUCCESS: Redirecting to {FRONTEND_URL}/auth/callback")
        
        return RedirectResponse(url=redirect_url, status_code=302)
        
    except ValueError as e:
        print(f"[GOOGLE CALLBACK] Token verification failed: {str(e)}")
        error_msg = urllib.parse.quote(f"Token verification failed: {str(e)}", safe='')
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error={error_msg}",
            status_code=302
        )
    except Exception as e:
        print(f"[GOOGLE CALLBACK] Error: {str(e)}")
        error_msg = urllib.parse.quote(f"Authentication error: {str(e)}", safe='')
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error={error_msg}",
            status_code=302
        )



@router.get("/users/me", response_model=UserResponse)
async def get_current_user(email: str, db=Depends(lambda: get_database("users"))):
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
    
    # Get the user_accounts collection from the users database
    users_collection = db["user_accounts"]
    
    user = await users_collection.find_one({"email": email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Convert ObjectId to string for response
    user["_id"] = str(user["_id"])
    
    return UserResponse(**user)


@router.get("/health")
async def health_check(db=Depends(lambda: get_database("users"))):
    """
    Health check endpoint for authentication service
    Verifies database connectivity
    """
    try:
        # Test database connection
        await db.command('ping')
        
        # Get the user_accounts collection and count documents
        users_collection = db["user_accounts"]
        user_count = await users_collection.count_documents({})
        
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
