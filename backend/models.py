"""
Pydantic Models for User Authentication
Defines data validation schemas for user registration, login, and responses
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserRegister(BaseModel):
    """
    Schema for user registration request
    """
    name: str = Field(..., min_length=2, max_length=100, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, max_length=100, description="User's password (min 6 characters)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "password": "securePassword123"
            }
        }


class UserLogin(BaseModel):
    """
    Schema for user login request
    """
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "password": "securePassword123"
            }
        }


class UserResponse(BaseModel):
    """
    Schema for user data in API responses
    (excludes sensitive information like password)
    """
    id: str = Field(..., alias="_id", description="User ID")
    name: str
    email: EmailStr
    role: str = "user"
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "role": "user",
                "created_at": "2024-01-01T00:00:00",
                "last_login": "2024-01-15T10:30:00"
            }
        }


class UserInDB(BaseModel):
    """
    Schema for user document stored in MongoDB
    """
    name: str
    email: EmailStr
    hashed_password: str
    role: str = "user"
    created_at: datetime
    last_login: Optional[datetime] = None


class LoginResponse(BaseModel):
    """
    Schema for successful login response
    """
    message: str
    user: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Login successful",
                "user": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "role": "user"
                }
            }
        }


class MessageResponse(BaseModel):
    """
    Generic message response schema
    """
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation successful"
            }
        }
