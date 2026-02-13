"""
Ultra minimal auth server - no startup events, just routes
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys

# Force UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI(title="SmartAgri Auth", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

@app.get("/")
def root():
    return {"service": "SmartAgri Auth", "status": "running"}

@app.get("/auth/health")
def health():
    return {
        "status": "healthy",
        "database": "not connected (test mode)",
        "message": "Server is operational"
    }

@app.post("/auth/register")
def register(user: UserCreate):
    print(f"[TEST] Registration attempt: {user.email}")
    return {"message": f"Test registration for {user.name} received"}

@app.post("/auth/login")
def login(credentials: UserLogin):
    print(f"[TEST] Login attempt: {credentials.email}")
    return {
        "user": {
            "id": "test123",
            "name": "Test User",
            "email": credentials.email,
            "role": "farmer"
        },
        "access_token": "test_token_12345",
        "token_type": "bearer"
    }

print("[SERVER] Auth server module loaded successfully", file=sys.stderr)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting ultra-minimal auth server...")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
