"""
Minimal FastAPI server with ONLY authentication endpoints
For testing auth functionality without ML model loading delays
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from database import connect_to_mongodb, close_mongodb_connection

app = FastAPI(
    title="SmartAgri Auth Service",
    description="Authentication-only service for testing",
    version="1.0.0"
)

# CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount authentication router
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection ONLY"""
    print("Starting Auth-Only Service...")
    await connect_to_mongodb()
    print("Auth service ready!")

 # Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection"""
    await close_mongodb_connection()
    print("Auth service stopped")

# Health check
@app.get("/")
async def root():
    return {
        "service": "SmartAgri Auth Service",
        "status": "running",
        "endpoints": ["/auth/register", "/auth/login", "/auth/health"]
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting Authentication-Only Server")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
