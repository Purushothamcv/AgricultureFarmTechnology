"""
MINIMAL FASTAPI APP FOR RENDER DEPLOYMENT
==========================================
This is a bare-bones version to verify uvicorn works.
Components will be added back one-by-one.
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment early
try:
    load_dotenv()
    print("[OK] Environment loaded")
except Exception as e:
    print(f"[WARN] Failed to load .env: {e}")

# Set TensorFlow suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

print("\n" + "="*70)
print("[BOOTSTRAP] SmartAgri Backend - Minimal Mode")
print("="*70)
print(f"[BOOTSTRAP] PORT: {os.getenv('PORT', 'NOT SET')}")
print(f"[BOOTSTRAP] Environment: {os.getenv('ENVIRONMENT', 'development')}")
print("="*70)

# Core imports only - nothing that blocks
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    print("[OK] FastAPI imported")
except Exception as e:
    print(f"[ERROR] Failed to import FastAPI: {e}")
    traceback.print_exc()
    sys.exit(1)

# Create app IMMEDIATELY
try:
    app = FastAPI(
        title="SmartAgri API",
        description="Smart Agriculture Decision Support System",
        version="1.0.0"
    )
    print("[OK] FastAPI app created")
except Exception as e:
    print(f"[ERROR] Failed to create app: {e}")
    traceback.print_exc()
    sys.exit(1)

# CORS middleware
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all for now
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("[OK] CORS middleware added")
except Exception as e:
    print(f"[WARN] CORS setup failed: {e}")

# Basic health check - MUST BE FAST
@app.get("/health")
async def health():
    """Health check endpoint - Render uses this to detect running app"""
    return {
        "status": "ok",
        "app": "SmartAgri-AI",
        "version": "1.0.0",
        "ready": True
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "message": "SmartAgri API is running",
        "version": "1.0.0"
    }

# Startup event - minimal
@app.on_event("startup")
async def startup_event():
    """Startup confirmation"""
    print("\n" + "="*70)
    print("[STARTUP] FastAPI startup event triggered")
    print("[STARTUP] App is ready to accept requests")
    print("="*70 + "\n")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown confirmation"""
    print("\n[SHUTDOWN] FastAPI shutting down\n")

print("\n[OK] All endpoints registered")
print("[OK] App ready for uvicorn\n")
