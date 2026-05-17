#!/usr/bin/env python
"""
SmartAgri-AI Startup Script for Render Deployment
=================================================

Purpose:
  1. Read PORT from environment variable (Render sets this dynamically)
  2. Start FastAPI app on the correct port
  3. Ensure port binds BEFORE attempting heavy operations
  4. Proper event loop configuration for 512MB memory constraint

Usage (in Dockerfile):
  CMD ["python", "startup_render.py"]

Environment Variables:
  - PORT: Dynamic port assigned by Render (default 8000)
  - LOW_MEMORY_MODE: Skip heavy model loading (default true)
  - MONGODB_URL: MongoDB Atlas connection string
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main startup function - handles PORT correctly for Render
    """
    
    # CRITICAL: Read PORT from environment
    # Render assigns a dynamic PORT value
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        logger.error(f"Invalid PORT value: {port_str}. Using default 8000")
        port = 8000
    
    host = os.getenv("HOST", "0.0.0.0")
    
    # Verify .env exists for local dev (not needed on Render, but safe)
    env_file = Path(".env")
    has_env = env_file.exists()
    
    logger.info("=" * 60)
    logger.info("SmartAgri-AI Backend Startup")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Low Memory Mode: {os.getenv('LOW_MEMORY_MODE', 'true')}")
    logger.info(f"Environment File (.env): {'Found' if has_env else 'Not found (using system env)'}")
    logger.info("=" * 60)
    
    # Import uvicorn and run
    try:
        import uvicorn
        
        # Run with optimized settings for Render
        uvicorn.run(
            "main_fastapi:app",
            host=host,
            port=port,
            workers=1,  # Single worker for 512MB constraint
            loop="uvloop",  # Faster event loop (uvloop in requirements.txt)
            log_level="info",
            access_log=True,  # Log all requests
            server_header=True,
            date_header=True,
            env_file=".env" if has_env else None
        )
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
