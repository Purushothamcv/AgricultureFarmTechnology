#!/usr/bin/env python
"""
SmartAgri-AI Startup Script for Render Deployment
=================================================

Minimalist production startup for Render free tier.
Prioritizes stability and immediate port binding.

CRITICAL: This script MUST:
1. Read PORT from environment variable
2. Bind to port IMMEDIATELY
3. Not block before port binding
4. Handle all exceptions gracefully
"""

import os
import sys

def main():
    """
    Main entry point for Render deployment.
    Simple, stable, minimal configuration.
    """
    
    # Get port - Render sets PORT in environment
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        print(f"[WARN] Invalid PORT: {port_str}, using 8000")
        port = 8000
    
    host = "0.0.0.0"
    
    print("\n" + "="*60)
    print(f"[STARTUP] SmartAgri Backend")
    print(f"[STARTUP] Binding to {host}:{port}")
    print(f"[STARTUP] LOW_MEMORY_MODE: {os.getenv('LOW_MEMORY_MODE', 'true')}")
    print("="*60 + "\n")
    
    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed")
        sys.exit(1)
    
    # Run uvicorn with MINIMAL settings for stability
    try:
        uvicorn.run(
            app="main_fastapi:app",
            host=host,
            port=port,
            # Minimal, stable settings only:
            workers=1,  # Single worker for 512MB
            log_level="info",
            access_log=False,  # Disable access log (less I/O)
            # REMOVED: loop, interface, reload, and other settings
            # These can cause compatibility issues on Render
        )
    except SystemExit:
        # uvicorn.run calls sys.exit - let it propagate
        raise
    except KeyboardInterrupt:
        print("\n[INFO] Shutdown signal received")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
