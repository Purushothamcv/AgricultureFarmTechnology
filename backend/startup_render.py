#!/usr/bin/env python
"""
SmartAgri-AI Production Startup Script for Render
================================================

CRITICAL REQUIREMENTS:
1. Read PORT from environment FIRST
2. Bind to port IMMEDIATELY
3. Handle ALL errors gracefully
4. Never block before port is bound
5. Print clear debug output

This script MUST succeed or print why it failed.
"""

import os
import sys
import time

def main():
    """
    Production startup with comprehensive error handling.
    """
    
    print("\n" + "="*70)
    print("[STARTUP] SmartAgri Backend - Render Production")
    print("="*70)
    
    # Step 1: Get configuration
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
        print(f"[OK] PORT from environment: {port}")
    except (ValueError, TypeError):
        print(f"[WARN] Invalid PORT '{port_str}', using 8000")
        port = 8000
    
    host = "0.0.0.0"
    print(f"[OK] HOST: {host}")
    print(f"[OK] LOW_MEMORY_MODE: {os.getenv('LOW_MEMORY_MODE', 'true')}")
    print(f"[OK] ENVIRONMENT: {os.getenv('ENVIRONMENT', 'production')}")
    print("="*70)
    
    # Step 2: Import uvicorn
    print("\n[INIT] Importing uvicorn...")
    try:
        import uvicorn
        print("[OK] uvicorn imported")
    except ImportError as e:
        print(f"[ERROR] Failed to import uvicorn: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error with uvicorn: {e}")
        sys.exit(1)
    
    # Step 3: Run uvicorn
    print(f"\n[STARTUP] Starting server on {host}:{port}")
    print("-"*70)
    
    try:
        # Minimal, stable parameters only
        uvicorn.run(
            app="main_fastapi:app",
            host=host,
            port=port,
            workers=1,
        )
    except SystemExit as e:
        # Normal shutdown
        sys.exit(e.code if e.code else 0)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] FATAL: {e}")
        import traceback
        traceback.print_exc()
        print("\n[ERROR] Error occurred before port binding")
        print("[ERROR] Check main_fastapi.py imports and app creation")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL] {e}")
        sys.exit(1)
