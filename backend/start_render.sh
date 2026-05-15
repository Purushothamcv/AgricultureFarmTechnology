#!/bin/bash
# ============================================
# RENDER DEPLOYMENT STARTUP SCRIPT
# ============================================
# This script is used by Render to start the SmartAgri-AI backend
# with optimized memory settings and proper port binding

set -e

echo "[STARTUP] SmartAgri-AI Backend - Render Deployment"
echo "=================================================="

# Environment configuration
export ENVIRONMENT=production
export RENDER=true
export LOW_MEMORY_MODE=true
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Get PORT from environment or default to 8000
PORT=${PORT:-8000}
echo "[CONFIG] PORT: $PORT"
echo "[CONFIG] ENVIRONMENT: production"
echo "[CONFIG] LOW_MEMORY_MODE: enabled"
echo "[CONFIG] TensorFlow logging: suppressed"

# Verify Python and dependencies
echo "[CHECK] Python version:"
python --version

echo "[CHECK] Installed packages:"
pip list | grep -E "fastapi|uvicorn|tensorflow|tensorflow|pymongo|motor" || true

# Start the backend with optimal settings
echo "[START] Starting FastAPI backend..."
echo "=================================================="

# Use uvicorn with optimized settings for low memory:
# - 1 worker (multi-worker would increase memory usage)
# - No access log (reduce I/O)
# - Limited request timeout
exec uvicorn main_fastapi:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --access-log \
    --timeout-keep-alive 5 \
    --timeout-notify 30
