# ============================================
# SmartAgri Backend - Production Dockerfile (Optimized for Render Free Tier)
# ============================================
# Optimized for 512MB memory limit with lazy-loaded ML models

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies for ML libraries
# Avoid unnecessary packages to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with cache optimization
# Use pip cache to speed up builds
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data if they don't exist
RUN mkdir -p model data static templates

# Set environment variables for production and memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV ENVIRONMENT=production
ENV LOW_MEMORY_MODE=true
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose port (default, overridden by Render's PORT env var)
EXPOSE 8000

# Health check disabled for Render deployment (uses dynamic ports)
# Render has its own health check mechanism

# Run application with uvicorn in production mode with low-memory optimizations
# - workers=1: Single worker to save memory
# - timeout-keep-alive=5: Aggressive keep-alive timeout
# - timeout-notify=30: Shorter shutdown timeout
CMD ["sh", "-c", "python -m uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 5 --timeout-notify 30"]
