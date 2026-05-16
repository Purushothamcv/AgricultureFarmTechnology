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

# Install Python dependencies with optimization for Render
# - Upgrade pip for better resolution algorithm
# - Use --no-cache-dir to save disk space
# - Pin all versions to prevent resolver backtracking (especially for SQLAlchemy)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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

# Run application with uvicorn in production mode
# Direct uvicorn call (not python -m) for proper Render integration
# Uses single worker by default for 512MB memory constraint
CMD ["sh", "-c", "uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000}"]
