# ============================================
# SmartAgri Backend - Production Dockerfile
# ============================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data if they don't exist
RUN mkdir -p model data static templates

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port (default, overridden by Render's PORT env var)
EXPOSE 8000

# Health check disabled for Render deployment (uses dynamic ports)
# Render has its own health check mechanism

# Run application with uvicorn in production mode
# Render provides PORT dynamically, bind to all interfaces
# Use explicit port from PORT env var, fallback to 8000
CMD ["sh", "-c", "python -m uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000}"]
