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

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application with uvicorn in production mode
# Uses 4 workers for better performance
CMD uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT} --workers 4
