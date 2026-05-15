"""
Production Logging Configuration
==================================
Configures logging for production deployment to reduce verbosity and memory usage.
Suppresses TensorFlow, Matplotlib, and other noisy loggers.
"""

import os
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress verbose libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('motor').setLevel(logging.WARNING)

# Production mode check
PRODUCTION_MODE = os.getenv('ENVIRONMENT', 'development') == 'production' or os.getenv('RENDER', False)

if PRODUCTION_MODE:
    # Only show errors and warnings in production
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
