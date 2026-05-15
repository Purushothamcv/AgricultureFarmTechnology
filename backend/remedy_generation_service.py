"""
Disease Remedy Generation Service
==================================
Uses comprehensive disease database and Groq LLM to provide detailed disease remedies, pesticide suggestions, and prevention tips.

Features:
- Uses comprehensive disease information database (primary)
- Dynamic remedy generation using Groq LLM (if disease not in database)
- Disease-specific agricultural guidance
- Pesticide/fungicide recommendations
- Prevention tips for farmers
- Graceful fallback to generic info if unavailable
"""

import logging
import os
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import disease database
try:
    from disease_information_db import DISEASE_DATABASE, get_disease_info, get_healthy_plant_info
    logger.info("✅ Disease information database loaded successfully")
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import disease database: {e}. Using LLM only.")
    DATABASE_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq's fastest model

# Initialize Groq client
groq_client = None

# ============================================================================
# DATA MODELS
# ============================================================================

class DiseaseRemedyRequest(BaseModel):
    crop: str
    disease: str
    isHealthy: bool = False

class RemedyResponse(BaseModel):
    remedy: str
    pesticide: str
    action: str
    prevention: str
    source: str = "database"  # "database", "llm", or "fallback"

# ============================================================================
# STATIC FALLBACK REMEDIES
# ============================================================================

FALLBACK_REMEDIES = {
    'Early blight': {
        'remedy': 'Use fungicides like chlorothalonil or mancozeb. Remove infected leaves immediately.',
        'pesticide': 'Chlorothalonil, Mancozeb, Azoxystrobin',
        'action': 'Remove infected leaves and improve air circulation.'
    },
    'Late blight': {
        'remedy': 'Apply copper-based fungicides immediately to prevent spread.',
        'pesticide': 'Copper sulfate, Metalaxyl, Phosphonites',
        'action': 'Avoid overhead irrigation and ensure proper drainage.'
    },
    'Septoria leaf spot': {
        'remedy': 'Use sulfur-based or copper fungicides weekly.',
        'pesticide': 'Sulfur, Copper, Chlorothalonil',
        'action': 'Remove affected leaves and improve air circulation.'
    },
    'Powdery mildew': {
        'remedy': 'Apply sulfur dust or neem oil sprays every 7-10 days.',
        'pesticide': 'Sulfur, Neem oil, Potassium bicarbonate',
        'action': 'Reduce humidity and ensure adequate spacing between plants.'
    },
    'Leaf spot': {
        'remedy': 'Use neem oil or sulfur sprays for organic control.',
        'pesticide': 'Neem oil, Sulfur, Copper',
        'action': 'Remove infected leaves and maintain good hygiene.'
    },
    'Healthy': {
        'remedy': 'No treatment needed. The plant is in excellent condition.',
        'pesticide': 'N/A - No pesticide required',
        'action': 'Continue regular maintenance and monitoring for early disease signs.'
    }
}

# ============================================================================
# GROQ LLM INTEGRATION
# ============================================================================

def initialize_groq():
    """Initialize Groq client."""
    global groq_client
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("✅ Groq client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Groq: {e}")
            groq_client = None
    else:
        logger.warning("⚠️  GROQ_API_KEY not set. Using fallback remedies.")
        groq_client = None

def generate_remedy_with_groq(crop: str, disease: str, is_healthy: bool) -> Optional[Dict[str, str]]:
    """
    Generate disease remedy using Groq LLM.
    
    Args:
        crop: Crop name (e.g., "Tomato", "Apple")
        disease: Disease name (e.g., "Early Blight", "Apple Scab")
        is_healthy: Whether the plant is healthy
        
    Returns:
        Dictionary with remedy, pesticide, and action fields
        Returns None if API call fails
    """
    if not groq_client:
        return None
    
    try:
        if is_healthy:
            prompt = f"""You are an agricultural expert helping farmers.

The {crop} plant is HEALTHY with no disease detected.

Provide concise farmer-friendly advice:
1. Basic maintenance tips (1-2 sentences)
2. Recommended preventive measures (1-2 sentences)
3. General care routine (1-2 sentences)

Keep the response brief and practical. Format as:
Remedy: [maintenance tips]
Pesticide: None - preventive monitoring only
Action: [general care routine]"""
        else:
            prompt = f"""You are an agricultural expert helping farmers.

Provide specific treatment for:
Crop: {crop}
Disease: {disease}

Give concise farmer-friendly advice:
1. Treatment recommendation (1-2 sentences)
2. Specific pesticide/fungicide name and application frequency
3. Practical action steps (1-2 sentences)

Format your response as:
Remedy: [treatment recommendation]
Pesticide: [specific product names and frequency]
Action: [practical steps to take]"""

        message = groq_client.messages.create(
            model=MODEL_NAME,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response_text = message.content[0].text
        logger.info(f"✅ Generated remedy for {crop} - {disease}")
        
        # Parse response
        remedy = parse_groq_response(response_text)
        if remedy:
            return remedy
            
    except Exception as e:
        logger.error(f"❌ Error generating remedy with Groq: {e}")
        return None
    
    return None

def parse_groq_response(response_text: str) -> Optional[Dict[str, str]]:
    """
    Parse Groq LLM response into structured format.
    
    Args:
        response_text: Raw response from Groq
        
    Returns:
        Dictionary with remedy, pesticide, and action
    """
    try:
        lines = response_text.strip().split('\n')
        remedy_dict = {
            'remedy': '',
            'pesticide': '',
            'action': ''
        }
        
        for line in lines:
            if line.startswith('Remedy:'):
                remedy_dict['remedy'] = line.replace('Remedy:', '').strip()
            elif line.startswith('Pesticide:'):
                remedy_dict['pesticide'] = line.replace('Pesticide:', '').strip()
            elif line.startswith('Action:'):
                remedy_dict['action'] = line.replace('Action:', '').strip()
        
        # Ensure all fields have content
        if remedy_dict['remedy'] and remedy_dict['pesticide'] and remedy_dict['action']:
            return remedy_dict
            
    except Exception as e:
        logger.warning(f"Failed to parse Groq response: {e}")
    
    return None

# ============================================================================
# FASTAPI ROUTER
# ============================================================================

router = APIRouter(prefix="/api", tags=["Disease Remedy Generation"])

@router.post("/generate-disease-remedy")
async def generate_disease_remedy(request: DiseaseRemedyRequest) -> RemedyResponse:
    """
    Generate disease remedy with comprehensive information.
    
    Priority:
    1. Check comprehensive disease database first (PRIMARY - detailed farmer-friendly info)
    2. Try Groq LLM if not in database
    3. Fall back to static remedies if LLM fails
    
    Args:
        request: DiseaseRemedyRequest with crop, disease, and isHealthy
        
    Returns:
        RemedyResponse with remedy, pesticide, prevention, action, and source
    """
    try:
        crop = request.crop.strip()
        disease = request.disease.strip()
        is_healthy = request.isHealthy
        
        logger.info(f"🔄 Generating detailed remedy for {crop} - {disease}")
        
        # ================================================================
        # STEP 1: Check comprehensive disease database
        # ================================================================
        
        if is_healthy or disease.lower() == 'healthy':
            logger.info("✅ Plant is healthy - returning maintenance info from database")
            healthy_info = get_healthy_plant_info()
            return RemedyResponse(
                remedy=healthy_info.get('remedy', ''),
                pesticide=healthy_info.get('pesticide', ''),
                prevention=healthy_info.get('prevention', ''),
                action=healthy_info.get('action', ''),
                source="database"
            )
        
        # Try to find disease in database
        disease_key = f"{disease}_{crop}"
        
        if DATABASE_AVAILABLE:
            disease_info = get_disease_info(disease_key)
            
            # Check if we got actual database info (not generic fallback)
            if disease_key in DISEASE_DATABASE:
                logger.info(f"✅ Found detailed info in database for {disease_key}")
                return RemedyResponse(
                    remedy=disease_info.get('remedy', ''),
                    pesticide=disease_info.get('pesticide', ''),
                    prevention=disease_info.get('prevention', ''),
                    action=disease_info.get('action', ''),
                    source="database"
                )
        
        # ================================================================
        # STEP 2: Try Groq LLM for diseases not in database
        # ================================================================
        
        logger.info(f"ℹ️ Disease not in database, trying Groq LLM...")
        ai_remedy = generate_remedy_with_groq(crop, disease, is_healthy)
        
        if ai_remedy and all(k in ai_remedy for k in ['remedy', 'pesticide', 'action']):
            logger.info(f"✅ Generated detailed remedy using Groq LLM")
            return RemedyResponse(
                remedy=ai_remedy.get('remedy', ''),
                pesticide=ai_remedy.get('pesticide', ''),
                prevention=ai_remedy.get('action', 'N/A'),  # Use action as prevention if not available
                action=ai_remedy.get('action', ''),
                source="llm"
            )
        
        # ================================================================
        # STEP 3: Fall back to static remedies
        # ================================================================
        
        logger.info(f"⚠️ Database and LLM unsuccessful, using fallback remedies")
        fallback_key = disease.title() if not is_healthy else 'Healthy'
        fallback = FALLBACK_REMEDIES.get(fallback_key, FALLBACK_REMEDIES.get('Healthy'))
        
        return RemedyResponse(
            remedy=fallback.get('remedy', ''),
            pesticide=fallback.get('pesticide', ''),
            prevention=fallback.get('action', 'Maintain good field sanitation'),
            action=fallback.get('action', ''),
            source="fallback"
        )
        
    except Exception as e:
        logger.error(f"❌ Error in generate_disease_remedy: {e}")
        # Return helpful error message
        return RemedyResponse(
            remedy="Disease information service encountered an error. Please try again.",
            pesticide="Contact local agricultural extension office",
            prevention="Maintain proper field sanitation and crop rotation practices.",
            action="Consult with agricultural officer for proper diagnosis.",
            source="error"
        )

@router.get("/remedy-health")
async def remedy_service_health():
    """
    Health check endpoint for remedy generation service.
    
    Returns:
        Service status and configuration
    """
    return {
        "status": "healthy",
        "groq_enabled": groq_client is not None,
        "model": MODEL_NAME,
        "groq_api_key_configured": bool(GROQ_API_KEY)
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

async def startup_event():
    """
    Initialize remedy generation service on application startup.
    
    Called automatically when FastAPI starts.
    """
    logger.info("🌿 Initializing Disease Remedy Generation Service...")
    initialize_groq()
    logger.info("✅ Disease Remedy Generation Service initialized")
