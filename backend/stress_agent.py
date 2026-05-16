"""
Crop Stress Prediction Agent
Uses Groq LLM to explain ML predictions and provide recommendations
"""

import os
from dotenv import load_dotenv

# LAZY LOAD: LangChain imports are optional
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None
    PromptTemplate = None
    StrOutputParser = None

load_dotenv()

class StressAgent:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize Groq LLM"""
        if not LANGCHAIN_AVAILABLE:
            print("[SKIP] LangChain not available - stress agent will be unavailable")
            return False
        
        try:
            if not self.groq_api_key:
                print("[SKIP] GROQ_API_KEY not found in environment")
                return False
            
            self.llm = ChatGroq(
                temperature=0.3,  # Lower temperature for more consistent explanations
                model="llama-3.1-8b-instant",
                api_key=self.groq_api_key,
                top_p=0.95
            )
            print("[OK] Groq LLM initialized successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error initializing Groq LLM: {e}")
            return False
    
    def generate_stress_explanation(self, input_data: dict, prediction: dict):
        """
        Generate AI-powered explanation for stress prediction
        
        Args:
            input_data: Original input data (crop type, soil, weather, etc.)
            prediction: ML model prediction result
        
        Returns:
            Dictionary with explanation and recommendations
        """
        try:
            if not self.llm:
                return self._get_fallback_explanation(prediction)
            
            # Format input data for prompt
            data_summary = self._format_input_summary(input_data)
            stress_factors = prediction.get('stress_factors', [])
            stress_level = prediction.get('stress_level', 'Unknown')
            confidence = prediction.get('confidence_percentage', '0%')
            
            # Create the prompt template
            prompt_template = PromptTemplate(
                input_variables=[
                    "stress_level", 
                    "confidence", 
                    "factors",
                    "data_summary"
                ],
                template="""You are an agricultural expert helping farmers understand crop stress levels.

**CROP STRESS ANALYSIS**

Predicted Stress Level: {stress_level}
Confidence: {confidence}

**Key Stress Factors Identified:**
{factors}

**Current Conditions:**
{data_summary}

Please provide a concise, farmer-friendly explanation (3-4 sentences) covering:
1. Why the crop is under this stress level
2. Which environmental or soil factors are the main contributors
3. Immediate actions the farmer should take

Keep language simple and avoid technical jargon. Focus on practical, actionable advice.

---
EXPLANATION:"""
            )
            
            # Format stress factors as bullet list
            factors_str = "\n".join([f"• {factor}" for factor in stress_factors])
            
            # Build the chain using modern LCEL syntax
            output_parser = StrOutputParser()
            chain = prompt_template | self.llm | output_parser
            
            # Generate explanation
            response = chain.invoke({
                "stress_level": stress_level,
                "confidence": confidence,
                "factors": factors_str,
                "data_summary": data_summary
            })
            
            explanation = response.strip()
            
            # Generate specific recommendations
            recommendations = self._generate_recommendations(
                input_data, 
                prediction,
                explanation
            )
            
            return {
                "success": True,
                "explanation": explanation,
                "recommendations": recommendations,
                "reasoning_source": "Groq LLM Analysis"
            }
            
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            return self._get_fallback_explanation(prediction)
    
    def _format_input_summary(self, data: dict):
        """Format input data into readable summary"""
        summary_points = []
        
        # Temperature
        temp = data.get('temperature', 'N/A')
        summary_points.append(f"Temperature: {temp}°C")
        
        # Soil Moisture
        moisture = data.get('soil_moisture', 'N/A')
        summary_points.append(f"Soil Moisture: {moisture}%")
        
        # Humidity
        humidity = data.get('humidity', 'N/A')
        summary_points.append(f"Humidity: {humidity}%")
        
        # Rainfall
        rainfall = data.get('rainfall', 'N/A')
        summary_points.append(f"Rainfall: {rainfall}mm")
        
        # Soil pH
        ph = data.get('soil_ph', 'N/A')
        summary_points.append(f"Soil pH: {ph}")
        
        # Wind Speed
        wind = data.get('wind_speed', 'N/A')
        summary_points.append(f"Wind Speed: {wind} km/h")
        
        # Pest Damage
        pest = data.get('pest_damage', 'N/A')
        summary_points.append(f"Pest Damage: {pest}%")
        
        # Weed Coverage
        weed = data.get('weed_coverage', 'N/A')
        summary_points.append(f"Weed Coverage: {weed}%")
        
        return ", ".join(summary_points)
    
    def _generate_recommendations(self, input_data: dict, prediction: dict, explanation: str):
        """Generate actionable recommendations based on stress level and factors"""
        recommendations = []
        stress_level = prediction.get('stress_level', 'Moderate')
        factors = prediction.get('stress_factors', [])
        
        # Map factors to actions
        action_map = {
            "Severe drought stress": {
                "action": "Increase irrigation frequency",
                "priority": "URGENT",
                "urgency": 1
            },
            "Moderate drought stress": {
                "action": "Increase irrigation frequency",
                "priority": "High",
                "urgency": 2
            },
            "Waterlogging risk": {
                "action": "Improve drainage or reduce irrigation",
                "priority": "High",
                "urgency": 2
            },
            "High temperature": {
                "action": "Provide shade using mulching or shade crops",
                "priority": "High",
                "urgency": 2
            },
            "Extreme high temperature": {
                "action": "Provide immediate cooling measures and increase irrigation",
                "priority": "URGENT",
                "urgency": 1
            },
            "Low temperature": {
                "action": "Provide frost protection or mulching",
                "priority": "High",
                "urgency": 2
            },
            "Severe pest damage": {
                "action": "Apply appropriate pesticides immediately",
                "priority": "URGENT",
                "urgency": 1
            },
            "Moderate pest damage": {
                "action": "Apply integrated pest management (IPM) techniques",
                "priority": "High",
                "urgency": 2
            },
            "High weed competition": {
                "action": "Manual weeding or apply herbicides",
                "priority": "High",
                "urgency": 2
            },
            "Acidic soil": {
                "action": "Apply lime to neutralize acidity",
                "priority": "Medium",
                "urgency": 3
            },
            "Alkaline soil": {
                "action": "Add sulfur or organic matter to lower pH",
                "priority": "Medium",
                "urgency": 3
            },
            "Low organic matter": {
                "action": "Add compost or farmyard manure",
                "priority": "Medium",
                "urgency": 3
            },
            "Insufficient rainfall": {
                "action": "Supplement with irrigation",
                "priority": "High",
                "urgency": 2
            },
            "Excessive rainfall": {
                "action": "Improve drainage to prevent waterlogging",
                "priority": "High",
                "urgency": 2
            },
            "Low humidity stress": {
                "action": "Increase irrigation and mulch to retain moisture",
                "priority": "Medium",
                "urgency": 3
            },
            "High humidity (disease risk)": {
                "action": "Improve air circulation and apply fungicides preventively",
                "priority": "Medium",
                "urgency": 3
            },
            "Poor drainage": {
                "action": "Create raised beds or improve soil structure",
                "priority": "High",
                "urgency": 2
            },
            "High wind stress": {
                "action": "Provide windbreaks using hedges or fencing",
                "priority": "Medium",
                "urgency": 3
            },
            "Optimal growing conditions": {
                "action": "Continue current management practices",
                "priority": "Low",
                "urgency": 4
            }
        }
        
        # Collect all relevant recommendations
        collected_recs = {}
        for factor in factors:
            if factor in action_map:
                action_info = action_map[factor]
                priority = action_info['priority']
                
                # Group by priority
                if priority not in collected_recs:
                    collected_recs[priority] = []
                
                collected_recs[priority].append({
                    "factor": factor,
                    "action": action_info['action'],
                    "priority": priority
                })
        
        # Build recommendations in priority order
        priority_order = ["URGENT", "High", "Medium", "Low"]
        for priority in priority_order:
            if priority in collected_recs:
                for rec in collected_recs[priority]:
                    recommendations.append(rec)
        
        # Limit to top recommendations
        return recommendations[:5]
    
    def _get_fallback_explanation(self, prediction: dict):
        """Fallback explanation when LLM is unavailable"""
        stress_level = prediction.get('stress_level', 'Moderate')
        confidence = prediction.get('confidence_percentage', '0%')
        factors = prediction.get('stress_factors', [])
        
        # Template-based fallback
        fallback_templates = {
            'Low': f"""Your crop is in good health with {confidence} confidence. 
            The current conditions are favorable for growth. 
            Continue with regular monitoring and current management practices to maintain this optimal state.
            Top factors: {', '.join(factors[:3]) if factors else 'All conditions are optimal'}.
            """,
            'Moderate': f"""Your crop is under moderate stress ({confidence} confidence) due to various environmental and soil factors. 
            Identified issues include: {', '.join(factors[:3]) if factors else 'multiple stress factors'}.
            Adjust irrigation, monitor pest/disease pressure, and optimize nutrient management to improve crop health.
            """,
            'High': f"""Your crop is under severe stress ({confidence} confidence) and requires immediate intervention. 
            Critical factors: {', '.join(factors[:3]) if factors else 'multiple critical factors'}.
            Take urgent action on irrigation, pest control, and nutrient management to prevent significant yield loss.
            """
        }
        
        explanation = fallback_templates.get(
            stress_level,
            f"Stress level predicted as {stress_level} with {confidence} confidence."
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "recommendations": prediction.get('recommendations', []),
            "reasoning_source": "Template-based Fallback"
        }


# Global instance
_stress_agent = None


def get_stress_agent():
    """Get or create global StressAgent instance"""
    global _stress_agent
    if _stress_agent is None:
        _stress_agent = StressAgent()
    return _stress_agent


def generate_stress_insights(input_data: dict, prediction: dict):
    """
    Main function to generate stress insights using LLM
    
    Args:
        input_data: Original input data
        prediction: ML model prediction result
    
    Returns:
        Dictionary with explanation and recommendations
    """
    agent = get_stress_agent()
    return agent.generate_stress_explanation(input_data, prediction)
