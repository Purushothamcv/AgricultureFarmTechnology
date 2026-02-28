"""
Soil Data Service
Fetches soil characteristics from external APIs for map-based location selection
"""

import requests
from typing import Dict, Optional
import time


class SoilDataService:
    """Service to fetch real soil data from external APIs"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = 86400  # 24 hours in seconds
    
    def get_soil_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch comprehensive soil data for given coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with soil characteristics
        """
        # Check cache first
        cache_key = f"{latitude:.4f},{longitude:.4f}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                print(f"📦 Using cached soil data for {cache_key}")
                return cached_data
        
        result = {
            "soil_pH": None,
            "soil_moisture": None,
            "organic_matter": None,
            "soil_type": None,
            "elevation": None,
            "soil_texture": None,
            "organic_carbon": None,
            "electrical_conductivity": None
        }
        
        # 1. Fetch soil properties from SoilGrids API (ISRIC)
        try:
            soil_data = self._fetch_soilgrids_data(latitude, longitude)
            if soil_data:
                result.update(soil_data)
        except Exception as e:
            print(f"⚠️ SoilGrids API error: {e}")
        
        # 2. Fetch elevation from OpenTopography/OpenElevation API
        try:
            elevation = self._fetch_elevation(latitude, longitude)
            if elevation is not None:
                result["elevation"] = elevation
        except Exception as e:
            print(f"⚠️ Elevation API error: {e}")
        
        # 3. Estimate soil moisture if not available (using simple heuristics)
        if result["soil_moisture"] is None:
            try:
                estimated_moisture = self._estimate_soil_moisture(latitude, longitude)
                if estimated_moisture is not None:
                    result["soil_moisture"] = estimated_moisture
            except Exception as e:
                print(f"⚠️ Soil moisture estimation error: {e}")
        
        # Cache the result
        self.cache[cache_key] = (result, time.time())
        
        return result
    
    def _fetch_soilgrids_data(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Fetch soil properties from ISRIC SoilGrids API
        Documentation: https://rest.isric.org/soilgrids/v2.0/docs
        """
        try:
            # SoilGrids REST API endpoint
            base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
            
            # Request soil properties at different depths
            # We'll use 0-5cm depth for surface soil characteristics
            params = {
                "lat": latitude,
                "lon": longitude,
                "property": [
                    "phh2o",      # pH in water
                    "soc",        # Soil organic carbon
                    "clay",       # Clay content
                    "sand",       # Sand content
                    "silt",       # Silt content
                    "cec",        # Cation exchange capacity
                    "bdod",       # Bulk density
                ],
                "depth": "0-5cm",
                "value": "mean"
            }
            
            response = requests.get(
                base_url,
                params=params,
                timeout=10,
                headers={'User-Agent': 'SmartAgri-SoilData/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {}
                
                # Extract pH (convert from pH*10 to pH)
                ph_data = data.get('properties', {}).get('layers', [])
                for prop in ph_data:
                    if prop.get('name') == 'phh2o':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            ph_value = depths[0].get('values', {}).get('mean')
                            if ph_value is not None:
                                result["soil_pH"] = round(ph_value / 10, 2)  # Convert from pH*10 to pH
                                break
                
                # Extract soil organic carbon (convert to organic matter)
                for prop in ph_data:
                    if prop.get('name') == 'soc':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            soc_value = depths[0].get('values', {}).get('mean')
                            if soc_value is not None:
                                # Convert dg/kg to g/kg, then to percentage
                                soc_percent = soc_value / 10 / 10  # dg/kg to %
                                result["organic_carbon"] = round(soc_percent, 2)
                                # Convert organic carbon to organic matter (approx. OM = OC * 1.724)
                                result["organic_matter"] = round(soc_percent * 1.724, 2)
                                break
                
                # Extract soil texture components
                clay_content = None
                sand_content = None
                silt_content = None
                
                for prop in ph_data:
                    if prop.get('name') == 'clay':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            clay_value = depths[0].get('values', {}).get('mean')
                            if clay_value is not None:
                                clay_content = clay_value / 10  # Convert g/kg to %
                    
                    elif prop.get('name') == 'sand':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            sand_value = depths[0].get('values', {}).get('mean')
                            if sand_value is not None:
                                sand_content = sand_value / 10  # Convert g/kg to %
                    
                    elif prop.get('name') == 'silt':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            silt_value = depths[0].get('values', {}).get('mean')
                            if silt_value is not None:
                                silt_content = silt_value / 10  # Convert g/kg to %
                
                # Determine soil type from texture
                if all(v is not None for v in [clay_content, sand_content, silt_content]):
                    soil_type = self._classify_soil_texture(clay_content, sand_content, silt_content)
                    result["soil_type"] = soil_type
                    result["soil_texture"] = f"Clay: {clay_content:.1f}%, Sand: {sand_content:.1f}%, Silt: {silt_content:.1f}%"
                
                # Extract CEC for electrical conductivity estimation
                for prop in ph_data:
                    if prop.get('name') == 'cec':
                        depths = prop.get('depths', [])
                        if depths and len(depths) > 0:
                            cec_value = depths[0].get('values', {}).get('mean')
                            if cec_value is not None:
                                # Rough estimation: EC (dS/m) ≈ CEC / 50
                                # This is a very rough approximation
                                result["electrical_conductivity"] = round((cec_value / 10) / 50, 2)
                                break
                
                print(f"✅ SoilGrids data fetched successfully")
                return result
            
            else:
                print(f"⚠️ SoilGrids API returned status {response.status_code}")
                return None
                
        except requests.Timeout:
            print("⚠️ SoilGrids API timeout")
            return None
        except Exception as e:
            print(f"⚠️ SoilGrids API error: {e}")
            return None
    
    def _classify_soil_texture(self, clay: float, sand: float, silt: float) -> str:
        """
        Classify soil type based on USDA soil texture triangle
        
        Args:
            clay: Clay percentage (0-100)
            sand: Sand percentage (0-100)
            silt: Silt percentage (0-100)
            
        Returns:
            Soil type classification
        """
        # Simplified USDA texture classification
        if clay >= 40:
            return "Clay"
        elif clay >= 27 and sand >= 20 and sand <= 45:
            return "Clay Loam"
        elif clay >= 27 and sand > 45:
            return "Sandy Clay"
        elif clay >= 20 and clay < 35 and silt >= 28 and sand <= 45:
            return "Loam"
        elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
            return "Silt Loam"
        elif silt >= 80 and clay < 12:
            return "Silt"
        elif sand >= 85 and clay <= 10:
            return "Sand"
        elif sand >= 70 and sand < 85 and clay <= 15:
            return "Sandy Loam"
        elif sand >= 45 and sand < 85 and clay <= 20:
            return "Sandy Loam"
        elif clay >= 7 and clay < 20 and sand >= 52:
            return "Sandy Loam"
        elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
            return "Loam"
        else:
            return "Loamy"  # Default fallback
    
    def _fetch_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Fetch elevation data from Open-Elevation API
        Documentation: https://open-elevation.com/
        """
        try:
            url = "https://api.open-elevation.com/api/v1/lookup"
            params = {
                "locations": f"{latitude},{longitude}"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results and len(results) > 0:
                    elevation = results[0].get('elevation')
                    print(f"✅ Elevation data fetched: {elevation}m")
                    return elevation
            
            return None
            
        except Exception as e:
            print(f"⚠️ Elevation fetch error: {e}")
            return None
    
    def _estimate_soil_moisture(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Estimate soil moisture based on weather data
        This is a rough estimation - actual soil moisture sensors would be more accurate
        """
        try:
            # Try to get recent weather data
            import os
            weather_api_key = os.getenv('OPENWEATHER_API_KEY', '90e50f067196b6d46932c52869d83ed6')
            
            if not weather_api_key:
                return None
            
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                weather_data = response.json()
                
                # Get relevant parameters
                humidity = weather_data.get('main', {}).get('humidity', 70)
                temp = weather_data.get('main', {}).get('temp', 25)
                rain_1h = weather_data.get('rain', {}).get('1h', 0)
                
                # Simple moisture estimation (0-100 scale)
                # Higher humidity + recent rain = higher moisture
                # Higher temperature = lower moisture (evaporation)
                
                base_moisture = humidity * 0.5  # Start with humidity influence
                
                # Add rain influence (0-10mm rain adds up to 20% moisture)
                rain_factor = min(rain_1h * 2, 20)
                
                # Temperature reduction (high temp reduces moisture)
                temp_factor = max(0, (35 - temp) / 35 * 15)  # up to 15% from ideal temp
                
                estimated_moisture = base_moisture + rain_factor + temp_factor
                estimated_moisture = min(100, max(0, estimated_moisture))  # Clamp to 0-100
                
                print(f"✅ Estimated soil moisture: {estimated_moisture:.1f}%")
                return round(estimated_moisture, 1)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Soil moisture estimation error: {e}")
            return None


# Singleton instance
_soil_data_service = None

def get_soil_data_service() -> SoilDataService:
    """Get singleton instance of SoilDataService"""
    global _soil_data_service
    if _soil_data_service is None:
        _soil_data_service = SoilDataService()
    return _soil_data_service
