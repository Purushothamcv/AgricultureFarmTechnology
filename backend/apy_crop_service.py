"""
APY Dataset-Based Crop Recommendation Service
Provides data-driven crop recommendations based on historical yield data
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


class APYCropService:
    """Service for crop recommendations based on APY (Area, Production, Yield) dataset"""
    
    def __init__(self, csv_path: str = "data/APY.csv"):
        """Initialize service and load APY dataset"""
        self.csv_path = csv_path
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load and clean APY dataset"""
        try:
            # Load CSV
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded APY dataset: {len(self.df)} rows")
            
            # Clean column names (remove trailing spaces)
            self.df.columns = self.df.columns.str.strip()
            
            # Clean string columns (remove trailing spaces)
            string_cols = ['State', 'District', 'Crop', 'Season']
            for col in string_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].str.strip()
            
            # Drop rows where Crop or Yield is missing
            original_count = len(self.df)
            self.df = self.df.dropna(subset=['Crop', 'Yield'])
            dropped = original_count - len(self.df)
            print(f"📊 Dropped {dropped} rows with missing Crop or Yield")
            
            # Remove rows with zero or negative yield
            self.df = self.df[self.df['Yield'] > 0]
            print(f"📊 Final dataset: {len(self.df)} rows")
            
            print(f"📈 Unique States: {self.df['State'].nunique()}")
            print(f"📈 Unique Districts: {self.df['District'].nunique()}")
            print(f"📈 Unique Crops: {self.df['Crop'].nunique()}")
            print(f"📈 Unique Seasons: {self.df['Season'].nunique()}")
            
        except Exception as e:
            print(f"❌ Error loading APY dataset: {e}")
            raise
    
    def get_states(self) -> List[str]:
        """Get list of unique states"""
        if self.df is None:
            return []
        states = sorted(self.df['State'].unique().tolist())
        return states
    
    def get_districts(self, state: Optional[str] = None) -> List[str]:
        """
        Get list of districts
        
        Args:
            state: Filter districts by state (optional)
        
        Returns:
            List of district names
        """
        if self.df is None:
            return []
        
        if state:
            districts = self.df[self.df['State'] == state]['District'].unique().tolist()
        else:
            districts = self.df['District'].unique().tolist()
        
        return sorted(districts)
    
    def get_seasons(self) -> List[str]:
        """Get list of unique seasons"""
        if self.df is None:
            return []
        seasons = sorted(self.df['Season'].unique().tolist())
        return seasons
    
    def get_crops(self) -> List[str]:
        """Get list of unique crops"""
        if self.df is None:
            return []
        crops = sorted(self.df['Crop'].unique().tolist())
        return crops
    
    def recommend_crop(
        self, 
        state: str, 
        district: str, 
        season: str, 
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Recommend crops based on highest average yield for given location and season
        
        Fallback logic:
        1. Try State + District + Season
        2. If no results, try State + Season
        3. If still no results, return "No historical data available"
        
        Args:
            state: State name
            district: District name
            season: Season name
            top_n: Number of top crops to recommend
        
        Returns:
            Dictionary with recommendations or error message
        """
        if self.df is None:
            return {
                "success": False,
                "message": "Dataset not loaded",
                "recommendations": []
            }
        
        # Clean inputs
        state = state.strip()
        district = district.strip()
        season = season.strip()
        
        # Try exact match: State + District + Season
        filtered = self.df[
            (self.df['State'] == state) & 
            (self.df['District'] == district) & 
            (self.df['Season'] == season)
        ]
        
        match_level = "exact"
        
        # Fallback to State + Season if no exact match
        if len(filtered) == 0:
            filtered = self.df[
                (self.df['State'] == state) & 
                (self.df['Season'] == season)
            ]
            match_level = "state_season"
        
        # If still no results, return error
        if len(filtered) == 0:
            return {
                "success": False,
                "message": f"No historical data available for {state}, {district}, {season}",
                "recommendations": [],
                "match_level": "none"
            }
        
        # Calculate average yield per crop
        crop_yields = filtered.groupby('Crop').agg({
            'Yield': 'mean',
            'Area': 'mean',
            'Production': 'mean'
        }).reset_index()
        
        # Sort by yield (descending) and get top N
        crop_yields = crop_yields.sort_values('Yield', ascending=False).head(top_n)
        
        # Format recommendations
        recommendations = []
        for _, row in crop_yields.iterrows():
            recommendations.append({
                "crop": row['Crop'],
                "average_yield": round(row['Yield'], 2),
                "average_area": round(row['Area'], 2),
                "average_production": round(row['Production'], 2)
            })
        
        return {
            "success": True,
            "message": f"Found {len(recommendations)} crop recommendations",
            "recommendations": recommendations,
            "match_level": match_level,
            "state": state,
            "district": district if match_level == "exact" else "All districts",
            "season": season,
            "total_records": len(filtered)
        }


# Global instance (singleton pattern)
_apy_service = None


def get_apy_service() -> APYCropService:
    """Get or create APY service instance"""
    global _apy_service
    if _apy_service is None:
        _apy_service = APYCropService()
    return _apy_service


# Convenience functions for FastAPI endpoints
def get_all_states() -> List[str]:
    """Get all states"""
    service = get_apy_service()
    return service.get_states()


def get_districts_by_state(state: str) -> List[str]:
    """Get districts for a specific state"""
    service = get_apy_service()
    return service.get_districts(state)


def get_all_seasons() -> List[str]:
    """Get all seasons"""
    service = get_apy_service()
    return service.get_seasons()


def get_all_crops() -> List[str]:
    """Get all crops"""
    service = get_apy_service()
    return service.get_crops()


def recommend_crops(state: str, district: str, season: str, top_n: int = 3) -> Dict[str, Any]:
    """Recommend crops based on historical yield data"""
    service = get_apy_service()
    return service.recommend_crop(state, district, season, top_n)
