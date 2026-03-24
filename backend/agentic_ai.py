"""
Agentic AI Crop Fetching Module
Uses APY.csv + Groq LLM to return relevant crops for a given state/district.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from groq import Groq

router = APIRouter(tags=["Agentic AI"])


class AgenticCropService:
    """Service that combines dataset filtering with Groq-based ranking."""

    def __init__(self, csv_path: str = "data/APY.csv"):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.client: Optional[Groq] = None
        self.cache: Dict[Tuple[str, str, int], List[str]] = {}

    def startup(self) -> None:
        """Load dataset and initialize Groq client."""
        self._load_dataset()
        if self.groq_api_key:
            self.client = Groq(api_key=self.groq_api_key)
            print(f"🤖 Agentic AI Groq client ready ({self.groq_model})")
        else:
            print("⚠️  Agentic AI: GROQ_API_KEY not set, using dataset-only fallback")

    def _load_dataset(self) -> None:
        """Load and clean APY dataset once at startup."""
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"APY dataset not found: {self.csv_path}")

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        required_cols = ["State", "District", "Crop"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"APY dataset missing required columns: {missing_cols}")

        # Dataset cleaning as requested.
        df = df.dropna(subset=["Crop", "State"])
        df["State"] = df["State"].astype(str).str.strip().str.title()
        df["District"] = df["District"].astype(str).str.strip().str.title()
        df["Crop"] = df["Crop"].astype(str).str.strip().str.title()
        df = df[df["Crop"] != ""]
        df = df[df["State"] != ""]

        self.df = df
        print(f"✅ Agentic AI dataset loaded: {len(df):,} rows")

    def _normalize(self, value: Optional[str]) -> str:
        return (value or "").strip().title()

    def get_crops_from_dataset(self, state: str, district: Optional[str] = None) -> List[str]:
        """
        Return unique crops from APY dataset filtered by state and optional district.
        """
        if self.df is None:
            self._load_dataset()

        normalized_state = self._normalize(state)
        normalized_district = self._normalize(district)

        df_filtered = self.df[self.df["State"].str.lower() == normalized_state.lower()]
        if normalized_district:
            df_filtered = df_filtered[
                df_filtered["District"].str.lower() == normalized_district.lower()
            ]

        crops = df_filtered["Crop"].dropna().unique().tolist()
        return sorted(crops)

    def _parse_llm_crops(self, content: str, allowed_crops: Optional[List[str]] = None) -> List[str]:
        """Parse LLM crop JSON. If allowed_crops is provided, validate against that set."""
        if not content:
            return []

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        allowed_lookup = {c.lower(): c for c in (allowed_crops or [])}

        try:
            payload = json.loads(cleaned)
            llm_crops = payload.get("crops", []) if isinstance(payload, dict) else []
            validated: List[str] = []
            for item in llm_crops:
                candidate = str(item).strip().title()
                key = candidate.lower()
                if not key:
                    continue

                if allowed_lookup:
                    if key in allowed_lookup and allowed_lookup[key] not in validated:
                        validated.append(allowed_lookup[key])
                else:
                    if candidate not in validated:
                        validated.append(candidate)
            return validated
        except Exception:
            # Fallback: pick crops mentioned in plain text response.
            response_lower = cleaned.lower()

            if allowed_crops:
                mentioned = [crop for crop in allowed_crops if crop.lower() in response_lower]
            else:
                mentioned = [part.strip().title() for part in cleaned.replace("\n", ",").split(",") if part.strip()]

            unique_mentioned: List[str] = []
            for crop in mentioned:
                if crop not in unique_mentioned:
                    unique_mentioned.append(crop)
            return unique_mentioned

    def get_ai_crop_recommendations(
        self,
        state: str,
        district: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get crops via dataset + Groq filtering.

        Fallback chain:
        1) Dataset filtered crops
        2) If Groq fails/unavailable, return dataset crops directly
        """
        if self.df is None:
            self._load_dataset()

        max_total = max(1, min(limit, 10))
        dataset_limit = 5
        llm_limit = 5
        normalized_state = self._normalize(state)
        normalized_district = self._normalize(district)

        dataset_crops_all = self.get_crops_from_dataset(normalized_state, normalized_district)
        dataset_crops = dataset_crops_all[:dataset_limit]

        if not dataset_crops_all:
            fallback_crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
            return {
                "crops": fallback_crops[:max_total],
                "dataset_crops": fallback_crops[:dataset_limit],
                "llm_crops": [],
                "source": "fallback",
                "message": "Dataset returned no crops; using fallback crops"
            }

        cache_key = (normalized_state.lower(), normalized_district.lower(), max_total)
        if cache_key in self.cache:
            return {
                "crops": self.cache[cache_key],
                "source": "cache"
            }

        if not self.client:
            direct = dataset_crops[:max_total]
            self.cache[cache_key] = direct
            return {
                "crops": direct,
                "dataset_crops": dataset_crops,
                "llm_crops": [],
                "source": "dataset"
            }

        prompt = (
            f"Given the agricultural conditions in {normalized_state}"
            f"{', ' + normalized_district if normalized_district else ''}, "
            f"suggest {llm_limit} additional suitable crops for cultivation. "
            f"Do NOT repeat crops from this list: {dataset_crops}. "
            f"Return ONLY valid JSON in this exact format: "
            f"{{\"crops\": [\"Crop1\", \"Crop2\", \"Crop3\", \"Crop4\", \"Crop5\"]}}"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an agricultural assistant. Only select crops from the provided list."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=180
            )

            response_text = completion.choices[0].message.content or ""
            print(f"🤖 Agentic AI raw LLM response: {response_text}")

            llm_crops = self._parse_llm_crops(response_text)[:llm_limit]
            final_crops = list(dict.fromkeys(dataset_crops + llm_crops))[:max_total]
            self.cache[cache_key] = final_crops

            return {
                "crops": final_crops,
                "dataset_crops": dataset_crops,
                "llm_crops": llm_crops,
                "source": "hybrid"
            }
        except Exception as e:
            print(f"⚠️  Agentic AI Groq error: {e}")
            direct = dataset_crops[:max_total]
            self.cache[cache_key] = direct
            return {
                "crops": direct,
                "dataset_crops": dataset_crops,
                "llm_crops": [],
                "source": "dataset",
                "message": "Groq unavailable; returned dataset crops"
            }


_agentic_service: Optional[AgenticCropService] = None


def get_agentic_service() -> AgenticCropService:
    global _agentic_service
    if _agentic_service is None:
        _agentic_service = AgenticCropService()
    return _agentic_service


async def startup_event() -> None:
    """Initialize agentic AI service at app startup."""
    print("🤖 Initializing Agentic AI Crop Service...")
    service = get_agentic_service()
    service.startup()


@router.get("/agent/crops")
async def get_agent_crops(
    state: str = Query(..., min_length=1),
    district: Optional[str] = Query(None),
):
    """Agentic endpoint: returns top relevant crops for state/district."""
    try:
        print("State:", state)
        print("District:", district)
        service = get_agentic_service()
        result = service.get_ai_crop_recommendations(state=state, district=district, limit=10)
        print("Crops found:", result.get("crops", []))

        if not result.get("crops"):
            return {
                "crops": [],
                "message": "No crops found for this location"
            }

        return {
            "crops": result["crops"],
            "source": result.get("source", "unknown"),
            **({"message": result["message"]} if "message" in result else {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic crop fetch failed: {str(e)}")
