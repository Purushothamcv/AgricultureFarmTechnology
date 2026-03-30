"""
Fertilizer Auto-Fill Service
Provides API-based soil and nutrient auto-fill values for fertilizer module.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import requests


class FertilizerAutoFillService:
    """Fetches and estimates fertilizer-related soil values for a location."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[float, float], Tuple[dict, float]] = {}
        self.cache_ttl_seconds = int(os.getenv("FERTILIZER_AUTOFILL_CACHE_TTL", "43200"))
        self.timeout_seconds = float(os.getenv("FERTILIZER_AUTOFILL_TIMEOUT", "0.9"))
        self.open_meteo_url = os.getenv("OPEN_METEO_API_URL", "https://api.open-meteo.com/v1/forecast")
        self.soilgrids_url = os.getenv(
            "SOILGRIDS_API_URL",
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
        )

    def get_auto_fill(self, latitude: float, longitude: float) -> dict:
        """
        Return fertilizer auto-fill values for a coordinate.

        On full failure, returns all-null payload as requested.
        """
        null_payload = self._null_payload()

        try:
            key = (round(float(latitude), 4), round(float(longitude), 4))
            now = time.time()

            cached = self.cache.get(key)
            if cached and (now - cached[1]) < self.cache_ttl_seconds:
                return cached[0]

            weather = self._fetch_open_meteo(latitude, longitude)
            soil = self._fetch_soilgrids(latitude, longitude)

            payload = self._compose_payload(weather, soil)
            self.cache[key] = (payload, now)
            return payload
        except Exception:
            return null_payload

    def _null_payload(self) -> dict:
        return {
            "soil_pH": None,
            "soil_moisture": None,
            "organic_carbon": None,
            "electrical_conductivity": None,
            "nitrogen": None,
            "phosphorus": None,
            "potassium": None,
        }

    def _fetch_open_meteo(self, latitude: float, longitude: float) -> dict:
        try:
            response = requests.get(
                self.open_meteo_url,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,precipitation",
                    "timezone": "auto",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            current = response.json().get("current", {})
            return {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "rainfall": current.get("precipitation"),
            }
        except Exception:
            return {"temperature": None, "humidity": None, "rainfall": None}

    def _fetch_soilgrids(self, latitude: float, longitude: float) -> dict:
        layers = {}

        try:
            response = requests.get(
                self.soilgrids_url,
                params=[
                    ("lat", latitude),
                    ("lon", longitude),
                    ("property", "phh2o"),
                    ("property", "soc"),
                    ("property", "cec"),
                    ("property", "clay"),
                    ("property", "sand"),
                    ("property", "silt"),
                    ("depth", "0-5cm"),
                    ("value", "mean"),
                ],
                headers={"User-Agent": "SmartAgri-Fertilizer/1.0"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            for layer in response.json().get("properties", {}).get("layers", []):
                layers[layer.get("name")] = layer
        except Exception:
            return {
                "ph": None,
                "soc_pct": None,
                "cec": None,
                "texture": None,
                "nitrogen": None,
            }

        ph = self._extract_layer_mean(layers.get("phh2o"))
        soc = self._extract_layer_mean(layers.get("soc"))
        cec = self._extract_layer_mean(layers.get("cec"))
        clay = self._extract_layer_mean(layers.get("clay"))
        sand = self._extract_layer_mean(layers.get("sand"))
        silt = self._extract_layer_mean(layers.get("silt"))

        ph_value = round(ph / 10.0, 2) if ph is not None else None
        soc_pct = round((soc / 100.0), 2) if soc is not None else None
        texture = self._classify_texture(clay, sand, silt)

        nitrogen = self._fetch_optional_nitrogen(latitude, longitude)

        return {
            "ph": ph_value,
            "soc_pct": soc_pct,
            "cec": cec,
            "texture": texture,
            "nitrogen": nitrogen,
        }

    def _fetch_optional_nitrogen(self, latitude: float, longitude: float) -> Optional[float]:
        """Try to fetch nitrogen from SoilGrids if available; ignore on failure."""
        try:
            response = requests.get(
                self.soilgrids_url,
                params=[
                    ("lat", latitude),
                    ("lon", longitude),
                    ("property", "nitrogen"),
                    ("depth", "0-5cm"),
                    ("value", "mean"),
                ],
                headers={"User-Agent": "SmartAgri-Fertilizer/1.0"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            layers = response.json().get("properties", {}).get("layers", [])
            for layer in layers:
                if layer.get("name") == "nitrogen":
                    value = self._extract_layer_mean(layer)
                    if value is not None:
                        return round(float(value), 2)
            return None
        except Exception:
            return None

    def _extract_layer_mean(self, layer: Optional[dict]) -> Optional[float]:
        if not layer:
            return None
        depths = layer.get("depths", [])
        if not depths:
            return None
        value = depths[0].get("values", {}).get("mean")
        if value is None:
            return None
        return float(value)

    def _classify_texture(
        self,
        clay_g_per_kg: Optional[float],
        sand_g_per_kg: Optional[float],
        silt_g_per_kg: Optional[float],
    ) -> Optional[str]:
        if clay_g_per_kg is None or sand_g_per_kg is None or silt_g_per_kg is None:
            return None

        clay = clay_g_per_kg / 10.0
        sand = sand_g_per_kg / 10.0
        silt = silt_g_per_kg / 10.0

        if clay >= 40:
            return "Clay"
        if sand >= 70:
            return "Sandy"
        if silt >= 50:
            return "Silty"
        return "Loamy"

    def _compose_payload(self, weather: dict, soil: dict) -> dict:
        humidity = weather.get("humidity")
        rainfall = weather.get("rainfall")

        # Soil moisture proxy from rainfall + humidity when direct sensor data is not available.
        soil_moisture = None
        if humidity is not None or rainfall is not None:
            h = float(humidity or 0.0)
            r = float(rainfall or 0.0)
            soil_moisture = round(min(100.0, max(0.0, (0.65 * h) + (3.5 * r))), 2)

        organic_carbon = soil.get("soc_pct")

        electrical_conductivity = None
        if soil.get("cec") is not None:
            electrical_conductivity = round(max(0.0, (float(soil["cec"]) / 10.0) / 50.0), 2)

        nitrogen = soil.get("nitrogen")
        phosphorus = None
        potassium = None

        # Estimate NPK if direct nitrogen is unavailable.
        if nitrogen is None and organic_carbon is not None:
            texture = soil.get("texture") or "Loamy"
            texture_factor = {"Clay": 1.05, "Loamy": 1.0, "Silty": 0.95, "Sandy": 0.85}.get(texture, 1.0)
            nitrogen = round(max(0.0, float(organic_carbon) * 120.0 * texture_factor), 2)

        if organic_carbon is not None:
            texture = soil.get("texture") or "Loamy"
            p_factor = {"Clay": 15.0, "Loamy": 13.0, "Silty": 12.0, "Sandy": 10.0}.get(texture, 12.0)
            k_factor = {"Clay": 90.0, "Loamy": 75.0, "Silty": 68.0, "Sandy": 55.0}.get(texture, 70.0)
            phosphorus = round(max(0.0, float(organic_carbon) * p_factor), 2)
            potassium = round(max(0.0, float(organic_carbon) * k_factor), 2)

        return {
            "soil_pH": self._round_or_none(soil.get("ph")),
            "soil_moisture": self._round_or_none(soil_moisture),
            "organic_carbon": self._round_or_none(organic_carbon),
            "electrical_conductivity": self._round_or_none(electrical_conductivity),
            "nitrogen": self._round_or_none(nitrogen),
            "phosphorus": self._round_or_none(phosphorus),
            "potassium": self._round_or_none(potassium),
        }

    def _round_or_none(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), 2)


_fertilizer_auto_fill_service: Optional[FertilizerAutoFillService] = None


def get_fertilizer_auto_fill_service() -> FertilizerAutoFillService:
    global _fertilizer_auto_fill_service
    if _fertilizer_auto_fill_service is None:
        _fertilizer_auto_fill_service = FertilizerAutoFillService()
    return _fertilizer_auto_fill_service
