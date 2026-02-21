import json
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "LLM_constraint_extraction" / "v1.txt"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "llm_cache_milp"


class LLMConstraintExtractor:
    """
    LLM-based constraint extraction (cached by request_id).

    Pipeline:
      user_request -> LLM -> {hard_constraints, soft_preferences} -> cache(reqID.json)
    """

    def __init__(self, api_key: str = None, model: str = "models/gemini-2.0-flash"):
        if api_key is None:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model

        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt template not found at {PROMPT_PATH}")
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Cache helpers
    # -------------------------
    def _get_cache_file_path(self, request_id: str) -> Path:
        return CACHE_DIR / f"{request_id}.json"

    def _load_from_cache(self, request_id: str) -> Optional[Dict[str, Any]]:
        cache_file = self._get_cache_file_path(request_id)
        if not cache_file.exists():
            return None
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            # Ensure metadata exists
            cached_data.setdefault("metadata", {})
            cached_data["metadata"]["from_cache"] = True
            cached_data["metadata"]["request_id"] = request_id
            return cached_data
        except Exception:
            return None

    def _save_to_cache(self, request_id: str, data: Dict[str, Any]) -> None:
        cache_file = self._get_cache_file_path(request_id)
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except IOError as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")

    # -------------------------
    # Robust JSON parsing
    # -------------------------
    def _extract_first_json_object(self, text: str) -> str:
        """
        Best-effort extraction of the first JSON object from LLM output.
        Handles code fences + extra commentary.
        """
        t = (text or "").strip()

        # Remove common fences
        t = t.replace("```json", "").replace("```", "").strip()

        # Find first {...} span
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1].strip()

        return t

    # -------------------------
    # Schema normalization
    # -------------------------
    def _empty_constraints(self) -> Dict[str, Any]:
        """
        Baseline-compatible schema (hard + soft).
        """
        return {
            "hard_constraints": {
                "departure_date_window": None,
                "duration_range": None,
                "max_budget": None,
                "allowed_destinations": None,
                "required_ports": [],
                "num_guests": None,
                "exclude_sold_out": True,
            },
            "soft_preferences": {
                "preferred_cruise_line": None,
                "preferred_cabin_category": None,
                "preferred_ports": [],
                "preferred_duration_days": None,
                "price_sensitivity": None,
                "cruise_type": None,
            },
        }

    def _normalize_schema(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forces LLM outputs into the baseline schema (missing keys filled).
        Drops unknown keys (optional: you can keep them if you want).
        """
        base = self._empty_constraints()

        hard_in = (obj or {}).get("hard_constraints", {}) or {}
        soft_in = (obj or {}).get("soft_preferences", {}) or {}

        # --- hard ---
        for k in base["hard_constraints"]:
            if k in hard_in:
                base["hard_constraints"][k] = hard_in[k]

        # Always enforce exclude_sold_out True (even if model returns False)
        base["hard_constraints"]["exclude_sold_out"] = True

        # --- soft ---
        for k in base["soft_preferences"]:
            if k in soft_in:
                base["soft_preferences"][k] = soft_in[k]

        # Safety: ensure lists are lists
        if base["hard_constraints"]["required_ports"] is None:
            base["hard_constraints"]["required_ports"] = []
        if base["soft_preferences"]["preferred_ports"] is None:
            base["soft_preferences"]["preferred_ports"] = []

        return base

    # -------------------------
    # Prompt building
    # -------------------------
    def _build_extraction_prompt(self, user_request: str) -> str:
        return f"""{self.prompt_template}

USER REQUEST:
{user_request}
"""

    # -------------------------
    # Main API
    # -------------------------
    def extract_constraints(self, user_request: str, request_id: str) -> Dict[str, Any]:
        # 1) Cache
        cached = self._load_from_cache(request_id)
        if cached is not None:
            return cached

        # 2) LLM
        prompt = self._build_extraction_prompt(user_request)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
            ),
        )

        raw_text = getattr(response, "text", "") or ""
        json_text = self._extract_first_json_object(raw_text)

        # 3) Parse
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(json_text)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}

        normalized = self._normalize_schema(parsed)

        result = {
            "hard_constraints": normalized["hard_constraints"],
            "soft_preferences": normalized["soft_preferences"],
            "metadata": {
                "request_id": request_id,
                "from_cache": False,
                "model": self.model,
                "prompt_path": str(PROMPT_PATH),
                # keep raw response for debugging (truncate to keep cache readable)
                "raw_llm_response_truncated": raw_text[:4000],
            },
        }

        # 4) Cache
        self._save_to_cache(request_id, result)
        return result

    # -------------------------
    # Utilities
    # -------------------------
    def clear_cache(self) -> None:
        try:
            for cache_file in CACHE_DIR.glob("*.json"):
                cache_file.unlink()
        except IOError as e:
            print(f"Warning: Failed to clear cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        cache_files = list(CACHE_DIR.glob("*.json"))
        return {
            "cached_requests": len(cache_files),
            "cache_directory": str(CACHE_DIR),
            "cache_files": [f.name for f in cache_files],
        }