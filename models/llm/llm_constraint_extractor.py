import json
import hashlib
from google import genai
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "LLM_constraint_extraction"
    / "v1.txt"
)
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "llm_cache_milp"


class LLMConstraintExtractor:
    """
    LLM-based constraint extraction for the hybrid MILP planning pipeline.

    Unlike the rule-based ConstraintExtractor, this uses an LLM to intelligently
    extract structured constraints from natural language user requests.

    Hybrid Architecture:
    1. User Request (natural language) → LLM (with persistent caching)
    2. LLM extracts structured constraints
    3. MILP Solver optimizes based on constraints
    """

    def __init__(self, api_key: str = None):
        """
        Initialize LLM-based constraint extractor with persistent caching.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY environment variable.
        """
        if api_key is None:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)

        # Load prompt template from v1.txt
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt template not found at {PROMPT_PATH}")
        self.prompt_template = PROMPT_PATH.read_text()

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cached constraint extraction."""
        return CACHE_DIR / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> dict or None:
        """Load cached constraints from disk."""
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    cached_data["metadata"]["from_cache"] = True
                    return cached_data
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        """Save constraints to disk cache."""
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")

    def _strip_json_fences(self, text: str) -> str:
        """
        Strips ```json fences and extracts the first JSON object if extra text exists.
        """
        t = (text or "").strip()

        # Remove fences if present
        if "```" in t:
            t = t.replace("```json", "").replace("```", "").strip()

        # Extract the first JSON object (robust against extra text)
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start:end + 1].strip()

        return t

    def extract_constraints(self, user_request: str, request_id: str) -> dict:
        """
        Extract structured constraints from natural language using LLM (with persistent caching).

        Args:
            user_request: Natural language user request
            request_id: Unique identifier for this request

        Returns:
            Dictionary with hard_constraints and soft_preferences
        """


        # Check disk cache first
        cached_result = self._load_from_cache(request_id)
        if cached_result:
            cached_result["metadata"]["request_id"] = request_id
            return cached_result

        # Call LLM if not cached
        prompt = self._build_extraction_prompt(user_request)

        response = self.client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt
        )

        raw_text = response.text
        clean_text = self._strip_json_fences(raw_text)

        # Parse LLM response
        try:
            constraints_dict = json.loads(clean_text)
        except json.JSONDecodeError:
            constraints_dict = self._parse_fallback_response(raw_text, user_request)

        result = {
            "hard_constraints": constraints_dict.get("hard_constraints", {}),
            "soft_preferences": constraints_dict.get("soft_preferences", {}),
            "metadata": {
                "request_id": request_id,
                "from_cache": False,
                "llm_response": response.text
            }
        }

        # Save to disk cache
        self._save_to_cache(request_id, result)

        return result

    def _build_extraction_prompt(self, user_request: str) -> str:
        """Build the prompt for constraint extraction using template from v1.txt."""
        return f"""{self.prompt_template}

USER REQUEST:
{user_request}

Please extract constraints and respond with ONLY valid JSON."""

    def _parse_fallback_response(self, response_text: str, user_request: str) -> dict:
        """
        Fallback parsing if LLM doesn't return valid JSON.
        Returns a basic constraints structure.
        """
        return {
            "hard_constraints": {
                "departure_date_window": None,
                "duration_range": None,
                "max_budget": None,
                "allowed_destinations": None,
                "required_ports": [],
                "num_guests": None,
                "exclude_sold_out": True
            },
            "soft_preferences": {
                "preferred_cruise_line": None,
                "cruise_type": None,
                "price_sensitivity": None
            }
        }

    def clear_cache(self) -> None:
        """Clear all cached files from disk."""
        try:
            for cache_file in CACHE_DIR.glob("*.json"):
                cache_file.unlink()
        except IOError as e:
            print(f"Warning: Failed to clear cache: {e}")

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        cache_files = list(CACHE_DIR.glob("*.json"))
        return {
            "cached_requests": len(cache_files),
            "cache_directory": str(CACHE_DIR),
            "cache_files": [f.name for f in cache_files]
        }
