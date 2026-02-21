import json
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


from google import genai
from google.genai import types


class LLMPlanner:
    """
    LLM-only cruise planner with:
    - Candidate grounding
    - Prompt loaded from txt file
    - Persistent caching using request_id.json
    - Post-LLM validation (including cache re-validation)
    """

    MODEL_NAME = "models/gemini-2.0-flash"
    PROMPT_VERSION = "v1"

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

        project_root = Path(__file__).resolve().parents[2]

        self.prompt_path = (
            project_root / "prompts" / "LLM_only_planner" / f"{self.PROMPT_VERSION}.txt"
        )
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found at {self.prompt_path}")

        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

        self.cache_dir = project_root / "data" / "processed" / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def plan(self, cruises: List[Dict], user_request: str, request_id: str) -> Dict[str, Any]:
        cache_file = self.cache_dir / f"{request_id}.json"

        # Load from cache
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            llm_output = cached.get("llm_output", "") or ""
            # Re-validate cached output against current candidate set
            return self._process_llm_response(
                llm_output=llm_output,
                cruises=cruises,
                request_id=request_id,
                from_cache=True,
            )

        # Call LLM
        prompt = self._build_prompt(cruises, user_request)

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
            ),
        )
        llm_output = getattr(response, "text", "") or ""

        # Save raw LLM output to cache under reqID.json
        cache_data = {
            "request_id": request_id,
            "model": self.MODEL_NAME,
            "prompt_version": self.PROMPT_VERSION,
            "prompt_path": str(self.prompt_path),
            "user_request": user_request,
            "llm_output": llm_output,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

        return self._process_llm_response(
            llm_output=llm_output,
            cruises=cruises,
            request_id=request_id,
            from_cache=False,
        )

    # -------------------------
    # VALIDATION
    # -------------------------
    def _process_llm_response(
        self,
        llm_output: str,
        cruises: List[Dict],
        request_id: str,
        from_cache: bool,
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {
            "request_id": request_id,
            "from_cache": from_cache,
            "llm_output": llm_output,
            "is_valid": False,
            "selected_cruise_id": None,
            "justification": None,
            "validation_error": None,
        }

        valid_ids = {
            c.get("cruiseId") or c.get("cruise_id") or c.get("id")
            for c in cruises
        }
        valid_ids.discard(None)

        parsed = self._safe_parse_json(llm_output)
        if parsed is None:
            result["validation_error"] = "Malformed JSON output"
            return result

        selected = parsed.get("selectedCruiseId")
        justification = parsed.get("justification", "") or ""

        if not selected:
            result["validation_error"] = "Missing selectedCruiseId"
            return result

        # Allow NO_VALID_CRUISE
        if selected == "NO_VALID_CRUISE":
            result["selected_cruise_id"] = "NO_VALID_CRUISE"
            result["justification"] = justification
            result["validation_error"] = "LLM returned NO_VALID_CRUISE"
            return result

        # Must be grounded in candidate set
        if selected not in valid_ids:
            result["validation_error"] = f"Invalid cruise ID: {selected}"
            return result

        result["is_valid"] = True
        result["selected_cruise_id"] = selected
        result["justification"] = justification
        return result

    # -------------------------
    # PROMPT
    # -------------------------
    def _build_prompt(self, cruises: List[Dict], user_request: str) -> str:
        cruises_json = json.dumps(cruises, indent=2, ensure_ascii=False)

        # prompt_template should already contain the "return only JSON" rule.
        return (
            f"{self.prompt_template}\n\n"
            f"Available Cruises (Candidate Set):\n{cruises_json}\n\n"
            f"User Request:\n{user_request}\n"
        )

    # -------------------------
    # JSON PARSER (ROBUST)
    # -------------------------
    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        t = text.strip()

        # Remove markdown fences
        t = t.replace("```json", "").replace("```", "").strip()

        # Try direct parse
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Try extracting first JSON object
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = t[start : end + 1].strip()
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        return None