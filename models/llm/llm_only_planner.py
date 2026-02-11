import json
from google import genai
from pathlib import Path
from typing import List, Dict

CACHE_DIR = Path("data/processed/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class LLMPlanner:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def plan(self, cruises: List[Dict], user_request: str, request_id: str) -> Dict:
        """
        Plan a cruise selection using LLM with validation.

        Step 1: Candidate Grounding - All candidates passed in structured JSON
        Step 2: LLM Prompting - LLM selects exactly one cruise or responds NO_VALID_CRUISE
        Step 3: Post-LLM Validation - Verify selected cruise ID exists in candidate set

        Args:
            cruises: List of available cruises in structured format
            user_request: User's preferences and constraints (natural language)
            request_id: Unique identifier for this request

        Returns:
            Dictionary with selected cruise, validation status, and metrics
        """
        cache_file = CACHE_DIR / f"{request_id}.json"

        # Load from cache if exists
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            # Re-validate cached result in case validation logic changed
            return self._process_llm_response(cached.get("llm_output"), cruises, request_id)

        # Step 1 & 2: Build prompt with grounded candidates and get LLM response
        prompt = self._build_prompt(cruises, user_request)

        response = self.client.models.generate_content(
            model="models/gemini-3-flash-preview",
            contents=prompt
        )

        llm_output = response.text

        # Cache the raw LLM output
        cache_data = {
            "request_id": request_id,
            "llm_output": llm_output
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

        # Step 3: Validate and process LLM response
        return self._process_llm_response(llm_output, cruises, request_id)

    def _process_llm_response(self, llm_output: str, cruises: List[Dict], request_id: str) -> Dict:
        """
        Step 3: Post-LLM Validation.

        Validates:
        - LLM output is valid JSON
        - Selected cruise ID exists in candidate set
        - Handles NO_VALID_CRUISE response

        Failure cases:
        - LLM outputs NO_VALID_CRUISE
        - Selection of cruise ID not in candidate set
        - Malformed or non-JSON output
        """
        result = {
            "request_id": request_id,
            "llm_output": llm_output,
            "is_valid": False,
            "selected_cruise_id": None,
            "justification": None,
            "validation_error": None
        }

        # Create a set of valid cruise IDs for O(1) lookup
        valid_cruise_ids = {c.get("cruiseId") or c.get("cruise_id") or c.get("id") for c in cruises}

        # Try to parse LLM output as JSON
        try:
            parsed = json.loads(llm_output)
        except json.JSONDecodeError as e:
            result["validation_error"] = f"Malformed JSON output: {str(e)}"
            return result

        # Check for NO_VALID_CRUISE response
        if parsed.get("selectedCruiseId") == "NO_VALID_CRUISE":
            result["validation_error"] = "LLM determined no valid cruise matches constraints"
            return result

        # Extract selected cruise ID and justification
        selected_cruise_id = parsed.get("selectedCruiseId")
        justification = parsed.get("justification", "")

        if not selected_cruise_id:
            result["validation_error"] = "Missing selectedCruiseId in LLM response"
            return result

        # Verify cruise ID exists in candidate set
        if selected_cruise_id not in valid_cruise_ids:
            result["validation_error"] = f"Selected cruise ID '{selected_cruise_id}' not found in candidate set"
            return result

        # Validation passed
        result["is_valid"] = True
        result["selected_cruise_id"] = selected_cruise_id
        result["justification"] = justification
        result["validation_error"] = None

        return result

    def _build_prompt(self, cruises: List[Dict], user_request: str) -> str:
        """
        Step 1 & 2: Build prompt with candidate grounding and explicit instructions.
        """
        cruises_json = json.dumps(cruises, indent=2)

        return f"""You are a cruise itinerary planner.

You are given:
1) A list of available cruises (JSON) 
2) A user request with preferences and constraints (natural language)

Your task:
Select the SINGLE BEST cruise that satisfies the user's preferences and constraints.

CRITICAL RULES:
- You MUST select exactly one cruise from the provided list
- You MUST NOT invent or hallucinate cruises outside the candidate set
- You MUST only use cruise data provided in the JSON
- If NO cruise in the candidate set satisfies the user's constraints, respond with:
  {{"selectedCruiseId": "NO_VALID_CRUISE", "justification": "explanation of why no cruise matches"}}

Available Cruises (Candidate Set):
{cruises_json}

User Request:
{user_request}

Response Format (MUST be valid JSON):
{{
  "selectedCruiseId": "<cruise_id_from_list>",
  "justification": "<brief explanation of why this cruise best matches the request>"
}}

If no suitable cruise exists, use:
{{
  "selectedCruiseId": "NO_VALID_CRUISE",
  "justification": "<explanation>"
}}"""
