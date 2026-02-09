import json
from google import genai
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "LLM_constraint_extraction"
    / "v1.txt"
)


class LLMConstraintExtractor:
    """
    LLM-based constraint extraction for the hybrid MILP planning pipeline.

    Unlike the rule-based ConstraintExtractor, this uses an LLM to intelligently
    extract structured constraints from natural language user requests.

    Hybrid Architecture:
    1. User Request (natural language) → LLM
    2. LLM extracts structured constraints
    3. MILP Solver optimizes based on constraints
    """

    def __init__(self, api_key: str = None):
        """
        Initialize LLM-based constraint extractor.

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

    def extract_constraints(self, user_request: str, request_id: str) -> dict:
        """
        Extract structured constraints from natural language using LLM.

        Args:
            user_request: Natural language user request
            request_id: Unique identifier for this request

        Returns:
            Dictionary with hard_constraints and soft_preferences
        """
        prompt = self._build_extraction_prompt(user_request)

        response = self.client.models.generate_content(
            model="models/gemini-3-flash-preview",
            contents=prompt
        )

        # Parse LLM response
        try:
            constraints_dict = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            constraints_dict = self._parse_fallback_response(response.text, user_request)

        return {
            "hard_constraints": constraints_dict.get("hard_constraints", {}),
            "soft_preferences": constraints_dict.get("soft_preferences", {}),
            "metadata": {
                "request_id": request_id,
                "llm_response": response.text
            }
        }

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
