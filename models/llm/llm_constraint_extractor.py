import json
from google import genai


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
            model="models/gemini-2.5-flash",
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
        """Build the prompt for constraint extraction."""
        return f"""You are a cruise constraint extraction specialist.

Extract structured constraints from the user's cruise request.

IMPORTANT CONSTRAINTS TO EXTRACT:
1. departure_date_window: Extract the month or date range the user wants to travel
   - Format: {{"earliest": "YYYY-MM-DD", "latest": "YYYY-MM-DD"}}
   - If month mentioned, use the full month range (e.g., August = 2026-08-01 to 2026-08-31)
   - If season mentioned, map appropriately
   
2. duration_range: Extract the desired cruise duration
   - Format: {{"min_days": number, "max_days": number}}
   - Support all duration formats:
     * Written numbers: "two weeks" = 14 days, "three days" = 3 days
     * Hyphenated: "7-day cruise", "14-day"
     * Spaced: "7 day cruise", "14 day"
     * Ranges: "7-14 days", "1-2 weeks"
     * Fuzzy: "around 8 days" = 6-10 days, "about 10 days" = 8-12 days
   - If no duration mentioned, set to null
   
3. max_budget: Extract the maximum budget if mentioned
   - Format: number (dollars)
   - Set to null if no budget mentioned
   
4. allowed_destinations: Extract desired cruise regions
   - Valid codes: AK (Alaska), CA/CS/CW (Caribbean), BH (Bahamas), BM (Bermuda),
     MA (Mediterranean), PC (Panama Canal), TC (Transatlantic), MC (Mexico), 
     NO (Norway), GI (Greek Islands)
   - Format: ["CODE1", "CODE2", ...]
   - Set to null if no destination mentioned
   
5. required_ports: List of specific ports required
   - Format: ["PORT_CODE", ...]
   - Set to [] if none mentioned

6. num_guests: Number of people traveling
   - Format: number
   - Set to null if not mentioned

7. exclude_sold_out: Always set to true for now

SOFT PREFERENCES (optional):
- preferred_cruise_line: Brand preference (e.g., "Royal Caribbean")
- cruise_type: Type preference (e.g., "luxury", "budget", "entertainment")
- price_sensitivity: "low" (luxury), "high" (budget), or null

USER REQUEST:
"{user_request}"

RESPONSE FORMAT (MUST BE VALID JSON):
{{
  "hard_constraints": {{
    "departure_date_window": {{"earliest": "YYYY-MM-DD", "latest": "YYYY-MM-DD"}} or null,
    "duration_range": {{"min_days": number, "max_days": number}} or null,
    "max_budget": number or null,
    "allowed_destinations": ["CODE1", "CODE2"] or null,
    "required_ports": [],
    "num_guests": number or null,
    "exclude_sold_out": true
  }},
  "soft_preferences": {{
    "preferred_cruise_line": "name" or null,
    "cruise_type": "type" or null,
    "price_sensitivity": "low"|"high"|null
  }}
}}"""

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
