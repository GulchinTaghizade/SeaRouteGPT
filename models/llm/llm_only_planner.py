import json
from google import genai
from pathlib import Path
from typing import List, Dict

CACHE_DIR = Path("data/processed/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class LLMPlanner:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def plan(self, cruises, user_request, request_id):
        cache_file = CACHE_DIR / f"{request_id}.json"

        # 🔁 Load from cache if exists
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        prompt = self._build_prompt(cruises, user_request)

        response = self.client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        result = {
            "request_id": request_id,
            "llm_output": response.text
        }

        cache_file.write_text(json.dumps(result, indent=2))
        return result

    def _build_prompt(self, cruises: List[Dict], user_request: str) -> str:
        cruises_json = json.dumps(cruises, indent=2)

        return f"""
You are a cruise itinerary planner.

You are given:
1) A list of available cruises (JSON)
2) A user request (natural language)

Rules:
- You MUST select one cruise from the list
- You MUST NOT invent cruises
- If no cruise matches, return null
- Output JSON only

Cruises:
{cruises_json}

User request:
{user_request}

Output format:
{{
  "selected_cruise_id": "...",
  "reason": "..."
}}
"""