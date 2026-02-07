import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "constraint_extraction"
    / "v1.txt"
)

DESTINATION_KEYWORDS = {
    "alaska": ["AK"],
    "caribbean": ["CA", "CS", "CW"],
    "bahamas": ["BH"],
    "bermuda": ["BM"],
    "mediterranean": ["MA"],
    "panama": ["PC"],
    "transatlantic": ["TC"],
    "mexico": ["MC"],
    "norway": ["NO"],
    "greek islands": ["GI"],
}

CRUISE_LINE_KEYWORDS = {
    "holland america": "HA",
    "princess": "PR",
    "royal caribbean": "RC",
    "norwegian": "NCL"
}

PORT_KEYWORDS = {
    "vancouver": "CAVAN",
    "seattle": "USSEA",
    "juneau": "USJNU",
    "anchorage": "USANC",
}



MONTH_TO_RANGE = {
    "january": ("01-01", "01-31"),
    "february": ("02-01", "02-28"),
    "march": ("03-01", "03-31"),
    "april": ("04-01", "04-30"),
    "may": ("05-01", "05-31"),
    "june": ("06-01", "06-30"),
    "july": ("07-01", "07-31"),
    "august": ("08-01", "08-31"),
    "september": ("09-01", "09-30"),
    "october": ("10-01", "10-31"),
    "november": ("11-01", "11-30"),
    "december": ("12-01", "12-31"),
}

SEASON_KEYWORDS = {
    "spring": ("2026-03-01", "2026-05-31"),
    "fall": ("2026-09-01", "2026-11-30"),
    "summer": ("2026-06-01", "2026-08-31"),
    "winter": ("2026-12-01", "2027-02-28")
}


class ConstraintExtractor:
    """
    LLM-based constraint extraction interface.
    Sprint 1: uses a mock LLM response for deterministic testing.
    """


    def __init__(self):
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"Constraint extraction prompt not found at {PROMPT_PATH}"
            )
        self.prompt_template = PROMPT_PATH.read_text()

    def _extract_destination(self, text: str):
        destinations = []
        for keyword, codes in DESTINATION_KEYWORDS.items():
            if keyword in text.lower():
                destinations.extend(codes)
        return destinations if destinations else None

    def _extract_budget(self, text: str):
        match = re.search(r"\$(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_duration(self, text: str):
        """Extract duration range from text (supports both days and weeks)."""
        # Try to match "X-Y day/days" pattern first
        match = re.search(r"(\d+)[–-](\d+)\s*days?", text)
        if match:
            return {
                "min_days": int(match.group(1)),
                "max_days": int(match.group(2))
            }

        # Try to match "X-Y weeks" or "X to Y weeks" pattern
        weeks_range_match = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*weeks?", text)
        if weeks_range_match:
            start_weeks = int(weeks_range_match.group(1))
            end_weeks = int(weeks_range_match.group(2))
            return {
                "min_days": start_weeks * 7,
                "max_days": end_weeks * 7
            }

        # Try to match single "X weeks" pattern (e.g., "2 weeks", "a 2 week cruise")
        single_week_match = re.search(r"(?:a\s+)?(\d+)\s*weeks?(?:\s+cruise)?", text)
        if single_week_match:
            weeks = int(single_week_match.group(1))
            return {
                "min_days": weeks * 7,
                "max_days": weeks * 7
            }

        return None


    def _extract_guests(self, text: str):
        if "two people" in text or "for two" in text:
            return 2
        return None

    def extract_constraints(self, user_request: str, request_id: str):
        """Extract hard and soft constraints from user request."""
        text = user_request.lower()
        warnings_list = []
        hard_constraints = {
            "departure_date_window": None,
            "duration_range": None,
            "max_budget": None,
            "num_guests": None,
            "allowed_destinations": None,
            "required_ports": [],
            "exclude_sold_out": True
        }

        soft_preferences = {
            "preferred_cruise_line": None,
            "preferred_cabin_category": None,
            "preferred_ports": [],
            "preferred_duration_days": None,
            "price_sensitivity": None,
            "cruise_type": None
        }

        # Extract destination
        destination = self._extract_destination(text)
        if destination:
            hard_constraints["allowed_destinations"] = destination
        else:
            warnings_list.append("No destination specified")

        # Extract duration
        duration = self._extract_duration(text)
        if duration is None:
            warnings_list.append("No duration specified")
            soft_preferences["preferred_duration_days"] = 7  # Default 7-day cruise
        else:
            hard_constraints["duration_range"] = duration
            soft_preferences["preferred_duration_days"] = (
                duration["min_days"] + duration["max_days"]
            ) // 2

        # Extract budget
        budget = self._extract_budget(text)
        hard_constraints["max_budget"] = budget
        if budget is None:
            warnings_list.append("No budget specified")

        # Extract guests
        guests = self._extract_guests(text)
        hard_constraints["num_guests"] = guests

        # Extract departure date window
        date_window = None
        found_months = []
        for month, (start, end) in MONTH_TO_RANGE.items():
            if month in text.lower():
                found_months.append((month, start, end))

        if found_months:
            if len(found_months) > 1:
                first_start = found_months[0][1]
                last_end = found_months[-1][2]
                date_window = {
                    "earliest": f"2026-{first_start}",
                    "latest": f"2026-{last_end}"
                }
            else:
                start, end = found_months[0][1], found_months[0][2]
                date_window = {
                    "earliest": f"2026-{start}",
                    "latest": f"2026-{end}"
                }
        else:
            for season, (start, end) in SEASON_KEYWORDS.items():
                if season in text.lower():
                    date_window = {
                        "earliest": start,
                        "latest": end
                    }
                    break

        hard_constraints["departure_date_window"] = date_window
        if date_window is None:
            warnings_list.append("No departure date specified")

        # Extract preferred ports
        preferred_ports = []
        for name, code in PORT_KEYWORDS.items():
            if name in text.lower():
                preferred_ports.append(code)
        soft_preferences["preferred_ports"] = preferred_ports

        # Extract cruise line preference
        for name, code in CRUISE_LINE_KEYWORDS.items():
            if name in text:
                soft_preferences["preferred_cruise_line"] = code
                break

        # Extract cruise type
        if any(word in text.lower() for word in ["luxury", "premium", "high-end"]):
            soft_preferences["price_sensitivity"] = "low"
            soft_preferences["cruise_type"] = "luxury"

        if any(word in text.lower() for word in ["cheap", "budget", "affordable"]):
            soft_preferences["price_sensitivity"] = "high"

        if "entertainment" in text.lower():
            soft_preferences["cruise_type"] = "entertainment"

        confidence = "high" if not warnings_list else "low"

        return {
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
            "metadata": {
                "request_id": request_id,
                "confidence": confidence,
                "warnings": warnings_list
            }
        }
