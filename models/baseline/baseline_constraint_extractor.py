import re

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
    Rule-based constraint extraction for baseline planner.
    Uses regex patterns and keyword matching for deterministic, offline extraction.
    """

    # Mapping of written numbers to digits
    WORD_NUMBERS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30
    }

    def __init__(self):
        """Initialize the baseline constraint extractor."""
        pass

    def _convert_word_number(self, word: str) -> int:
        """Convert written number (e.g., 'two') to integer."""
        return self.WORD_NUMBERS.get(word.lower(), None)

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
        """Extract duration range from text (supports days, weeks, written numbers, fuzzy expressions)."""
        # Try to match "X-Y day/days" pattern with digits (e.g., "7-14 days")
        match = re.search(r"(\d+)[–-](\d+)\s*days?", text)
        if match:
            return {
                "min_days": int(match.group(1)),
                "max_days": int(match.group(2))
            }

        # Try to match "X-Y weeks" or "X to Y weeks" pattern with digits
        weeks_range_match = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*weeks?", text)
        if weeks_range_match:
            start_weeks = int(weeks_range_match.group(1))
            end_weeks = int(weeks_range_match.group(2))
            return {
                "min_days": start_weeks * 7,
                "max_days": end_weeks * 7
            }

        # Try to match single "X weeks" pattern with digits (e.g., "2 weeks")
        single_week_match = re.search(r"(?:a\s+)?(\d+)\s*weeks?(?:\s+cruise)?", text)
        if single_week_match:
            weeks = int(single_week_match.group(1))
            return {
                "min_days": weeks * 7,
                "max_days": weeks * 7
            }

        # Try to match written number weeks (e.g., "two weeks", "a two week cruise")
        written_week_match = re.search(
            r"(?:a\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)\s*weeks?(?:\s+cruise)?",
            text
        )
        if written_week_match:
            weeks = self._convert_word_number(written_week_match.group(1))
            if weeks:
                return {
                    "min_days": weeks * 7,
                    "max_days": weeks * 7
                }

        # Try to match "around/about X days" pattern (e.g., "around 8 days")
        around_days_match = re.search(r"(?:around|about|approx|approximately)\s+(\d+)\s*days?", text)
        if around_days_match:
            days = int(around_days_match.group(1))
            # For "around X days", use a range of ±2 days
            return {
                "min_days": max(1, days - 2),
                "max_days": days + 2
            }

        # Try to match "X-day" or "X day" pattern (e.g., "7-day cruise", "5 day", "7-day")
        # Supports both hyphenated (7-day) and non-hyphenated (7 day) formats
        hyphenated_or_spaced_day_match = re.search(r"(\d+)(?:[–-]\s*|\s+)days?(?:\s+cruise)?", text)
        if hyphenated_or_spaced_day_match:
            days = int(hyphenated_or_spaced_day_match.group(1))
            return {
                "min_days": days,
                "max_days": days
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
