import re
from typing import Dict, Any, Optional, List, Tuple


DESTINATION_KEYWORDS = {
    "alaska": ["AK"],
    "caribbean": ["CA", "CE", "CS", "CW"],
    "bahamas": ["BH"],
    "bermuda": ["BM"],
    "mediterranean": ["MA", "ME", "MW"],
    "panama": ["PC"],
    "panama canal": ["PC"],
    "transatlantic": ["TC"],
    "mexico": ["MC"],
    "mexico & central america": ["MC"],
    "norway": ["NO"],
    "greek islands": ["GI"],
}

# Allow short codes and common variants
CRUISE_LINE_KEYWORDS = {
    "holland america": "HA",
    "holland": "HA",
    "princess": "PR",
    "royal caribbean": "RC",
    "royal": "RC",
    "norwegian": "NCL",
    "norwegian cruise line": "NCL",
    "ncl": "NCL",
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
    "summer": ("2026-06-01", "2026-08-31"),
    "fall": ("2026-09-01", "2026-11-30"),
    "autumn": ("2026-09-01", "2026-11-30"),
    "winter": ("2026-12-01", "2027-02-28"),
}


class ConstraintExtractor:
    """
    Rule-based constraint extraction (baseline).
    Deterministic, offline extraction using keywords + regex.
    """

    WORD_NUMBERS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30
    }

    # Precompiled regex for speed + consistency
    _RE_MONEY = re.compile(
        r"""
        (?:
            \$\s*(?P<d1>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?   # $3,500 or $3500
            |
            (?P<d2>\d+(?:\.\d+)?)\s*(?:usd|dollars?)      # 3500 dollars
            |
            (?P<d3>\d+(?:\.\d+)?)\s*(?:k)\b               # 5k
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    _RE_BUDGET_HINT = re.compile(r"\b(under|below|less than|<=|max(?:imum)?|budget)\b", re.IGNORECASE)

    _RE_RANGE_DAYS = re.compile(r"(\d+)\s*(?:to|–|-)\s*(\d+)\s*days?\b", re.IGNORECASE)
    _RE_RANGE_WEEKS = re.compile(r"(\d+)\s*(?:to|–|-)\s*(\d+)\s*weeks?\b", re.IGNORECASE)
    _RE_SINGLE_WEEK = re.compile(r"\b(?:a\s+)?(\d+)\s*weeks?\b", re.IGNORECASE)
    _RE_SINGLE_WEEK_WORD = re.compile(
        r"\b(?:a\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)\s*weeks?\b",
        re.IGNORECASE
    )
    _RE_AROUND_DAYS = re.compile(r"\b(?:around|about|approx(?:imately)?)\s+(\d+)\s*days?\b", re.IGNORECASE)
    _RE_SINGLE_DAYS = re.compile(r"\b(\d+)\s*(?:–|-)?\s*days?\b", re.IGNORECASE)

    _RE_GUESTS_NUM = re.compile(r"\bfor\s+(\d+)\b|\b(\d+)\s*(?:people|persons|guests|adults)\b", re.IGNORECASE)
    _RE_GUESTS_WORD = re.compile(
        r"\bfor\s+(one|two|three|four|five|six)\b|\b(one|two|three|four|five|six)\s*(?:people|persons|guests|adults)\b",
        re.IGNORECASE
    )

    def _convert_word_number(self, word: str) -> Optional[int]:
        return self.WORD_NUMBERS.get(word.lower())

    def _normalize_text(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # -------------------------
    # Destination
    # -------------------------
    def _extract_destination(self, text: str) -> Optional[List[str]]:
        t = self._normalize_text(text)
        found: List[str] = []
        # Longest keyword first to prefer "greek islands" over "islands" etc.
        for keyword in sorted(DESTINATION_KEYWORDS.keys(), key=len, reverse=True):
            if keyword in t:
                found.extend(DESTINATION_KEYWORDS[keyword])
        # De-duplicate while preserving order
        if not found:
            return None
        seen = set()
        out = []
        for x in found:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # -------------------------
    # Budget
    # -------------------------
    def _extract_budget(self, text: str) -> Optional[int]:
        t = self._normalize_text(text)

        # If user never said "budget-ish" words, still allow "$3500"
        m = self._RE_MONEY.search(t)
        if not m:
            return None

        raw = m.group("d1") or m.group("d2") or m.group("d3")
        if raw is None:
            return None

        # handle commas
        raw = raw.replace(",", "")
        val = float(raw)

        # if "5k" matched, multiply by 1000
        if m.group("d3") is not None:
            val *= 1000.0

        # Round down to int dollars
        return int(val)

    # -------------------------
    # Duration
    # -------------------------
    def _extract_duration(self, text: str) -> Optional[Dict[str, int]]:
        t = self._normalize_text(text)

        m = self._RE_RANGE_DAYS.search(t)
        if m:
            return {"min_days": int(m.group(1)), "max_days": int(m.group(2))}

        m = self._RE_RANGE_WEEKS.search(t)
        if m:
            return {"min_days": int(m.group(1)) * 7, "max_days": int(m.group(2)) * 7}

        m = self._RE_SINGLE_WEEK.search(t)
        if m:
            w = int(m.group(1))
            return {"min_days": w * 7, "max_days": w * 7}

        m = self._RE_SINGLE_WEEK_WORD.search(t)
        if m:
            w = self._convert_word_number(m.group(1))
            if w:
                return {"min_days": w * 7, "max_days": w * 7}

        m = self._RE_AROUND_DAYS.search(t)
        if m:
            d = int(m.group(1))
            return {"min_days": max(1, d - 2), "max_days": d + 2}

        # last resort: a single number of days (avoid matching years like 2026)
        # require context like "day(s)" already in regex
        m = self._RE_SINGLE_DAYS.search(t)
        if m:
            d = int(m.group(1))
            if 1 <= d <= 30:
                return {"min_days": d, "max_days": d}

        return None

    # -------------------------
    # Guests
    # -------------------------
    def _extract_guests(self, text: str) -> Optional[int]:
        t = self._normalize_text(text)

        if any(x in t for x in ["couple", "my wife and i", "my husband and i", "we are two"]):
            return 2

        m = self._RE_GUESTS_NUM.search(t)
        if m:
            g = m.group(1) or m.group(2)
            if g:
                try:
                    return int(g)
                except Exception:
                    pass

        m = self._RE_GUESTS_WORD.search(t)
        if m:
            w = m.group(1) or m.group(2)
            if w:
                return self._convert_word_number(w)

        return None

    # -------------------------
    # Departure date window
    # -------------------------
    def _extract_date_window(self, text: str, default_year: int = 2026) -> Optional[Dict[str, str]]:
        t = self._normalize_text(text)

        # Months: detect all mentioned months in calendar order
        found_months: List[Tuple[str, str, str]] = []
        for month, (start, end) in MONTH_TO_RANGE.items():
            if re.search(rf"\b{re.escape(month)}\b", t):
                found_months.append((month, start, end))

        if found_months:
            # Use first month start to last month end
            first_start = found_months[0][1]
            last_end = found_months[-1][2]
            return {"earliest": f"{default_year}-{first_start}", "latest": f"{default_year}-{last_end}"}

        # Seasons
        for season, (start, end) in SEASON_KEYWORDS.items():
            if re.search(rf"\b{re.escape(season)}\b", t):
                return {"earliest": start, "latest": end}

        return None

    # -------------------------
    # Ports and cruise line
    # -------------------------
    def _extract_preferred_ports(self, text: str) -> List[str]:
        t = self._normalize_text(text)
        out: List[str] = []
        for name, code in PORT_KEYWORDS.items():
            if re.search(rf"\b{re.escape(name)}\b", t):
                out.append(code)
        # de-dupe
        seen = set()
        final = []
        for p in out:
            if p not in seen:
                seen.add(p)
                final.append(p)
        return final

    def _extract_cruise_line(self, text: str) -> Optional[str]:
        t = self._normalize_text(text)
        for name in sorted(CRUISE_LINE_KEYWORDS.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", t):
                return CRUISE_LINE_KEYWORDS[name]
        return None

    def _extract_soft_flags(self, text: str) -> Dict[str, Optional[str]]:
        t = self._normalize_text(text)

        price_sensitivity = None
        cruise_type = None

        if any(w in t for w in ["luxury", "premium", "high-end", "upscale"]):
            price_sensitivity = "low"
            cruise_type = "luxury"

        if any(w in t for w in ["cheap", "budget", "affordable", "lowest price"]):
            price_sensitivity = "high"

        if "entertainment" in t:
            cruise_type = cruise_type or "entertainment"

        return {"price_sensitivity": price_sensitivity, "cruise_type": cruise_type}

    # -------------------------
    # Public API
    # -------------------------
    def extract_constraints(self, user_request: str, request_id: str) -> Dict[str, Any]:
        text = self._normalize_text(user_request)
        warnings: List[str] = []

        hard_constraints = {
            "departure_date_window": None,
            "duration_range": None,
            "max_budget": None,
            "num_guests": None,
            "allowed_destinations": None,
            "required_ports": [],
            "exclude_sold_out": True,
        }

        soft_preferences = {
            "preferred_cruise_line": None,
            "preferred_cabin_category": None,
            "preferred_ports": [],
            "preferred_duration_days": None,
            "price_sensitivity": None,
            "cruise_type": None,
        }

        # Destination
        dests = self._extract_destination(text)
        if dests:
            hard_constraints["allowed_destinations"] = dests
        else:
            warnings.append("No destination specified")

        # Duration
        dur = self._extract_duration(text)
        if dur is None:
            warnings.append("No duration specified")
            # Keep hard duration unset; set a reasonable soft default
            soft_preferences["preferred_duration_days"] = 7
        else:
            hard_constraints["duration_range"] = dur
            soft_preferences["preferred_duration_days"] = (dur["min_days"] + dur["max_days"]) // 2

        # Budget
        budget = self._extract_budget(text)
        hard_constraints["max_budget"] = budget
        if budget is None:
            warnings.append("No budget specified")

        # Guests
        guests = self._extract_guests(text)
        hard_constraints["num_guests"] = guests
        if guests is None:
            warnings.append("No guest count specified")

        # Date window
        date_window = self._extract_date_window(text, default_year=2026)
        hard_constraints["departure_date_window"] = date_window
        if date_window is None:
            warnings.append("No departure date specified")

        # Preferred ports
        soft_preferences["preferred_ports"] = self._extract_preferred_ports(text)

        # Cruise line
        soft_preferences["preferred_cruise_line"] = self._extract_cruise_line(text)

        # Soft flags (luxury/budget/etc.)
        flags = self._extract_soft_flags(text)
        soft_preferences["price_sensitivity"] = flags["price_sensitivity"]
        soft_preferences["cruise_type"] = flags["cruise_type"]

        confidence = "high" if len(warnings) == 0 else ("medium" if len(warnings) <= 2 else "low")

        return {
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
            "metadata": {
                "request_id": request_id,
                "confidence": confidence,
                "warnings": warnings,
            },
        }