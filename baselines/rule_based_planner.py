from typing import List, Dict


class RuleBasedPlanner:
    """
    Simple rule-based cruise planner that selects the cheapest feasible cruise.
    """

    def __init__(self):
        pass

    def plan(self, cruises: List[Dict], constraints: Dict) -> Dict:

        feasible_cruises = []

        # Filter cruises that satisfy hard constraints
        for cruise in cruises:
            if self._satisfies_constraints(cruise, constraints):
                feasible_cruises.append(cruise)

        preferred_duration = constraints["soft_preferences"].get("preferred_duration_days")

        max_price = max(c["roomPriceWithTaxesFees"] for c in feasible_cruises)

        def score(cruise):
            normalized_price = cruise["roomPriceWithTaxesFees"] / max_price

            if preferred_duration is not None:
                duration_deviation = abs(cruise["duration"] - preferred_duration) / preferred_duration
            else:
                duration_deviation = 0

            alpha = 0.6
            beta = 0.4

            return alpha * normalized_price + beta * duration_deviation
        
        if not feasible_cruises:
         return None

        best_cruise = min(feasible_cruises, key=score)

        return {
            "cruiseId": best_cruise["cruiseId"],
            "cruiseName": best_cruise["cruiseName"],
            "price": best_cruise["roomPriceWithTaxesFees"],
            "departureDate": best_cruise["departureDate"],
            "duration": best_cruise["duration"]
        }

    def _satisfies_constraints(self, cruise: Dict, constraints: Dict) -> bool:
        """Check if a cruise satisfies the given constraints."""
        hc = constraints["hard_constraints"]

        # Budget check - skip if no budget specified
        if hc["max_budget"] is not None and cruise["roomPriceWithTaxesFees"] > hc["max_budget"]:
            return False

        # Duration check - skip if no duration specified
        if hc["duration_range"] is not None:
            if not (hc["duration_range"]["min_days"] <= cruise["duration"] <= hc["duration_range"]["max_days"]):
                return False

        # Destination check - skip if no destinations specified
        if hc["allowed_destinations"] is not None:
            if not any(dest in cruise["itineraryDestinations"] for dest in hc["allowed_destinations"]):
                return False

        # Sold out check
        if hc.get("exclude_sold_out", False) and cruise["soldOut"]:
            return False

        return True