from typing import List, Dict


class ConstraintValidator:
    def __init__(self, cruises: List[Dict]):
        self.cruises = cruises

    def validate(self, constraints: Dict) -> Dict:
        feasible = []

        for cruise in self.cruises:
            if self._satisfies_hard_constraints(cruise, constraints):
                feasible.append(cruise)

        return {
            "feasible_count": len(feasible),
            "is_feasible": len(feasible) > 0,
            "feasible_cruises": feasible 
        }

    def _satisfies_hard_constraints(self, cruise: Dict, constraints: Dict) -> bool:
        hc = constraints["hard_constraints"]

        # Budget check - skip if no budget specified or if price is None
        if hc["max_budget"] is not None and cruise.get("roomPriceWithTaxesFees") is not None:
            if cruise["roomPriceWithTaxesFees"] > hc["max_budget"]:
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
        if hc.get("exclude_sold_out") and cruise["soldOut"]:
            return False
        
        # Departure date window check
        if hc["departure_date_window"] is not None:
            dep = cruise["departureDate"]
            if not (
                hc["departure_date_window"]["earliest"]
                <= dep
                <= hc["departure_date_window"]["latest"]
            ):
                return False
        # Required ports check
        if hc["required_ports"]:
            if not all(port in cruise["itineraryPorts"] for port in hc["required_ports"]):
                return False
        
        return True