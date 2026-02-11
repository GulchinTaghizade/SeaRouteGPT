from typing import List, Dict


class RuleBasedPlanner:

    def __init__(self):
        pass

    def plan(self, cruises: List[Dict], constraints: Dict) -> Dict:

        feasible_cruises = []

        # Filter cruises that satisfy hard constraints
        for cruise in cruises:
            if self._satisfies_constraints(cruise, constraints):
                feasible_cruises.append(cruise)

        if not feasible_cruises:
            return None

        # Filter to only cruises with valid prices for scoring
        cruises_with_prices = [c for c in feasible_cruises if c.get("roomPriceWithTaxesFees") is not None]

        if not cruises_with_prices:
            # If no prices available, just return the first feasible cruise
            best_cruise = feasible_cruises[0]
        else:
            preferred_duration = constraints["soft_preferences"].get("preferred_duration_days")

            # Prices for min-max normalization
            prices = [float(c["roomPriceWithTaxesFees"]) for c in cruises_with_prices]
            cmin, cmax = min(prices), max(prices)
            den = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

            # Calculate all duration deviations to find max for normalization
            duration_deviations = []
            for cruise in cruises_with_prices:
                if preferred_duration is not None:
                    dev = abs(cruise["duration"] - preferred_duration)
                    duration_deviations.append(dev)
                else:
                    duration_deviations.append(0)

            max_duration_deviation = max(duration_deviations) if duration_deviations else 1

            def score(cruise):
                # Step 2: Normalize price (min-max)
                price = cruise.get("roomPriceWithTaxesFees")
                if price is None:
                    normalized_price = 1.0 # Penalize unknown prices
                else:
                    normalized_price = (float(price) - cmin) / den

                # Step 2: Normalize duration deviation
                if preferred_duration is not None:
                    duration_deviation = abs(cruise["duration"] - preferred_duration)
                    normalized_duration_deviation = (
                        duration_deviation / max_duration_deviation if max_duration_deviation > 0 else 0
                    )
                else:
                    normalized_duration_deviation = 0

                # Step 2: Apply weights (α and β are fixed globally)
                alpha = 0.6
                beta = 0.4
                return alpha * normalized_price + beta * normalized_duration_deviation

            # Step 3: Select cruise with lowest score (greedy minimum cost approach)
            best_cruise = min(cruises_with_prices, key=score)

        return {
            "cruiseId": best_cruise["cruiseId"],
            "cruiseName": best_cruise["cruiseName"],
            "price": best_cruise.get("roomPriceWithTaxesFees"),
            "departureDate": best_cruise["departureDate"],
            "duration": best_cruise["duration"]
        }

    def _satisfies_constraints(self, cruise: Dict, constraints: Dict) -> bool:
        """Check if a cruise satisfies the given constraints.

        Step 1 of the Rule-Based Planner: Hard Filtering
        Applies strict, must-satisfy constraints to all candidate cruises.
        """
        hc = constraints["hard_constraints"]

        # Budget check - skip if no budget specified or if price is None
        if hc["max_budget"] is not None and cruise.get("roomPriceWithTaxesFees") is not None:
            if cruise["roomPriceWithTaxesFees"] > hc["max_budget"]:
                return False

        # Departure date window check
        if hc["departure_date_window"] is not None:
            dep_date = cruise.get("departureDate")
            if dep_date:
                earliest = hc["departure_date_window"].get("earliest")
                latest = hc["departure_date_window"].get("latest")
                if not (earliest <= dep_date <= latest):
                    return False

        # Duration check - skip if no duration specified
        if hc["duration_range"] is not None:
            if not (hc["duration_range"]["min_days"] <= cruise["duration"] <= hc["duration_range"]["max_days"]):
                return False

        # Number of guests check - skip if no num_guests specified
        if hc["num_guests"] is not None:
            cruise_max_guests = cruise.get("max_guests")
            if cruise_max_guests is not None and hc["num_guests"] > cruise_max_guests:
                return False

        # Required ports check - skip if no required ports specified
        if hc["required_ports"] is not None and len(hc["required_ports"]) > 0:
            cruise_ports = cruise.get("itineraryPorts", [])
            if not any(port in cruise_ports for port in hc["required_ports"]):
                return False

        # Destination check - skip if no destinations specified
        if hc["allowed_destinations"] is not None:
            if not any(dest in cruise["itineraryDestinations"] for dest in hc["allowed_destinations"]):
                return False

        # Sold out check
        if hc.get("exclude_sold_out", False) and cruise["soldOut"]:
            return False

        return True