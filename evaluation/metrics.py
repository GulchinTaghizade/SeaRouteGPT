
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Itinerary:
    cruise_id: str
    ports: List[str]
    departure_date: str
    duration_days: int
    total_price: float
    cruise_line: Optional[str] = None
    cabin_category: Optional[str] = None
    sold_out: bool = False
    destinations: List[str] = field(default_factory=list)

    constraint_violations: List[str] = field(default_factory=list)
    preference_matches: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    request_id: str
    method_name: str
    run_number: int
    seed: int
    feasibility: float
    personalization: float
    optimization_utility: float
    itinerary: Optional[Itinerary] = None
    error_msg: str = ""


class CruiseMetrics:
    """
    Thesis metrics:
      - Feasibility: strict hard-constraint satisfaction (binary)
      - Personalization: weighted soft-preference satisfaction (0..1)
      - Utility: alpha(1-C_hat)+beta(1-T_hat) among feasible candidates (0..1)
    Unified failure handling: if no itinerary or infeasible -> all metrics = 0
    """

    def __init__(self, cruise_catalog: List[Dict[str, Any]]):
        self.cruise_catalog = cruise_catalog
        # Your catalog uses "cruiseId" (not cruise_id)
        self.catalog_index = {c.get("cruiseId"): c for c in cruise_catalog}

    # ---------- conversion ----------
    def to_itinerary(self, cruise: Optional[Dict[str, Any]]) -> Optional[Itinerary]:
        if cruise is None:
            return None

        cruise_id = cruise.get("cruiseId")
        if cruise_id is None:
            return None

        price = cruise.get("roomPriceWithTaxesFees")
        try:
            total_price = float(price) if price is not None else float("nan")
        except (ValueError, TypeError):
            total_price = float("nan")

        ports = cruise.get("itineraryPorts", []) or []
        dests = cruise.get("itineraryDestinations", []) or []

        return Itinerary(
            cruise_id=cruise_id,
            ports=list(ports),
            destinations=list(dests),
            departure_date=str(cruise.get("departureDate") or ""),
            duration_days=int(cruise.get("duration") or 0),
            total_price=total_price,
            cruise_line=cruise.get("cruiseLineCode"),
            cabin_category=cruise.get("roomTypeCategoryCode"),
            sold_out=bool(cruise.get("soldOut", False)),
        )

    # ---------- feasibility ----------
    def compute_feasibility(
        self,
        hard_constraints: Dict[str, Any],
        itinerary: Optional[Itinerary],
    ) -> Tuple[float, List[str]]:
        violations: List[str] = []

        if itinerary is None:
            return 0.0, ["no_itinerary_returned"]

        # 1) Existence
        if itinerary.cruise_id not in self.catalog_index:
            return 0.0, ["hallucinated_cruise_id"]

        cruise = self.catalog_index[itinerary.cruise_id]

        # 2) Availability
        if bool(cruise.get("soldOut", False)) or itinerary.sold_out:
            return 0.0, ["cruise_sold_out"]

        # 3) Temporal validity (window + duration range)
        dw = hard_constraints.get("departure_date_window")
        if dw:
            earliest = dw.get("earliest")
            latest = dw.get("latest")
            dep = itinerary.departure_date
            if not dep:
                violations.append("missing_departure_date")
            else:
                # ISO YYYY-MM-DD compares lexicographically correctly
                if earliest and dep < earliest:
                    violations.append("departure_before_earliest")
                if latest and dep > latest:
                    violations.append("departure_after_latest")

        dr = hard_constraints.get("duration_range")
        if dr:
            mn = int(dr.get("min_days"))
            mx = int(dr.get("max_days"))
            d = int(itinerary.duration_days)
            if not (mn <= d <= mx):
                violations.append("duration_out_of_range")

        # 4) Destination validity: set intersection with allowed_destinations
        allowed = hard_constraints.get("allowed_destinations")
        if allowed:
            cruise_dests = set(itinerary.destinations or [])
            if cruise_dests.isdisjoint(set(allowed)):
                violations.append("destination_mismatch")

        # required_ports: interpret as "must include ALL listed ports"
        required_ports = hard_constraints.get("required_ports") or []
        if required_ports:
            ports = set(itinerary.ports or [])
            if not set(required_ports).issubset(ports):
                violations.append("missing_required_ports")

        itinerary.constraint_violations = violations
        return (0.0, violations) if violations else (1.0, [])

    # ---------- personalization ----------
    def compute_personalization(
        self,
        feasibility: float,
        soft_preferences: Dict[str, Any],
        itinerary: Optional[Itinerary],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        if feasibility < 0.5 or itinerary is None:
            return 0.0

        # Use ONLY preferences you actually extract (keep it aligned with v1.txt)
        prefs = {
            "preferred_cruise_line": soft_preferences.get("preferred_cruise_line"),
            "cruise_type": soft_preferences.get("cruise_type"),
            "price_sensitivity": soft_preferences.get("price_sensitivity"),
            "preferred_duration_days": soft_preferences.get("preferred_duration_days"),
        }

        # Indicators
        indicators: Dict[str, bool] = {}

        # cruise line
        if prefs["preferred_cruise_line"] is not None:
            indicators["preferred_cruise_line"] = (itinerary.cruise_line == prefs["preferred_cruise_line"])
        else:
            indicators["preferred_cruise_line"] = False

        # duration closeness (soft)
        if prefs["preferred_duration_days"] is not None:
            target = int(prefs["preferred_duration_days"])
            indicators["preferred_duration_days"] = (abs(itinerary.duration_days - target) <= 1)
        else:
            indicators["preferred_duration_days"] = False

        # cruise_type / price_sensitivity are not reliably derivable from catalog unless you map them.
        # Keep them as unmet unless you implement mappings later.
        indicators["cruise_type"] = False if prefs["cruise_type"] is not None else False
        indicators["price_sensitivity"] = False if prefs["price_sensitivity"] is not None else False

        itinerary.preference_matches = indicators

        if weights is None:
            # uniform weights over the preference keys we measure
            k = len(indicators)
            weights = {p: 1.0 / k for p in indicators}

        num = sum(weights.get(p, 0.0) * (1.0 if indicators[p] else 0.0) for p in indicators)
        den = sum(weights.values()) if weights else 1.0
        return float(num / den) if den > 0 else 0.0

    # ---------- utility ----------
    def compute_optimization_utility(
        self,
        feasibility: float,
        hard_constraints: Dict[str, Any],
        soft_preferences: Dict[str, Any],
        itinerary: Optional[Itinerary],
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        feasible_candidates: Optional[List[Itinerary]] = None,
    ) -> float:
        if feasibility < 0.5 or itinerary is None:
            return 0.0

        # candidates: ideally all cruises that satisfy hard constraints (not just the chosen one)
        candidates = feasible_candidates or []
        if not candidates:
            return 0.0

        prices = [c.total_price for c in candidates if c.total_price == c.total_price]  # not NaN
        if not prices:
            return 0.0

        cmin, cmax = min(prices), max(prices)
        cden = (cmax - cmin) if (cmax - cmin) != 0 else 1.0
        c_hat = (itinerary.total_price - cmin) / cden

        preferred_duration = soft_preferences.get("preferred_duration_days")
        if preferred_duration is None:
            # if missing, treat deviation term as 0 (perfect)
            t_hat = 0.0
        else:
            d_star = float(preferred_duration)
            deviations = [abs(c.duration_days - d_star) for c in candidates]
            tden = max(deviations) if max(deviations) != 0 else 1.0
            t_hat = abs(itinerary.duration_days - d_star) / tden

        utility = alpha * (1.0 - c_hat) + beta * (1.0 - t_hat)
        return float(max(0.0, min(1.0, utility)))

    # ---------- candidate filtering for utility ----------
    def feasible_candidate_set(
        self,
        hard_constraints: Dict[str, Any],
    ) -> List[Itinerary]:
        out: List[Itinerary] = []
        for c in self.cruise_catalog:
            it = self.to_itinerary(c)
            if it is None:
                continue
            feas, _ = self.compute_feasibility(hard_constraints, it)
            if feas >= 0.5:
                out.append(it)
        return out