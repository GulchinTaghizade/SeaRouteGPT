from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Data containers
# =========================
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


# =========================
# Metrics engine
# =========================
class CruiseMetrics:
    """
    Thesis metrics:
      - Feasibility: strict hard-constraint satisfaction (binary)
      - Personalization: weighted soft-preference satisfaction (0..1)
      - Utility: alpha(1-C_hat)+beta(1-T_hat) among feasible candidates (0..1)

    Unified failure handling:
      if no itinerary or infeasible -> all metrics = 0
    """

    def __init__(self, cruise_catalog: List[Dict[str, Any]]):
        self.cruise_catalog = cruise_catalog

        # Catalog uses cruiseId; support cruise_id too just in case
        self.catalog_index: Dict[str, Dict[str, Any]] = {}
        for c in cruise_catalog:
            cid = c.get("cruiseId") or c.get("cruise_id")
            if cid:
                self.catalog_index[str(cid)] = c

    # -------------------------
    # Convert catalog item -> Itinerary
    # -------------------------
    def to_itinerary(self, cruise: Optional[Dict[str, Any]]) -> Optional[Itinerary]:
        if not cruise:
            return None

        cid = cruise.get("cruiseId") or cruise.get("cruise_id")
        if not cid:
            return None

        # price
        price = cruise.get("roomPriceWithTaxesFees")
        try:
            total_price = float(price)
        except (ValueError, TypeError):
            # if price can't be parsed, skip (utility can't be computed reliably)
            total_price = float("nan")

        return Itinerary(
            cruise_id=str(cid),
            ports=list(cruise.get("itineraryPorts", []) or []),
            destinations=list(cruise.get("itineraryDestinations", []) or []),
            departure_date=str(cruise.get("departureDate") or ""),
            duration_days=int(cruise.get("duration") or 0),
            total_price=total_price,
            cruise_line=cruise.get("cruiseLineCode"),
            cabin_category=cruise.get("roomTypeCategoryCode"),
            sold_out=bool(cruise.get("soldOut", False)),
        )

    # -------------------------
    # 1) FEASIBILITY (binary)
    # -------------------------
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
        if bool(cruise.get("soldOut", False)) or bool(itinerary.sold_out):
            return 0.0, ["cruise_sold_out"]

        # 3) Temporal validity
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
            try:
                mn = int(dr.get("min_days"))
                mx = int(dr.get("max_days"))
            except (TypeError, ValueError):
                mn, mx = None, None

            if mn is not None and mx is not None:
                d = int(itinerary.duration_days)
                if not (mn <= d <= mx):
                    violations.append("duration_out_of_range")

        # 4) Destination validity: INTERSECTION with allowed destinations
        allowed = hard_constraints.get("allowed_destinations")
        if allowed:
            cruise_dests = set(itinerary.destinations or [])
            if cruise_dests.isdisjoint(set(allowed)):
                violations.append("destination_mismatch")

        # 5) Budget validity (if present)
        max_budget = hard_constraints.get("max_budget")
        if max_budget is not None:
            try:
                if itinerary.total_price != itinerary.total_price:  # NaN check
                    violations.append("missing_price_for_budget")
                elif itinerary.total_price > float(max_budget):
                    violations.append("budget_exceeded")
            except Exception:
                violations.append("invalid_budget_or_price")


        itinerary.constraint_violations = violations
        return (0.0, violations) if violations else (1.0, [])

    # -------------------------
    # 2) PERSONALIZATION (0..1)
    # -------------------------
    def compute_personalization(
            self,
            feasibility: float,
            soft_preferences: Dict[str, Any],
            itinerary: Optional[Itinerary],
            weights: Optional[Dict[str, float]] = None,
    ) -> float:
        if feasibility < 0.5 or itinerary is None:
            return 0.0

        indicators: Dict[str, bool] = {}

        # --- cruise line ---
        pref_line = soft_preferences.get("preferred_cruise_line")
        if pref_line is not None:
            indicators["preferred_cruise_line"] = (itinerary.cruise_line == pref_line)

        # --- cabin category ---
        pref_cabin = soft_preferences.get("preferred_cabin_category")
        if pref_cabin is not None:
            indicators["preferred_cabin_category"] = (itinerary.cabin_category == pref_cabin)

        # --- preferred ports (intersection) ---
        pref_ports = soft_preferences.get("preferred_ports")
        if pref_ports:
            ports = set(itinerary.ports or [])
            indicators["preferred_ports"] = not ports.isdisjoint(set(pref_ports))

        # --- duration closeness (±1 day) ---
        pref_duration = soft_preferences.get("preferred_duration_days")
        if pref_duration is not None:
            try:
                target = int(pref_duration)
                indicators["preferred_duration_days"] = (abs(itinerary.duration_days - target) <= 1)
            except Exception:
                indicators["preferred_duration_days"] = False

        # Save matches for debugging/analysis
        itinerary.preference_matches = indicators

        # If no measurable preferences were provided, personalization = 0 by definition
        if not indicators:
            return 0.0

        # Weights: uniform over *only the included preferences* unless provided
        if weights is None:
            k = len(indicators)
            weights = {k_: 1.0 / k for k_ in indicators}

        num = sum(weights.get(k_, 0.0) * (1.0 if indicators[k_] else 0.0) for k_ in indicators)
        den = sum(weights.get(k_, 0.0) for k_ in indicators)  # only included keys
        return float(num / den) if den > 0 else 0.0

    # -------------------------
    # 3) UTILITY (0..1)
    # -------------------------
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

        # thesis: alpha + beta = 1
        s = alpha + beta
        if s <= 0:
            return 0.0
        alpha, beta = alpha / s, beta / s

        candidates = feasible_candidates or []
        if not candidates:
            return 0.0

        # prices for normalization (ignore NaN)
        prices = [c.total_price for c in candidates if c.total_price == c.total_price]
        if not prices or itinerary.total_price != itinerary.total_price:
            return 0.0

        cmin, cmax = min(prices), max(prices)
        cden = (cmax - cmin) if (cmax - cmin) != 0 else 1.0
        c_hat = (itinerary.total_price - cmin) / cden
        c_hat = max(0.0, min(1.0, c_hat))

        # duration deviation normalization
        pref_duration = soft_preferences.get("preferred_duration_days")
        if pref_duration is None:
            t_hat = 0.0
        else:
            try:
                d_star = float(pref_duration)
            except Exception:
                d_star = float(itinerary.duration_days)

            deviations = [abs(c.duration_days - d_star) for c in candidates]
            tden = max(deviations) if max(deviations) != 0 else 1.0
            t_hat = abs(itinerary.duration_days - d_star) / tden
            t_hat = max(0.0, min(1.0, t_hat))

        utility = alpha * (1.0 - c_hat) + beta * (1.0 - t_hat)
        return float(max(0.0, min(1.0, utility)))

    # -------------------------
    # Candidate set for utility normalization
    # -------------------------
    def feasible_candidate_set(self, hard_constraints: Dict[str, Any]) -> List[Itinerary]:
        out: List[Itinerary] = []
        for c in self.cruise_catalog:
            it = self.to_itinerary(c)
            if it is None:
                continue
            feas, _ = self.compute_feasibility(hard_constraints, it)
            if feas >= 0.5:
                out.append(it)
        return out

    def utility_candidate_set(self, hard_constraints: Dict[str, Any]) -> List[Itinerary]:
        feas = self.feasible_candidate_set(hard_constraints)
        return [it for it in feas if it.total_price == it.total_price]  # drop NaN