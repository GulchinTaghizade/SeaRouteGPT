from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any
from ortools.linear_solver import pywraplp


class MILPSolver:
    """
    MILP solver for selecting ONE optimal cruise given hard constraints.
    Robust to both RapidAPI schema and normalized UI schema.
    """

    # ---------- small schema helpers ----------
    @staticmethod
    def _get_price(c: Dict[str, Any]) -> Optional[float]:
        """Return numeric total price if present, else None."""
        price = c.get("roomPriceWithTaxesFees", None)
        if price is None:
            price = c.get("price", None)

        if price is None:
            return None

        try:
            v = float(price)
            # guard against NaN
            if v != v:
                return None
            return v
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_dests(c: Dict[str, Any]) -> List[str]:
        d = c.get("itineraryDestinations", None)
        if d is None:
            d = c.get("destinations", None)
        return [str(x) for x in d] if isinstance(d, list) else []

    @staticmethod
    def _get_ports(c: Dict[str, Any]) -> List[str]:
        p = c.get("itineraryPorts", None)
        if p is None:
            p = c.get("ports", None)
        return [str(x) for x in p] if isinstance(p, list) else []

    def solve(
        self,
        cruises: List[Dict],
        constraints: Dict,
        objective_fn: Callable,
        *,
        preferred_duration: Optional[int] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        time_limit_seconds: int = 10,
    ) -> Optional[Dict]:
        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            raise RuntimeError("CBC solver is not available.")

        solver.SetTimeLimit(int(time_limit_seconds) * 1000)

        # IMPORTANT:
        # - If user has a budget constraint, we must exclude cruises with missing price.
        # - If no budget constraint, allow missing price cruises (but objective_fn may not like it).
        hard = (constraints or {}).get("hard_constraints", constraints or {})  # tolerate both shapes
        max_budget = hard.get("max_budget")

        filtered: List[Dict[str, Any]] = []
        for c in cruises:
            price = self._get_price(c)

            # If budget is specified, price must be known
            if max_budget is not None and price is None:
                continue

            c2 = dict(c)
            if price is not None:
                # normalize for internal use so objective_fn can rely on it
                c2["roomPriceWithTaxesFees"] = price
            filtered.append(c2)

        if not filtered:
            return None

        cruises = filtered
        n = len(cruises)

        # Decision variables: x_i in {0,1}
        x: Dict[int, pywraplp.Variable] = {i: solver.BoolVar(f"x_{i}") for i in range(n)}

        # Select exactly one cruise
        solver.Add(sum(x.values()) == 1)

        soft = (constraints or {}).get("soft_preferences", {})

        # Hard constraint enforcement
        for i, cruise in enumerate(cruises):

            # soldOut exclusion
            if hard.get("exclude_sold_out") and bool(cruise.get("soldOut", False)):
                solver.Add(x[i] == 0)

            # budget (only safe because we filtered missing price when max_budget exists)
            if max_budget is not None:
                if float(cruise.get("roomPriceWithTaxesFees", 1e18)) > float(max_budget):
                    solver.Add(x[i] == 0)

            # duration range
            dr = hard.get("duration_range")
            if dr is not None:
                d = int(cruise.get("duration", 0) or 0)
                if not (int(dr["min_days"]) <= d <= int(dr["max_days"])):
                    solver.Add(x[i] == 0)

            # departure date window (string YYYY-MM-DD works lexicographically)
            dw = hard.get("departure_date_window")
            if dw is not None:
                dep = cruise.get("departureDate") or ""
                if not dep:
                    solver.Add(x[i] == 0)
                else:
                    earliest = dw.get("earliest")
                    latest = dw.get("latest")
                    if earliest and dep < earliest:
                        solver.Add(x[i] == 0)
                    if latest and dep > latest:
                        solver.Add(x[i] == 0)

            # allowed destinations (match ANY allowed code)
            allowed = hard.get("allowed_destinations") or []
            if allowed:
                dests = set(self._get_dests(cruise))
                if dests.isdisjoint(set(allowed)):
                    solver.Add(x[i] == 0)

            # required ports (must include ALL required ports)
            required_ports = hard.get("required_ports") or []
            if required_ports:
                ports = set(self._get_ports(cruise))
                if not set(required_ports).issubset(ports):
                    solver.Add(x[i] == 0)

        # Objective
        preferred_duration = soft.get("preferred_duration_days", preferred_duration)
        objective_fn(
            solver=solver,
            x=x,
            cruises=cruises,
            preferred_duration=preferred_duration,
            alpha=alpha,
            beta=beta,
        )

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        # Extract chosen cruise
        best_i = None
        best_val = -1.0
        for i in range(n):
            val = x[i].solution_value()
            if val > best_val:
                best_val = val
                best_i = i

        if best_i is None or best_val < 0.5:
            return None

        return cruises[best_i]