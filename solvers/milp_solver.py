from typing import Dict, List, Optional, Callable
from ortools.linear_solver import pywraplp


class MILPSolver:
    """
    MILP solver for selecting ONE optimal cruise given hard constraints.
    """

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

        solver.SetTimeLimit(time_limit_seconds * 1000)

        # Filter cruises with valid numeric price
        filtered = []
        for c in cruises:
            price = c.get("roomPriceWithTaxesFees")
            if price is None:
                continue
            try:
                c2 = dict(c)
                c2["roomPriceWithTaxesFees"] = float(price)
                filtered.append(c2)
            except (ValueError, TypeError):
                continue

        if not filtered:
            return None

        cruises = filtered
        n = len(cruises)

        # Decision variables: x_i in {0,1}
        x: Dict[int, pywraplp.Variable] = {i: solver.BoolVar(f"x_{i}") for i in range(n)}

        # Select exactly one cruise
        solver.Add(sum(x.values()) == 1)

        hard = constraints.get("hard_constraints", {})
        soft = constraints.get("soft_preferences", {})

        # Hard constraint enforcement
        for i, cruise in enumerate(cruises):

            # soldOut exclusion
            if hard.get("exclude_sold_out") and cruise.get("soldOut", False):
                solver.Add(x[i] == 0)

            # budget
            max_budget = hard.get("max_budget")
            if max_budget is not None and cruise["roomPriceWithTaxesFees"] > float(max_budget):
                solver.Add(x[i] == 0)

            # duration range
            dr = hard.get("duration_range")
            if dr is not None:
                d = int(cruise.get("duration", 0))
                if not (int(dr["min_days"]) <= d <= int(dr["max_days"])):
                    solver.Add(x[i] == 0)

            # departure date window (string YYYY-MM-DD works lexicographically)
            dw = hard.get("departure_date_window")
            if dw is not None:
                dep = cruise.get("departureDate")
                if dep is None:
                    solver.Add(x[i] == 0)
                else:
                    earliest = dw.get("earliest")
                    latest = dw.get("latest")
                    if earliest and dep < earliest:
                        solver.Add(x[i] == 0)
                    if latest and dep > latest:
                        solver.Add(x[i] == 0)

            # allowed destinations
            allowed = hard.get("allowed_destinations")
            if allowed:
                dests = cruise.get("itineraryDestinations", [])
                if not any(d in dests for d in allowed):
                    solver.Add(x[i] == 0)

            # required ports
            required_ports = hard.get("required_ports") or []
            if required_ports:
                ports = cruise.get("itineraryPorts", [])
                if not any(p in ports for p in required_ports):
                    solver.Add(x[i] == 0)

        # Objective
        preferred_duration = soft.get("preferred_duration_days")
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
        best_val = -1
        for i in range(n):
            val = x[i].solution_value()
            if val > best_val:
                best_val = val
                best_i = i

        if best_i is None or best_val < 0.5:
            return None

        return cruises[best_i]