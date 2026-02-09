from ortools.linear_solver import pywraplp


class MILPSolver:
    """
    MILP solver for selecting an optimal cruise given structured constraints.
    """

    def __init__(self):
        self.solver = pywraplp.Solver.CreateSolver("CBC")
        if not self.solver:
            raise RuntimeError("CBC solver is not available.")

    def solve(self, cruises, constraints, objective_fn,**kwargs):
        """
        cruises: List[Dict] – cruise catalog
        constraints: Dict – structured constraints from LLM
        objective_fn: callable – objective function from objective.py
         **kwargs: additional arguments (like weights) passed to objective_fn
        """

        #  Filter cruises with invalid prices
        filtered_cruises = []
        for c in cruises:
            price = c.get("roomPriceWithTaxesFees")
            if price is None:
                continue
            try:
                c["roomPriceWithTaxesFees"] = float(price)
                filtered_cruises.append(c)
            except (ValueError, TypeError):
                continue

        if not filtered_cruises:
            return None

        cruises = filtered_cruises

        x = {}

        # Decision variables: x_i = 1 if cruise i is selected
        for i in range(len(cruises)):
            x[i] = self.solver.BoolVar(f"x_{i}")

        # Exactly one cruise must be selected
        self.solver.Add(sum(x.values()) == 1)

        hard = constraints.get("hard_constraints", {})

        for i, cruise in enumerate(cruises):

            # Budget constraint
            if hard.get("max_budget") is not None:
                self.solver.Add(
                    x[i] * cruise["roomPriceWithTaxesFees"]
                    <= hard["max_budget"]
                )

            # Duration constraint
            if hard.get("duration_range"):
                min_d = hard["duration_range"]["min_days"]
                max_d = hard["duration_range"]["max_days"]

                if not (min_d <= cruise["duration"] <= max_d):
                    self.solver.Add(x[i] == 0)

            # Destination constraint
            if hard.get("allowed_destinations"):
                if not any(
                    d in cruise.get("itineraryDestinations", [])
                    for d in hard["allowed_destinations"]
                ):
                    self.solver.Add(x[i] == 0)

            # Exclude sold-out cruises
            if hard.get("exclude_sold_out") and cruise.get("soldOut", False):
                self.solver.Add(x[i] == 0)

        # Apply objective function
        objective_fn(self.solver, x, cruises,**kwargs)

        status = self.solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            return None

        # Extract selected cruise
        for i in x:
            if x[i].solution_value() == 1:
                return cruises[i]

        return None