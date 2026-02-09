def minimize_price(solver, x, cruises):
    """
    Minimize total cruise price.
    """
    solver.Minimize(
        sum(
            x[i] * cruises[i]["roomPriceWithTaxesFees"]
            for i in x
        )
    )


def maximize_duration(solver, x, cruises):
    """
    Maximize cruise duration (converted to minimization).
    """
    solver.Minimize(
        sum(
            -x[i] * cruises[i]["duration"]
            for i in x
        )
    )



def weighted_objective(solver, x, cruises, weights=None):
    """
    Weighted objective: balance price and duration.

    Args:
        solver: MILP solver instance
        x: decision variables (cruise selection)
        cruises: list of cruise objects
        weights: dict with "price" and "duration" keys
    """
    if weights is None:
        weights = {"price": 1.0, "duration": 0.5}

    price_weight = weights.get("price", 1.0)
    duration_weight = weights.get("duration", 0.5)

    solver.Minimize(
        sum(
            x[i] * (
                    price_weight * cruises[i]["roomPriceWithTaxesFees"]
                    - duration_weight * cruises[i]["duration"]
            )
            for i in x
        )
    )

