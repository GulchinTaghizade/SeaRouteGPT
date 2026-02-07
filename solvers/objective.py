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
    Weighted multi-objective optimization.

    weights example:
    {
        "price": 1.0,
        "duration": 0.3
    }
    """
    if weights is None:
        weights = {"price": 1.0, "duration": 0.0}

    solver.Minimize(
        sum(
            x[i] * (
                weights.get("price", 0.0) * cruises[i]["roomPriceWithTaxesFees"]
                - weights.get("duration", 0.0) * cruises[i]["duration"]
            )
            for i in x
        )
    )