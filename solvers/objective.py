from typing import Dict, List, Optional
from ortools.linear_solver import pywraplp


def utility_objective(
    solver: pywraplp.Solver,
    x: Dict[int, pywraplp.Variable],
    cruises: List[Dict],
    preferred_duration: Optional[int] = None,
    alpha: float = 0.6,
    beta: float = 0.4,
):
    """
    Maximize utility:
        U = alpha * (1 - C_hat) + beta * (1 - T_hat)

    where:
        C_hat = (C - Cmin) / (Cmax - Cmin)
        T_hat = |D - D*| / max|D - D*|  (over candidate cruises)

    OR-Tools CBC minimizes, so we minimize (-U).
    """

    # Collect numeric prices
    prices = [float(c["roomPriceWithTaxesFees"]) for c in cruises]
    cmin, cmax = min(prices), max(prices)
    cden = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

    # Duration deviation normalization
    if preferred_duration is None:
        tden = 1.0
        deviations = [0.0 for _ in cruises]
    else:
        deviations = [abs(int(c["duration"]) - int(preferred_duration)) for c in cruises]
        tden = max(deviations) if max(deviations) != 0 else 1.0

    # Build linear objective: minimize (-U)
    utility_terms=[]
    for i, c in enumerate(cruises):
        price = float(c["roomPriceWithTaxesFees"])
        c_hat = (price - cmin) / cden

        if preferred_duration is None:
            t_hat = 0.0
        else:
            t_hat = deviations[i] / tden

        u_i = alpha * (1.0 - c_hat) + beta * (1.0 - t_hat)
        utility_terms.append(x[i] * (-u_i))

    solver.Minimize(solver.Sum(utility_terms))