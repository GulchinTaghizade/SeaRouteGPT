from typing import Dict, List, Optional, Any
from ortools.linear_solver import pywraplp


def _safe_price(c: Dict[str, Any], default: float) -> float:
    p = c.get("roomPriceWithTaxesFees")
    try:
        return float(p)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


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

    # ---- prices (robust) ----
    known_prices: List[float] = []
    for c in cruises:
        p = c.get("roomPriceWithTaxesFees")
        try:
            known_prices.append(float(p))
        except (TypeError, ValueError):
            pass

    # If everything is missing, use 1.0 just to avoid division by 0.
    # (Then C_hat becomes 0 for all, so price doesn't affect the objective.)
    default_price = max(known_prices) if known_prices else 1.0

    prices = [_safe_price(c, default=default_price) for c in cruises]

    cmin, cmax = min(prices), max(prices)
    cden = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

    # ---- duration deviation normalization (robust) ----
    if preferred_duration is None:
        deviations = [0.0 for _ in cruises]
        tden = 1.0
    else:
        d_star = _safe_int(preferred_duration, default=0)
        deviations = [abs(_safe_int(c.get("duration"), default=0) - d_star) for c in cruises]
        tmax = max(deviations) if deviations else 0.0
        tden = tmax if tmax != 0 else 1.0

    # ---- linear objective: minimize (-U) ----
    utility_terms = []
    for i, c in enumerate(cruises):
        price = prices[i]  # safe float
        c_hat = (price - cmin) / cden

        if preferred_duration is None:
            t_hat = 0.0
        else:
            t_hat = deviations[i] / tden

        u_i = alpha * (1.0 - c_hat) + beta * (1.0 - t_hat)
        utility_terms.append(x[i] * (-u_i))

    solver.Minimize(solver.Sum(utility_terms))