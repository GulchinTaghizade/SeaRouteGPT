from typing import List, Dict, Optional, Any


class RuleBasedPlanner:
    """
    Baseline Rule-Based Planner

    Step 1: Hard filtering (strict hard constraints)
    Step 2: Greedy selection via thesis-style utility proxy:
            maximize alpha*(1 - C_hat) + beta*(1 - T_hat)
    """

    def __init__(self):
        pass

    def plan(self, cruises: List[Dict], constraints: Dict) -> Optional[Dict]:
        hard = constraints.get("hard_constraints", {}) or {}
        soft = constraints.get("soft_preferences", {}) or {}

        # Step 1: Hard filter
        feasible_cruises: List[Dict] = [c for c in cruises if self._satisfies_constraints(c, hard)]

        if not feasible_cruises:
            return None

        preferred_duration = soft.get("preferred_duration_days")

        # If no budget constraint, we still prefer cruises with prices for scoring.
        # But unlike budget, missing price is not automatically infeasible unless max_budget is set.
        cruises_with_prices = [c for c in feasible_cruises if c.get("roomPriceWithTaxesFees") is not None]

        # If we can't score anything (no prices anywhere), just return first feasible cruise
        if not cruises_with_prices:
            best = feasible_cruises[0]
            return self._return_minimal(best)

        # Normalize prices across cruises_with_prices
        prices = []
        for c in cruises_with_prices:
            try:
                prices.append(float(c["roomPriceWithTaxesFees"]))
            except Exception:
                pass

        if not prices:
            best = feasible_cruises[0]
            return self._return_minimal(best)

        cmin, cmax = min(prices), max(prices)
        cden = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

        # Duration deviation normalization: use max deviation in candidate set
        deviations: List[float] = []
        if preferred_duration is not None:
            try:
                d_star = float(preferred_duration)
            except Exception:
                d_star = None
        else:
            d_star = None

        for c in cruises_with_prices:
            d = self._safe_int(c.get("duration"))
            if d_star is None:
                deviations.append(0.0)
            else:
                deviations.append(abs(d - d_star))

        tden = max(deviations) if deviations and max(deviations) != 0 else 1.0

        alpha = 0.6
        beta = 0.4

        def utility(c: Dict) -> float:
            # C_hat
            try:
                price = float(c.get("roomPriceWithTaxesFees"))
                c_hat = (price - cmin) / cden
            except Exception:
                # unknown price is worst (so 1 - C_hat becomes 0)
                c_hat = 1.0

            # T_hat
            d = self._safe_int(c.get("duration"))
            if d_star is None:
                t_hat = 0.0
            else:
                t_hat = abs(d - d_star) / tden

            # Thesis utility (higher is better)
            u = alpha * (1.0 - c_hat) + beta * (1.0 - t_hat)
            return float(u)

        best = max(cruises_with_prices, key=utility)
        return self._return_minimal(best)

    # -------------------------
    # HARD CONSTRAINTS (strict)
    # -------------------------
    def _satisfies_constraints(self, cruise: Dict, hard: Dict) -> bool:
        # 1) Availability
        if hard.get("exclude_sold_out", False) and bool(cruise.get("soldOut", False)):
            return False

        # 2) Budget (strict): if max_budget is provided, price MUST exist and be <= budget
        max_budget = hard.get("max_budget")
        if max_budget is not None:
            price = cruise.get("roomPriceWithTaxesFees")
            if price is None:
                return False
            try:
                if float(price) > float(max_budget):
                    return False
            except Exception:
                return False

        # 3) Temporal validity: departure date window
        dw = hard.get("departure_date_window")
        if dw is not None:
            dep = cruise.get("departureDate")
            if not dep:
                return False
            earliest = dw.get("earliest")
            latest = dw.get("latest")
            if earliest and dep < earliest:
                return False
            if latest and dep > latest:
                return False

        # 4) Temporal validity: duration range
        dr = hard.get("duration_range")
        if dr is not None:
            d = self._safe_int(cruise.get("duration"))
            try:
                mn = int(dr.get("min_days"))
                mx = int(dr.get("max_days"))
            except Exception:
                return False
            if not (mn <= d <= mx):
                return False

        # 5) Destination validity (intersection)
        allowed = hard.get("allowed_destinations")
        if allowed:
            dests = cruise.get("itineraryDestinations") or []
            if not any(a in dests for a in allowed):
                return False

        # num_guests: your catalog likely doesn't support max_guests consistently; keep it optional
        num_guests = hard.get("num_guests")
        if num_guests is not None:
            cruise_max = cruise.get("max_guests")
            if cruise_max is not None:
                try:
                    if int(num_guests) > int(cruise_max):
                        return False
                except Exception:
                    # if can't compare, don't fail (since catalog may not define it reliably)
                    pass

        return True

    # -------------------------
    # DEBUG COUNTS
    # -------------------------
    def debug_feasibility_counts(self, cruises: List[Dict], constraints: Dict) -> Dict[str, int]:
        hard = constraints.get("hard_constraints", {}) or {}

        def pass_budget(c):
            mb = hard.get("max_budget")
            if mb is None:
                return True
            p = c.get("roomPriceWithTaxesFees")
            if p is None:
                return False
            try:
                return float(p) <= float(mb)
            except Exception:
                return False

        def pass_date(c):
            dw = hard.get("departure_date_window")
            if not dw:
                return True
            dep = c.get("departureDate")
            if not dep:
                return False
            e = dw.get("earliest")
            l = dw.get("latest")
            if e and dep < e:
                return False
            if l and dep > l:
                return False
            return True

        def pass_duration(c):
            dr = hard.get("duration_range")
            if not dr:
                return True
            d = self._safe_int(c.get("duration"))
            try:
                mn = int(dr.get("min_days"))
                mx = int(dr.get("max_days"))
            except Exception:
                return False
            return mn <= d <= mx

        def pass_dest(c):
            allowed = hard.get("allowed_destinations")
            if not allowed:
                return True
            dests = c.get("itineraryDestinations") or []
            return any(a in dests for a in allowed)

        def pass_soldout(c):
            if not hard.get("exclude_sold_out", False):
                return True
            return not bool(c.get("soldOut", False))

        total = len(cruises)

        # sequential counts are the most useful
        seq1 = [c for c in cruises if pass_budget(c)]
        seq2 = [c for c in seq1 if pass_date(c)]
        seq3 = [c for c in seq2 if pass_duration(c)]
        seq4 = [c for c in seq3 if pass_dest(c)]
        seq5 = [c for c in seq4 if pass_soldout(c)]

        return {
            "total": total,
            "after_budget": len(seq1),
            "after_date_window": len(seq2),
            "after_duration": len(seq3),
            "after_destination": len(seq4),
            "after_sold_out": len(seq5),
        }

    # -------------------------
    # HELPERS
    # -------------------------
    def _return_minimal(self, cruise: Dict) -> Dict:
        return {
            "cruiseId": cruise.get("cruiseId"),
            "cruiseName": cruise.get("cruiseName"),
            "price": cruise.get("roomPriceWithTaxesFees"),
            "departureDate": cruise.get("departureDate"),
            "duration": cruise.get("duration"),
        }

    def _safe_int(self, x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0