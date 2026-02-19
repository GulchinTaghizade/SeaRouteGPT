from typing import Optional, Dict, Any, List

from models.llm.llm_constraint_extractor import LLMConstraintExtractor
from solvers.milp_solver import MILPSolver
from solvers.objective import utility_objective


class HybridSolver:
    """
    Hybrid LLM + MILP solver for cruise planning.

    1) LLM extracts structured constraints from natural language (cached by request_id)
    2) MILP selects best cruise under hard constraints using utility objective
    """

    def __init__(self, api_key: str = None):
        self.llm_extractor = LLMConstraintExtractor(api_key=api_key)
        self.milp_solver = MILPSolver()

    @staticmethod
    def _midpoint_duration(duration_range: Optional[Dict[str, Any]]) -> Optional[int]:
        if not duration_range:
            return None
        try:
            mn = int(duration_range.get("min_days"))
            mx = int(duration_range.get("max_days"))
            return (mn + mx) // 2
        except Exception:
            return None

    def solve(
        self,
        user_request: str,
        cruises: List[dict],
        preferred_duration: Optional[int] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        request_id: Optional[str] = None,
        time_limit_seconds: int = 10,
    ) -> Dict[str, Any]:

        if request_id is None:
            raise ValueError("request_id is required (e.g., 'req_001') for stable caching/reproducibility.")
        rid = request_id

        # 1) LLM constraint extraction (cached by rid)
        extracted = self.llm_extractor.extract_constraints(user_request, rid)
        hard = extracted.get("hard_constraints", {}) or {}
        soft = extracted.get("soft_preferences", {}) or {}

        # 2) Preferred duration selection logic (for objective + metrics)
        # Priority:
        #   (a) explicit override argument
        #   (b) LLM soft preference
        #   (c) midpoint of hard duration_range
        if preferred_duration is not None:
            pref_duration = preferred_duration
        else:
            pref_duration = soft.get("preferred_duration_days")
            if pref_duration is None:
                pref_duration = self._midpoint_duration(hard.get("duration_range"))

        # ensure it's stored for later metrics/logging
        soft = {**soft, "preferred_duration_days": pref_duration}

        constraints = {
            "hard_constraints": hard,
            "soft_preferences": soft,
        }

        # 3) MILP solve
        selected_cruise = self.milp_solver.solve(
            cruises=cruises,
            constraints=constraints,
            objective_fn=utility_objective,
            preferred_duration=pref_duration,
            alpha=alpha,
            beta=beta,
            time_limit_seconds=time_limit_seconds,
        )

        if selected_cruise is None:
            return {
                "status": "no_valid_cruises",
                "message": "No cruises satisfy extracted constraints (or solver infeasible/time-limit).",
                "request_id": rid,
                "constraints_extracted": hard,
                "preferences_extracted": soft,
                "llm_metadata": extracted.get("metadata", {}) or {},
            }

        return {
            "status": "success",
            "request_id": rid,
            "selected_cruise": selected_cruise,
            "constraints_extracted": hard,
            "preferences_extracted": soft,
            "llm_metadata": extracted.get("metadata", {}) or {},
        }