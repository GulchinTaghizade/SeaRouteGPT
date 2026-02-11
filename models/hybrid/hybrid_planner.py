from models.llm.llm_constraint_extractor import LLMConstraintExtractor
from solvers.milp_solver import MILPSolver
from solvers.objective import utility_objective
import uuid
from typing import Optional, Dict, Any, List


class HybridSolver:
    """
    Hybrid LLM + MILP solver for cruise planning.

    Pipeline:
    1) LLM extracts structured constraints from natural language
    2) MILP selects best cruise under hard constraints using utility objective
    """

    def __init__(self, api_key: str = None):
        self.llm_extractor = LLMConstraintExtractor(api_key=api_key)
        self.milp_solver = MILPSolver()

    @staticmethod
    def _midpoint_duration(duration_range: Optional[Dict[str, Any]]) -> Optional[int]:
        """Return midpoint of duration range if present."""
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
        """
        Args:
            user_request: Natural language user request
            cruises: Candidate cruise list
            preferred_duration: Optional override (days). If None, inferred from LLM duration_range midpoint.
            alpha/beta: Utility weights (alpha+beta should be 1 in your thesis metric)
            request_id: Optional stable ID (e.g., "req_001"). If None, random UUID is used.
            time_limit_seconds: MILP time limit

        Returns:
            Dict with selected cruise + extracted constraints + metadata
        """
        rid = request_id or str(uuid.uuid4())

        # 1) LLM constraint extraction
        extracted = self.llm_extractor.extract_constraints(user_request, rid)
        hard = extracted.get("hard_constraints", {}) or {}
        soft = extracted.get("soft_preferences", {}) or {}

        # 2) Determine preferred duration:
        #    - If caller passed preferred_duration, use it.
        #    - Else infer from extracted duration_range midpoint.
        inferred_pref_duration = self._midpoint_duration(hard.get("duration_range"))
        pref_duration = preferred_duration if preferred_duration is not None else inferred_pref_duration

        # Keep it in soft prefs for logging/metrics, but also pass it into objective via kwargs.
        soft = {**soft, "preferred_duration_days": pref_duration}

        constraints = {
            "hard_constraints": hard,
            "soft_preferences": soft,
        }

        # 3) MILP optimization (objective MUST receive preferred_duration to use duration term)
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
                "message": "No cruises satisfy the extracted constraints (or solver hit infeasible/time limit).",
                "request_id": rid,
                "constraints_extracted": hard,
                "preferences_extracted": soft,
            }
        print(extracted["metadata"]["llm_response"])

        return {
            "status": "success",
            "request_id": rid,
            "selected_cruise": selected_cruise,
            "constraints_extracted": hard,
            "preferences_extracted": soft,
        }