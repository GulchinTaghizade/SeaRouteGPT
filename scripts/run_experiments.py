# run_experiments.py  (BASELINE ONLY)
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import CruiseMetrics, ExperimentRun
from evaluation.aggregation import summarize_all


from models.baseline.baseline_constraint_extractor import ConstraintExtractor
from models.baseline.rule_based_planner import RuleBasedPlanner


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def main():
    # ---- config ----
    cruises_path = "data/raw/cruises.json"
    requests_path = "data/synthetic/user_requests.json"

    out_raw = Path("results/baseline_raw_runs.jsonl")
    out_summary = Path("results/baseline_summary.json")

    n_runs_per_request = 1  # baseline is deterministic; keep 1
    base_seed = 123

    cruises_json = load_json(cruises_path)
    cruises = cruises_json["data"] if isinstance(cruises_json, dict) and "data" in cruises_json else cruises_json

    requests = load_json(requests_path)

    # ---- init ----
    extractor = ConstraintExtractor()
    planner = RuleBasedPlanner()
    metrics = CruiseMetrics(cruise_catalog=cruises)

    all_runs: List[ExperimentRun] = []

    out_raw.parent.mkdir(parents=True, exist_ok=True)

    with out_raw.open("w", encoding="utf-8") as fraw:
        for req in requests:
            rid = req["request_id"]
            text = req["text"]

            for run_idx in range(1, n_runs_per_request + 1):
                seed = base_seed + (hash(rid) % 100000) + run_idx
                random.seed(seed)

                try:
                    # 1) Extract constraints
                    extracted = extractor.extract_constraints(text, rid)
                    hard = extracted.get("hard_constraints", {}) or {}
                    soft = extracted.get("soft_preferences", {}) or {}

                    constraints = {"hard_constraints": hard, "soft_preferences": soft}

                    # 2) Baseline plan
                    selected = planner.plan(cruises, constraints)

                    # IMPORTANT: selected might be a reduced dict.
                    # Convert it to full cruise dict using cruiseId.
                    selected_full = None
                    if selected is not None:
                        sel_id = selected.get("cruiseId") or selected.get("cruise_id")
                        if sel_id:
                            selected_full = next((c for c in cruises if c.get("cruiseId") == sel_id), None)

                    itinerary = metrics.to_itinerary(selected_full)

                    # 3) Metrics (baseline uses extracted hard constraints)
                    feas, violations = metrics.compute_feasibility(hard, itinerary)

                    # Unified failure handling
                    if feas < 0.5:
                        pers = 0.0
                        util = 0.0
                    else:
                        # Personalization uses what baseline extractor returned as soft prefs
                        pers = metrics.compute_personalization(feas, soft, itinerary)

                        # Utility uses feasible candidate set for normalization
                        feasible_candidates = metrics.feasible_candidate_set(hard)
                        util = metrics.compute_optimization_utility(
                            feas, hard, soft, itinerary,
                            alpha=0.6, beta=0.4,
                            feasible_candidates=feasible_candidates,
                        )

                    run = ExperimentRun(
                        request_id=rid,
                        method_name="BASELINE",
                        run_number=run_idx,
                        seed=seed,
                        feasibility=feas,
                        personalization=pers,
                        optimization_utility=util,
                        itinerary=itinerary,
                        error_msg="" if feas >= 0.5 else ",".join(violations),
                    )

                except Exception as e:
                    run = ExperimentRun(
                        request_id=rid,
                        method_name="BASELINE",
                        run_number=run_idx,
                        seed=seed,
                        feasibility=0.0,
                        personalization=0.0,
                        optimization_utility=0.0,
                        itinerary=None,
                        error_msg=f"exception:{type(e).__name__}:{str(e)[:160]}",
                    )

                all_runs.append(run)

                # write raw row (jsonl)
                fraw.write(json.dumps({
                    "request_id": run.request_id,
                    "method": run.method_name,
                    "run_number": run.run_number,
                    "seed": run.seed,
                    "feasibility": run.feasibility,
                    "personalization": run.personalization,
                    "utility": run.optimization_utility,
                    "error_msg": run.error_msg,
                    "constraints": {
                        "hard": hard if "hard" in locals() else None,
                        "soft": soft if "soft" in locals() else None,
                    },
                    "itinerary": None if run.itinerary is None else {
                        "cruise_id": run.itinerary.cruise_id,
                        "departure_date": run.itinerary.departure_date,
                        "duration_days": run.itinerary.duration_days,
                        "total_price": run.itinerary.total_price,
                        "ports": run.itinerary.ports,
                        "destinations": run.itinerary.destinations,
                        "violations": run.itinerary.constraint_violations,
                        "pref_matches": run.itinerary.preference_matches,
                    }
                }, ensure_ascii=False) + "\n")

    summary = summarize_all(all_runs)
    out_summary.write_text(json.dumps(summary, indent=2))
    print("Wrote:", out_raw, out_summary)


if __name__ == "__main__":
    main()