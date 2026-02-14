import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import CruiseMetrics, ExperimentRun, Itinerary
from evaluation.aggregation import summarize_all

from solvers.milp_solver import MILPSolver
from solvers.objective import utility_objective


# ===== PATHS (edit if your folders differ) =====
CRUISE_CATALOG_PATH = Path("data/raw/cruises.json")
USER_REQUESTS_PATH = Path("data/synthetic/user_requests.json")

# cached constraints per request_id (your hybrid constraint extraction cache)
CONSTRAINTS_CACHE_DIR = Path("data/processed/llm_cache_milp")

OUT_RAW = Path("results/hybrid_raw_runs.jsonl")
OUT_SUMMARY = Path("results/hybrid_summary.json")


# ===== HELPERS =====
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_requests(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        if "requests" in data and isinstance(data["requests"], list):
            return {r["request_id"]: r for r in data["requests"]}
        if all(isinstance(k, str) and isinstance(v, dict) for k, v in data.items()):
            return data
        if "request_id" in data:
            return {data["request_id"]: data}
    if isinstance(data, list):
        return {r["request_id"]: r for r in data}
    raise ValueError(f"Unsupported requests format: {path}")


def load_cruise_catalog(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported cruise catalog format: {path}")


def load_constraints_cache(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Expects one JSON file per request_id, named like req_001.json, etc.
    Each file should contain:
      {
        "hard_constraints": {...},
        "soft_preferences": {...},
        "metadata": {...}
      }
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"Constraints cache dir not found: {cache_dir}")

    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(cache_dir.glob("*.json")):
        obj = load_json(p)
        rid = obj.get("metadata", {}).get("request_id") or obj.get("request_id") or p.stem
        out[rid] = obj
    return out


def build_itinerary_from_catalog(cruise: Dict[str, Any]) -> Optional[Itinerary]:
    """Convert a selected cruise dict into evaluation.Itinerary"""
    if not cruise:
        return None
    cid = cruise.get("cruiseId") or cruise.get("cruise_id")
    if not cid:
        return None

    try:
        price = float(cruise.get("roomPriceWithTaxesFees") or 0.0)
    except (ValueError, TypeError):
        price = 0.0

    return Itinerary(
        cruise_id=cid,
        ports=list(cruise.get("itineraryPorts", []) or []),
        departure_date=str(cruise.get("departureDate") or ""),
        duration_days=int(cruise.get("duration") or 0),
        total_price=price,
        cruise_line=cruise.get("cruiseLineCode"),
        cabin_category=cruise.get("roomTypeCategoryCode"),
        sold_out=bool(cruise.get("soldOut", False)),
        destinations=list(cruise.get("itineraryDestinations", []) or []),
    )


# ===== MAIN =====
def main():
    print("=== HYBRID (LLM constraints + MILP) Evaluation ===\n")

    cruise_catalog = load_cruise_catalog(CRUISE_CATALOG_PATH)
    requests_by_id = load_requests(USER_REQUESTS_PATH)
    constraints_by_id = load_constraints_cache(CONSTRAINTS_CACHE_DIR)

    print(f"Catalog cruises: {len(cruise_catalog)}")
    print(f"User requests: {len(requests_by_id)}")
    print(f"Constraints cached: {len(constraints_by_id)}\n")

    metrics_engine = CruiseMetrics(cruise_catalog)
    milp = MILPSolver()

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    runs: List[ExperimentRun] = []

    with OUT_RAW.open("w", encoding="utf-8") as fraw:
        for idx, (request_id, user_req) in enumerate(sorted(requests_by_id.items()), 1):
            cached = constraints_by_id.get(request_id)
            if cached is None:
                # If constraints missing, treat as failure consistently
                exp = ExperimentRun(
                    request_id=request_id,
                    method_name="HYBRID",
                    run_number=1,
                    seed=0,
                    feasibility=0.0,
                    personalization=0.0,
                    optimization_utility=0.0,
                    itinerary=None,
                    error_msg="missing_constraints_cache",
                )
                runs.append(exp)
                fraw.write(json.dumps({"request_id": request_id, "error": "missing_constraints_cache"}) + "\n")
                continue

            hard = cached.get("hard_constraints", {}) or {}
            soft = cached.get("soft_preferences", {}) or {}
            constraints = {"hard_constraints": hard, "soft_preferences": soft}

            # Run MILP on full catalog
            selected_cruise = milp.solve(
                cruises=cruise_catalog,
                constraints=constraints,
                objective_fn=utility_objective,
                preferred_duration=soft.get("preferred_duration_days"),
                alpha=0.6,
                beta=0.4,
                time_limit_seconds=10,
            )

            itinerary = build_itinerary_from_catalog(selected_cruise) if selected_cruise else None

            # Metrics:
            feas, violations = metrics_engine.compute_feasibility(hard, itinerary)
            pers = metrics_engine.compute_personalization(feasibility=feas, soft_preferences=soft, itinerary=itinerary)

            feasible_candidates = metrics_engine.feasible_candidate_set(hard)
            util = metrics_engine.compute_optimization_utility(
                feasibility=feas,
                hard_constraints=hard,
                soft_preferences=soft,
                itinerary=itinerary,
                alpha=0.6,
                beta=0.4,
                feasible_candidates=feasible_candidates,
            )

            exp = ExperimentRun(
                request_id=request_id,
                method_name="HYBRID",
                run_number=1,
                seed=0,
                feasibility=feas,
                personalization=pers,
                optimization_utility=util,
                itinerary=itinerary,
                error_msg="",
            )
            runs.append(exp)

            fraw.write(json.dumps({
                "request_id": request_id,
                "selected_cruise_id": itinerary.cruise_id if itinerary else "NO_ITINERARY",
                "feasibility": feas,
                "personalization": pers,
                "optimization_utility": util,
                "violations": violations,
            }) + "\n")

            if idx % 10 == 0:
                print(f"  ✓ processed {idx}/{len(requests_by_id)}")

    summary = summarize_all(runs)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n✓ wrote:")
    print(f"  raw: {OUT_RAW}")
    print(f"  summary: {OUT_SUMMARY}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()