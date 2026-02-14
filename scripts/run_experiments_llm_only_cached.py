import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import fields

from evaluation.metrics import CruiseMetrics, ExperimentRun, Itinerary
from evaluation.aggregation import summarize_all

# ========= PATHS =========
LLM_CACHE_DIR = Path("data/processed/llm_cache")
CONSTRAINTS_CACHE_DIR = Path("data/processed/llm_cache_milp")

CRUISE_CATALOG_PATH = Path("data/raw/cruises.json")

OUT_RAW = Path("results/llm_only_cached_raw_runs.jsonl")
OUT_SUMMARY = Path("results/llm_only_cached_summary.json")


# ========= HELPERS =========
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    for block in parts:
        b = block.strip()
        if not b:
            continue
        if b.lower().startswith("json"):
            b = b[4:].strip()
        if b.startswith("{") and b.endswith("}"):
            return b
    return t.replace("```json", "").replace("```", "").strip()


def parse_selected_cruise_id(llm_output: str) -> str:
    if not llm_output:
        return ""
    clean = strip_json_fences(llm_output)
    try:
        obj = json.loads(clean)
        return obj.get("selectedCruiseId", "") or ""
    except Exception:
        return ""


def load_cruise_catalog(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported cruise catalog format: {path}")


def _make_itinerary_safe(**kwargs) -> Itinerary:
    allowed = {f.name for f in fields(Itinerary)}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return Itinerary(**filtered)


def build_itinerary_from_catalog(selected_id: str, cruise_catalog: List[Dict[str, Any]]) -> Optional[Itinerary]:
    # NO_VALID_CRUISE means no itinerary returned
    if not selected_id or selected_id == "NO_VALID_CRUISE":
        return None

    match = None
    for c in cruise_catalog:
        if c.get("cruiseId") == selected_id:
            match = c
            break

    # Hallucinated ID → return placeholder itinerary (feasibility will mark it invalid)
    if match is None:
        return _make_itinerary_safe(
            cruise_id=selected_id,
            ports=[],
            departure_date="",
            duration_days=0,
            total_price=0.0,
            cabin_category="",
            cruise_line="",
            sold_out=False,
        )

    try:
        price_val = float(match.get("roomPriceWithTaxesFees") or 0.0)
    except (ValueError, TypeError):
        price_val = 0.0

    return _make_itinerary_safe(
        cruise_id=match.get("cruiseId", ""),
        ports=match.get("itineraryPorts", []) or [],
        departure_date=match.get("departureDate", "") or "",
        duration_days=int(match.get("duration") or 0),
        total_price=price_val,
        cabin_category=str(match.get("roomTypeCategoryCode", "") or ""),
        cruise_line=str(match.get("cruiseLineCode", "") or ""),
        sold_out=bool(match.get("soldOut", False)),
    )


def load_constraints_cached(request_id: str) -> Dict[str, Any]:
    p = CONSTRAINTS_CACHE_DIR / f"{request_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing constraints cache for {request_id}: {p}")
    blob = load_json(p)
    return {
        "hard_constraints": blob.get("hard_constraints", {}) or {},
        "soft_preferences": blob.get("soft_preferences", {}) or {},
    }


def main():
    print("=== LLM-Only (CACHED) Evaluation ===\n")

    if not LLM_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing LLM cache dir: {LLM_CACHE_DIR}")
    if not CONSTRAINTS_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing constraints cache dir: {CONSTRAINTS_CACHE_DIR}")

    cruise_catalog = load_cruise_catalog(CRUISE_CATALOG_PATH)
    print(f"Loaded cruise catalog: {len(cruise_catalog)} cruises\n")

    metrics_engine = CruiseMetrics(cruise_catalog)

    cache_files = sorted(LLM_CACHE_DIR.glob("*.json"))
    print(f"Found cached LLM outputs: {len(cache_files)}\n")

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    runs: List[ExperimentRun] = []

    with OUT_RAW.open("w", encoding="utf-8") as fraw:
        for idx, cache_path in enumerate(cache_files, 1):
            cached = load_json(cache_path)
            request_id = cached.get("request_id") or cache_path.stem
            llm_output = cached.get("llm_output", "")

            # Load cached constraints for THIS request
            constraints = load_constraints_cached(request_id)
            hard = constraints["hard_constraints"]
            soft = constraints["soft_preferences"]

            # Parse selected cruise
            selected_id = parse_selected_cruise_id(llm_output)
            itinerary = build_itinerary_from_catalog(selected_id, cruise_catalog)

            # ---- METRICS ----
            feas, violations = metrics_engine.compute_feasibility(
                hard_constraints=hard,
                itinerary=itinerary,
            )

            pers = metrics_engine.compute_personalization(
                feasibility=feas,
                soft_preferences=soft,
                itinerary=itinerary,
            )

            util = metrics_engine.compute_optimization_utility(
                feasibility=feas,
                hard_constraints=hard,
                soft_preferences=soft,
                itinerary=itinerary,
                alpha=0.6,
                beta=0.4,
                feasible_candidates=None,  # ok if your function handles None -> 0 or uses full catalog internally
            )

            run = ExperimentRun(
                request_id=request_id,
                method_name="LLM_ONLY",
                run_number=1,
                seed=0,
                feasibility=feas,
                personalization=pers,
                optimization_utility=util,
                itinerary=itinerary,
                error_msg="",
            )
            runs.append(run)

            fraw.write(json.dumps({
                "request_id": request_id,
                "selected_cruise_id": selected_id,
                "feasibility": feas,
                "personalization": pers,
                "optimization_utility": util,
                "violations": violations,
            }) + "\n")

            if idx % 10 == 0:
                print(f"  ✓ processed {idx}/{len(cache_files)}")

    summary = summarize_all(runs)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Raw runs:    {OUT_RAW}")
    print(f"Summary:     {OUT_SUMMARY}")


if __name__ == "__main__":
    main()