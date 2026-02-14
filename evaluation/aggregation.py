
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from scipy import stats

from evaluation.metrics import ExperimentRun


def aggregate_runs(runs: List[ExperimentRun], metric_name: str) -> Dict[str, Any]:
    if not runs:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_runs": 0}

    vals = np.array([float(getattr(r, metric_name)) for r in runs], dtype=float)
    mean = float(np.mean(vals))
    n = len(vals)
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    if n > 1 and std > 0:
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
        lo, hi = float(ci[0]), float(ci[1])
    else:
        lo = hi = mean

    return {"mean": mean, "std": std, "ci_lower": lo, "ci_upper": hi, "n_runs": n}


def group_by_method(runs: List[ExperimentRun]) -> Dict[str, List[ExperimentRun]]:
    grouped: Dict[str, List[ExperimentRun]] = {}
    for r in runs:
        grouped.setdefault(r.method_name, []).append(r)
    return grouped


def compare_methods(method_a: List[ExperimentRun], method_b: List[ExperimentRun], metric_name: str) -> Dict[str, Any]:
    # Pair by request_id + run_number for a clean paired test
    index_a = {(r.request_id, r.run_number): getattr(r, metric_name) for r in method_a}
    index_b = {(r.request_id, r.run_number): getattr(r, metric_name) for r in method_b}
    keys = sorted(set(index_a.keys()).intersection(index_b.keys()))

    if not keys:
        return {"t_statistic": 0.0, "p_value": 1.0, "cohens_d": 0.0, "significant": False}

    a = np.array([float(index_a[k]) for k in keys], dtype=float)
    b = np.array([float(index_b[k]) for k in keys], dtype=float)

    t_stat, p_val = stats.ttest_rel(a, b)

    diff = a - b
    denom = np.std(diff, ddof=1)
    d = float(np.mean(diff) / denom) if denom > 0 else 0.0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": d,
        "significant": bool(p_val < 0.05),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
    }


def summarize_all(runs: List[ExperimentRun]) -> Dict[str, Any]:
    by_method = group_by_method(runs)
    summary: Dict[str, Any] = {}

    for method, rs in by_method.items():
        summary[method] = {
            "feasibility": aggregate_runs(rs, "feasibility"),
            "personalization": aggregate_runs(rs, "personalization"),
            "utility": aggregate_runs(rs, "optimization_utility"),
        }

    return summary