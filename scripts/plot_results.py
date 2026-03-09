from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
MODEL_DIR = RESULTS_DIR / "gemini-2-5-pro"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_mean(summary: Dict, method_name: str, metric_key: str) -> float:
    return float(summary[method_name][metric_key]["mean"])


def average_metric_across_runs(paths: List[Path], method_name: str, metric_key: str) -> float:
    vals = []
    for p in paths:
        summary = load_json(p)
        vals.append(get_metric_mean(summary, method_name, metric_key))
    return sum(vals) / len(vals) if vals else 0.0


def collect_results() -> Dict[str, Dict[str, float]]:
    baseline_path = RESULTS_DIR / "baseline_summary.json"
    llm_paths = sorted(MODEL_DIR.glob("llm_only_cached_summary_run_*.json"))
    hybrid_paths = sorted(MODEL_DIR.glob("hybrid_summary_run_*.json"))

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {baseline_path}")
    if not llm_paths:
        raise FileNotFoundError(f"No LLM-only run summaries found in: {MODEL_DIR}")
    if not hybrid_paths:
        raise FileNotFoundError(f"No Hybrid run summaries found in: {MODEL_DIR}")

    baseline_summary = load_json(baseline_path)

    return {
        "LLM-Only": {
            "feasibility": average_metric_across_runs(llm_paths, "LLM_ONLY", "feasibility"),
            "personalization": average_metric_across_runs(llm_paths, "LLM_ONLY", "personalization"),
            "utility": average_metric_across_runs(llm_paths, "LLM_ONLY", "utility"),
        },
        "Baseline": {
            "feasibility": get_metric_mean(baseline_summary, "BASELINE", "feasibility"),
            "personalization": get_metric_mean(baseline_summary, "BASELINE", "personalization"),
            "utility": get_metric_mean(baseline_summary, "BASELINE", "utility"),
        },
        "Hybrid": {
            "feasibility": average_metric_across_runs(hybrid_paths, "HYBRID", "feasibility"),
            "personalization": average_metric_across_runs(hybrid_paths, "HYBRID", "personalization"),
            "utility": average_metric_across_runs(hybrid_paths, "HYBRID", "utility"),
        },
    }


def make_bar_chart(
    results: Dict[str, Dict[str, float]],
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim_max: float = 0.35,
) -> None:
    methods = list(results.keys())
    values = [results[m][metric_key] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, values)
    plt.ylim(0, ylim_max)
    plt.ylabel(ylabel)
    plt.title(title)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    results = collect_results()

    print("Loaded aggregated results:")
    print(json.dumps(results, indent=2))

    make_bar_chart(
        results=results,
        metric_key="feasibility",
        title="Feasibility Comparison Across Planning Methods",
        ylabel="Feasibility Score",
        filename="figure_7_1_feasibility.png",
    )

    make_bar_chart(
        results=results,
        metric_key="personalization",
        title="Personalization Comparison Across Planning Methods",
        ylabel="Personalization Score",
        filename="figure_7_2_personalization.png",
    )

    make_bar_chart(
        results=results,
        metric_key="utility",
        title="Optimization Efficiency Comparison Across Planning Methods",
        ylabel="Optimization Score",
        filename="figure_7_3_optimization.png",
    )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()