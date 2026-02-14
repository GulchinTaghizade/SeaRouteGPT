# scripts/debug_llm_only_results.py
import json
from collections import Counter
from pathlib import Path

RAW = Path("results/llm_only_cached_raw_runs.jsonl")

def main():
    counts = Counter()
    total = 0
    feasible = 0
    non_no_valid = 0

    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1

        sid = r.get("selected_cruise_id", "")
        if sid and sid != "NO_VALID_CRUISE":
            non_no_valid += 1

        v = r.get("violations", []) or []
        counts.update(v)

        if r.get("feasibility", 0.0) >= 0.5:
            feasible += 1

    print("total:", total)
    print("non_NO_VALID:", non_no_valid)
    print("feasible:", feasible)
    print("\nTop violations:")
    for k, c in counts.most_common(10):
        print(f"  {k}: {c}")

if __name__ == "__main__":
    main()