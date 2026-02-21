import os
import json
import time
from dotenv import load_dotenv

from data.synthetic.load_requests import load_user_requests
from models.llm.llm_only_planner import LLMPlanner

load_dotenv()


def load_cached_cruises():
    """Loads a frozen snapshot of cruise data."""
    with open("data/raw/cruises.json", "r") as f:
        return json.load(f)["data"]


def main():
    # 0) API key sanity check
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in .env or export it in your shell.")

    # 1) Load data
    user_requests = load_user_requests()
    cruises = load_cached_cruises()

    planner = LLMPlanner(api_key=api_key)


    cache_hits = 0
    api_calls = 0

    for i, req in enumerate(user_requests, start=1):
        rid = req.get("request_id")
        text = req.get("text", "")

        print(f"\n🔹 [{i}/{len(user_requests)}] Processing {rid}")

        try:
            result = planner.plan(
                cruises=cruises,
                user_request=text,
                request_id=rid
            )
            from_cache = False
            try:
                from_cache = bool(result.get("from_cache", False))
            except Exception:
                pass

            if from_cache:
                cache_hits += 1
                print("  ✅ LLM completed (cache hit)")
            else:
                api_calls += 1
                print("  ✅ LLM completed (API call)")

            # Helpful status
            print(f"  ↳ valid={result.get('is_valid')} selected={result.get('selected_cruise_id')}")

        except Exception as e:
            print("  ❌ LLM failed:", repr(e))

        # Rate-limit safety: sleep only when we likely called the API
        # If you add from_cache, this becomes accurate.
        if api_calls > 0:
            time.sleep(15)

    print("\n==== DONE ====")
    print(f"API calls: {api_calls}")
    print(f"Cache hits: {cache_hits}")


if __name__ == "__main__":
    main()