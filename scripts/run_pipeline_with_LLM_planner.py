import os
import json
import time
from data.synthetic.load_requests import load_user_requests
from models.llm.llm_only_planner import LLMPlanner



from dotenv import load_dotenv
load_dotenv()


def load_cached_cruises():
    """
    Loads a frozen snapshot of cruise data.
    """
    with open("data/raw/cruises.json", "r") as f:
        return json.load(f)["data"]

#LLM planner pipeline
def main():
    # 1️⃣ Load data
    user_requests = load_user_requests()
    cruises = load_cached_cruises()

    planner = LLMPlanner(api_key=os.getenv("GOOGLE_API_KEY"))

    for i, req in enumerate(user_requests, start=1):
        print(f"\n🔹 [{i}/{len(user_requests)}] Processing {req['request_id']}")

        # 2️⃣ Plan itinerary
        try:
            result = planner.plan(
                cruises=cruises,
                user_request=req["text"],
                request_id=req["request_id"]
            )
            print("  ✅ LLM completed")

        except Exception as e:
            print("  ❌ LLM failed:", e)

        # ⏱️ IMPORTANT: rate-limit safety
        time.sleep(15)


if __name__ == "__main__":
    main()