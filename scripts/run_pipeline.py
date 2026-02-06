from models.llm.constraint_extractor import ConstraintExtractor
from validation.constraint_validator import ConstraintValidator
from baselines.rule_based_planner import RuleBasedPlanner
import os
import json
import time
from data.synthetic.load_requests import load_user_requests
from models.llm.llm_only_planner import LLMPlanner

from dotenv import load_dotenv
load_dotenv()

def load_cached_cruises():
    with open("data/raw/cruise_snapshot_20260201_222236.json", "r") as f:
        return json.load(f)["data"]
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

# def main():
#     # 1️⃣ Load data
#     user_requests = load_user_requests()
#     cruises = load_cached_cruises()
#
#     extractor = ConstraintExtractor()
#     validator = ConstraintValidator(cruises)
#     planner = RuleBasedPlanner()
#
#
#     for req in user_requests[:1]:
#         print(f"\n🔹 Processing {req['request_id']}")
#
#         # 2️⃣ Extract constraints
#         constraints = extractor.extract_constraints(
#             req["text"],
#             request_id=req["request_id"]
#         )
#
#         # 3️⃣ VALIDATE CONSTRAINTS
#         validation = validator.validate(constraints)
#
#         if not validation["is_feasible"]:
#             print("⚠️ No feasible cruises found")
#             continue
#
#         feasible_cruises = validation["feasible_cruises"]
#
#         # 4️⃣ Plan itinerary
#         itinerary = planner.plan(feasible_cruises, constraints)
#
#         if itinerary is None:
#             print("⚠️ Planner failed to select cruise")
#             continue
#
#         print("✅ Selected cruise:", itinerary["cruiseId"])



if __name__ == "__main__":
    main()