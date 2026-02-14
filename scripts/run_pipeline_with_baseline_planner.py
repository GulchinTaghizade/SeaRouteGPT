from models.baseline.baseline_constraint_extractor import ConstraintExtractor
from validation.constraint_validator import ConstraintValidator
from models.baseline.rule_based_planner import RuleBasedPlanner
import json
from data.synthetic.load_requests import load_user_requests


from dotenv import load_dotenv
load_dotenv()


def load_cached_cruises():
    """
    Loads a frozen snapshot of cruise data.
    """
    with open("data/raw/cruises.json", "r") as f:
        return json.load(f)["data"]



# Baseline pipeline with constraint extraction, validation, and rule-based planning

def main():
    # 1️⃣ Load data
    user_requests = load_user_requests()
    cruises = load_cached_cruises()

    extractor = ConstraintExtractor()
    validator = ConstraintValidator(cruises)
    planner = RuleBasedPlanner()


    for req in user_requests:
        print(f"\n🔹 Processing {req['request_id']}")
        print(f"   📝 Request: {req['text']}")

        # 2️⃣ Extract constraints
        constraints = extractor.extract_constraints(
            req["text"],
            request_id=req["request_id"]
        )

        # Print extracted constraints
        hard_constraints = constraints.get('hard_constraints', {})
        print(f"   📋 Hard constraints: {hard_constraints}")

        soft_preferences = constraints.get('soft_preferences', {})
        if any(v is not None for v in soft_preferences.values()):
            print(f"   🎯 Soft preferences: {soft_preferences}")

        # 3️⃣ VALIDATE CONSTRAINTS
        validation = validator.validate(constraints)

        if not validation["is_feasible"]:
            print("⚠️ No feasible cruises found")
            continue

        feasible_cruises = validation["feasible_cruises"]

        # 4️⃣ Plan itinerary
        itinerary = planner.plan(feasible_cruises, constraints)

        if itinerary is None:
            print("⚠️ Planner failed to select cruise")
            continue

        print("✅ Selected cruise:", itinerary["cruiseId"])

if __name__ == "__main__":
    main()