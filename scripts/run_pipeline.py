from data.synthetic.load_requests import load_user_requests
from models.llm.constraint_extractor import ConstraintExtractor
from validation.constraint_validator import ConstraintValidator
from baselines.rule_based_planner import RuleBasedPlanner
import json


def load_cached_cruises():
    with open("data/raw/cruise_snapshot_20260201_222236.json", "r") as f:
        return json.load(f)["data"]


def main():
    # 1️⃣ Load data
    user_requests = load_user_requests()
    cruises = load_cached_cruises()

    extractor = ConstraintExtractor()
    validator = ConstraintValidator(cruises)
    planner = RuleBasedPlanner()

    for req in user_requests:
        print(f"\n🔹 Processing {req['request_id']}")

        # 2️⃣ Extract constraints
        constraints = extractor.extract_constraints(
            req["text"],
            request_id=req["request_id"]
        )
        print(f"  📋 Extracted constraints: {constraints.get('hard_constraints', {})}")

        # 3️⃣ VALIDATE CONSTRAINTS  
        validation = validator.validate(constraints)

        print(f"  🔍 Validated {len(cruises)} total cruises")
        print(f"  ✔️  Found {validation['feasible_count']} feasible cruises")

        if not validation["is_feasible"]:
            print("  ⚠️ No feasible cruises found")
            continue

        feasible_cruises = validation["feasible_cruises"]

        # 4️⃣ Plan itinerary
        itinerary = planner.plan(feasible_cruises, constraints)

        if itinerary is None:
            print("  ⚠️ Planner failed to select cruise")
            continue

        print(f"  ✅ Selected cruise: {itinerary['cruiseId']}")


if __name__ == "__main__":
    main()