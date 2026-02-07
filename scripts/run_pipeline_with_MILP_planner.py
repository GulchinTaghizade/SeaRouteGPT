from models.llm.constraint_extractor import ConstraintExtractor
import json
from data.synthetic.load_requests import load_user_requests

from solvers.milp_solver import MILPSolver
from solvers.objective import minimize_price


from dotenv import load_dotenv
load_dotenv()


def load_cached_cruises():
    """
    Loads a frozen snapshot of cruise data.
    """
    with open("data/raw/cruise_snapshot_20260206_192232.json", "r") as f:
        return json.load(f)["data"]



def main():
    print("\n🚢 SeaRouteGPT — MILP Optimization Pipeline\n")

    # 1️⃣ Load data
    cruises = load_cached_cruises()
    user_requests = load_user_requests()

    print(f"✔ Loaded {len(cruises)} cruises")
    print(f"✔ Loaded {len(user_requests)} user requests\n")

    # 2️⃣ Initialize components
    extractor = ConstraintExtractor()

    # 3️⃣ Run pipeline
    for req in user_requests:
        print(f"🔹 Processing {req['request_id']}")
        print(f"   📝 Request: {req['text']}")

        # 4️⃣ Extract structured constraints via LLM
        constraints = extractor.extract_constraints(
            req["text"],
            request_id=req["request_id"]
        )

        print(f"   📋 Hard constraints: {constraints.get('hard_constraints', {})}")

        # 5️⃣ Solve MILP
        solver = MILPSolver()
        best_cruise = solver.solve(
            cruises=cruises,
            constraints=constraints,
            objective_fn=minimize_price
        )

        if best_cruise is None:
            print("   ❌ No feasible cruise found\n")
            continue

        # 6️⃣ Output result
        print("   ✅ Selected cruise:")
        print(f"      🆔 ID: {best_cruise.get('cruiseId')}")
        print(f"      📍 Destinations: {best_cruise.get('itineraryDestinations')}")
        print(f"      📆 Duration: {best_cruise.get('duration')} days")
        print(f"      💰 Price: ${best_cruise.get('roomPriceWithTaxesFees')}")
        print()

    print("🏁 MILP pipeline completed.\n")

if __name__ == "__main__":
    main()