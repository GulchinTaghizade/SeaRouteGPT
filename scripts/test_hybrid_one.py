import json
import os
from dotenv import load_dotenv

from models.hybrid.hybrid_planner import HybridSolver

load_dotenv()

def load_cached_cruises():
    with open("data/raw/cruise_snapshot_20260206_192232.json", "r") as f:
        return json.load(f)["data"]

def main():
    cruises = load_cached_cruises()

    # ✅ Pick ONE request to test
    user_request = "I want a 10–14 day Alaska cruise departing on June 1, 2026 for two people under $5000."
    request_id = "debug_req_001"

    solver = HybridSolver(api_key=os.getenv("GOOGLE_API_KEY"))

    result = solver.solve(
        user_request=user_request,
        cruises=cruises,
        alpha=0.6,
        beta=0.4,
        request_id=request_id,
        time_limit_seconds=10
    )

    print("\n===== HYBRID RESULT =====")
    print("STATUS:", result["status"])
    print("REQUEST ID:", result["request_id"])
    print("\nHARD CONSTRAINTS:", result.get("constraints_extracted"))
    print("\nSOFT PREFS:", result.get("preferences_extracted"))

    if result["status"] == "success":
        c = result["selected_cruise"]
        print("\nSELECTED CRUISE:")
        print("  cruiseId:", c.get("cruiseId"))
        print("  name:", c.get("cruiseName"))
        print("  departureDate:", c.get("departureDate"))
        print("  duration:", c.get("duration"))
        print("  price:", c.get("roomPriceWithTaxesFees"))
        print("  destinations:", c.get("itineraryDestinations"))

if __name__ == "__main__":
    main()