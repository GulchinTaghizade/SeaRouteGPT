from data.synthetic.load_requests import load_user_requests
from models.llm.constraint_extractor import ConstraintExtractor
import json

def main():
    extractor = ConstraintExtractor()  
    requests = load_user_requests()

    results = []

    for req in requests:
        constraints = extractor.extract_constraints(
            user_request=req["text"],
            request_id=req["request_id"]
        )
        results.append({
            "request_id": req["request_id"],
            "text": req["text"],
            "constraints": constraints
        })

    # Save results for inspection
    with open("data/processed/extracted_constraints.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Extracted constraints for {len(results)} requests.")

if __name__ == "__main__":
    main()