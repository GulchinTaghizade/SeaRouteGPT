from models.llm.constraint_extractor import ConstraintExtractor

extractor = ConstraintExtractor()
constraints = extractor.extract_constraints(
    "I want a 10-14 day Alaska cruise in May under $5000 for two people",
    request_id="req_test"
)

print(constraints)