# SeaRouteGPT System Architecture

SeaRouteGPT is a hybrid cruise itinerary planning system that combines Large Language Models (LLMs) for natural-language understanding with Mixed Integer Linear Programming (MILP) for constraint-based optimization. The system is designed to transform unstructured user requests into feasible and optimized cruise itineraries in a modular, reproducible manner.

## High-Level Pipeline

User Request (Natural Language)
        ↓
LLM Constraint Extraction
        ↓
Structured Constraint JSON
        ↓
Validation & Ambiguity Handling
        ↓
MILP Optimization Solver
        ↓
Optimized Cruise Itinerary

## Component Description

### 1. User Request
The system accepts a natural-language cruise planning request containing a mix of hard constraints (e.g., budget, dates, destinations) and soft preferences (e.g., cruise line, preferred ports).

### 2. LLM Constraint Extraction
An LLM processes the user request and extracts a structured representation of constraints using a predefined JSON schema. The model is instructed to:
- Separate hard constraints from soft preferences
- Avoid inventing information not present in the request
- Mark ambiguous preferences explicitly

The output of this stage is a strictly structured JSON object.

### 3. Validation and Ambiguity Handling
The extracted constraint JSON is validated against a schema to ensure correctness and completeness. Ambiguous or weakly specified preferences are handled using explicit rules:
- Mapped to soft constraints with low weight, or
- Discarded with a logged warning if no safe mapping exists

Invalid or malformed outputs are rejected before optimization.

### 4. MILP Optimization Solver
The validated constraints are passed to a Mixed Integer Linear Programming (MILP) solver implemented using Google OR-Tools. The solver:
- Enforces all hard feasibility constraints
- Optimizes a utility-based objective balancing cost and schedule fit
- Produces a single feasible and optimized cruise itinerary

### 5. Output Itinerary
The final output is a structured itinerary containing the selected cruise and its schedule. This output is guaranteed to satisfy all hard constraints and is scored using predefined evaluation metrics.

## Design Principles
- Separation of concerns between language understanding and optimization
- API-agnostic constraint representation
- Deterministic optimization behavior
- Explicit failure handling and logging