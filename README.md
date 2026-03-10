# SeaRouteGPT

SeaRouteGPT is a hybrid AI-powered cruise itinerary planning system that combines Large Language Models (LLMs) for natural-language understanding with Mixed Integer Linear Programming (MILP) for constraint-based optimization. The system transforms unstructured user requests into feasible and optimized cruise itineraries. This project was developed as part of a graduate research thesis on hybrid AI planning for travel itinerary recommendation.


## Features

- **Hybrid Planning**: Integrates LLM-based constraint extraction with MILP optimization for robust itinerary planning
- **Multiple Planners**: Supports baseline rule-based, LLM-only, and hybrid approaches
- **Web UI**: Streamlit-based interface for interactive cruise planning
- **Evaluation Framework**: Comprehensive metrics and experiment aggregation tools
- **API Integration**: Fetches real-time cruise data from external APIs
- **Caching**: Efficient data caching for repeated experiments

## Architecture

The system follows a modular pipeline:

1. **User Request** → Natural language cruise planning query
2. **LLM Constraint Extraction** → Structured JSON constraints
3. **Validation** → Schema validation and ambiguity handling
4. **MILP Optimization** → Feasible itinerary generation
5. **Output** → Optimized cruise itinerary

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SeaRouteGPT.git
   cd SeaRouteGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with your API keys:
   ```
   RAPIDAPI_KEY=your_rapidapi_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

### Running Experiments

Run baseline experiments:
```bash
python scripts/run_experiments.py
```

Run LLM-only experiments:
```bash
python scripts/run_experiments_llm_only_cached.py
```

Run hybrid experiments:
```bash
python scripts/run_experiments_hybrid_cached.py
```

### Web Interface

Launch the Streamlit UI:
```bash
streamlit run UI/app.py
```

### Individual Pipeline Runs

- Baseline planner: `python scripts/run_pipeline_with_baseline_planner.py`
- LLM planner: `python scripts/run_pipeline_with_LLM_planner.py`
- MILP planner: `python scripts/run_pipeline_with_MILP_planner.py`

## Data

- **Cruise Catalog**: Real cruise data stored in `data/raw/cruises.json`
- **Synthetic Requests**: Generated user queries in `data/synthetic/user_requests.json`
- **Processed Data**: Cached and processed data in `data/processed/`

## Results

Experiment results are stored in the `results/` directory, including:
- Raw run data (JSONL files)
- Summary statistics
- Performance metrics
- Visualization figures

## Project Structure

```
SeaRouteGPT/
├── api/                    # API providers
├── config/                 # Configuration files
├── data/                   # Data storage and processing
├── docs/                   # Documentation
├── evaluation/             # Metrics and aggregation
├── models/                 # Planning models (baseline, hybrid, LLM)
├── prompts/                # LLM prompt templates
├── results/                # Experiment outputs
├── scripts/                # Experiment and utility scripts
├── solvers/                # MILP solver implementation
├── UI/                     # Streamlit web interface
└── validation/             # Constraint validation
```

## Dependencies

- **Google Generative AI**: For LLM constraint extraction
- **OR-Tools**: For MILP optimization
- **Streamlit**: Web UI framework
- **Pandas/NumPy**: Data processing
- **Requests**: API calls

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SeaRouteGPT in your research, please cite:

```
@software{SeaRouteGPT,
  title={SeaRouteGPT: AI-Powered Cruise Travel Planning},
  author={Gulchin Taghizade},
  year={2026}
}
```
