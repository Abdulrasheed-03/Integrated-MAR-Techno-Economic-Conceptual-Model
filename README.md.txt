# Integrated Subsurface Engineering, Desalination and NPV Valuation Suite

# https://doi.org/10.5281/zenodo.20679475

This repository bundles our 5 simulation modules into a single, cohesive conceptual framework designed for external verification.

## Getting Started

Ensure you have Python 3.10+ installed, then set up your environment:

```bash
# 1. Initialize environment
python -m venv environment_layer
source environment_layer/bin/activate  # Windows: environment_layer\Scripts\activate

# 2. Install required software dependencies
pip install -r requirements.txt

# 3. Run specific simulation verification suites
python main.py --mode volumetrics
python main.py --mode migration
python main.py --mode salinity
python main.py --mode statistics
python main.py --mode financial
```

## Module Descriptions
1. **Volumetrics (`core_model.py`)**: Calculates baseline deterministic OOIP and FWIP fluid distributions.
2. **Fluid Migration (`migration_engine.py`)**: Models 4D fluid boundary migration over a 20-year production timeline.
3. **Salinity Transport (`salinity_engine.py`)**: Simulates downward solute transport through engineered filtration layers.
4. **Sensitivity & Uncertainty (`statistics_engine.py`)**: Performs variance-based global sensitivity analysis using Sobol indices and Monte Carlo loops.
5. **Financial Valuation (`financial_engine.py`)**: Evaluates project risk profiles using NPV distributions and Tornado chart sensitivity analysis.

## Academic Integrity & AI Refactoring Disclosure
All mathematical algorithms, engineering physics equations, economic discount models, and data distribution thresholds were conceptualized and coded by the primary author. AI tools were used strictly to refactor the initial scripts into a clean, modular repository architecture (`/src`) to improve code readability and ensure reproducibility.
