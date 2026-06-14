# Integrated Simulation Project (Yerima et al.)
**Subsurface Engineering, Desalination, and Financial Valuation Suite**

This repository bundles five specialized simulation modules into a single, cohesive computational framework designed for research verification and analysis. It integrates physical volumetric models, 4D fluid migration tracking, hydrogeological salinity transport, robust statistical uncertainty quantification, and financial net present value (NPV) distributions.

---

## 🚀 Getting Started

### Prerequisites
- **Python:** Version 3.10 or higher is required.

### Installation & Environment Setup
We highly recommend running this project inside a virtual environment to ensure dependency isolation.

```bash
# 1. Initialize the virtual environment
python -m venv environment_layer

# 2. Activate the environment
# On macOS/Linux:
source environment_layer/bin/activate
# On Windows:
environment_layer\Scripts\activate

# 3. Install required software dependencies
pip install -r requirements.txt
```

---

## 💻 Usage & Execution

The project uses a unified Command Line Interface (CLI) gateway via `main.py`. You can run individual modules or execute the entire pipeline sequentially. 

**All generated figures and interactive HTML animations will be automatically saved to the `results/` folder.**

```bash
# Run the entire simulation suite (Modules 1 through 5 sequentially)
python main.py --mode all

# Or, run individual simulation modules:
python main.py --mode volumetrics
python main.py --mode migration
python main.py --mode salinity
python main.py --mode statistics
python main.py --mode financial
```

---

## 📦 Module Architecture

The core physics, statistics, and financial logic is contained within the `src/` package:

1. **Volumetrics (`core_model.py`)** 
   Calculates baseline deterministic Original Oil In Place (OOIP) and Free Water In Place (FWIP) fluid distributions across varied physical reservoir zones.
   
2. **Fluid Migration (`migration_engine.py`)**
   Models 4D fluid boundary migration and dynamically calculates multi-zone fluid tracking over a 20-year simulated production timeline.

3. **Salinity Transport (`salinity_engine.py`)** 
   Simulates downward solute (Total Dissolved Solids) transport and attenuation through engineered filtration layers using physical residuals.

4. **Sensitivity & Uncertainty (`statistics_engine.py`)** 
   Performs variance-based global sensitivity analysis using Sobol indices alongside Monte Carlo loops to quantify stochastic uncertainty envelopes.

5. **Financial Valuation (`financial_engine.py`)** 
   Evaluates aggregate project risk profiles using Monte Carlo NPV distributions and Tornado chart local sensitivity analyses against externalities.

---

## 📝 Academic Integrity & Refactoring Disclosure
All mathematical algorithms, engineering physics equations, economic discount models, and data distribution thresholds were conceptualized and manually coded by the primary author. 

AI refactoring tools were utilized strictly to translate the initial flat scripts into a clean, modular repository architecture (the `/src` package), deduplicate plotting utilities, and construct a unified CLI gateway (`main.py`) to enforce reproducibility and structural best practices. No physical equations or logic were fundamentally altered by AI generation.
