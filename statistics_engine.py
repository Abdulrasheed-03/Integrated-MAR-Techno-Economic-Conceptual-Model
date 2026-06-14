"""
Statistical engine executing global sensitivity analyses and stochastic uncertainty calculations.
"""
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol

# Relative import — salinity_engine lives in the same package
from .salinity_engine import compute_tds_profile


def execute_sobol_analysis(layers: list, bounds: list, sample_size: int = 2000) -> pd.DataFrame:
    """Executes variance-based global sensitivity analysis using Sobol indices."""
    problem = {
        "num_vars": len(layers),
        "names":    layers,
        "bounds":   bounds,
    }

    # Generate parameters using Saltelli sampling scheme
    param_values = saltelli.sample(problem, sample_size, calc_second_order=False)

    # Evaluate model rows using the sequential TDS reduction formula
    results = []
    for row in param_values:
        profile = compute_tds_profile(initial_tds=35000.0, layer_residuals=row)
        results.append(profile[-1])

    y_evaluations = np.array(results)
    sensitivity_indices = sobol.analyze(problem, y_evaluations, calc_second_order=False)

    analysis_df = pd.DataFrame({
        "Layer": layers,
        "S1":    sensitivity_indices["S1"],
        "ST":    sensitivity_indices["ST"],
    }).sort_values("ST")

    return analysis_df


def execute_monte_carlo_uncertainty(layers: list, triangular_bounds: dict, iterations: int = 15000) -> np.ndarray:
    """Runs a Monte Carlo loop to model parameter variations along a profile depth."""
    all_profiles = []

    for _ in range(iterations):
        stochastic_residuals = []
        for layer_name in layers:
            # Draw random residual value from its unique triangular distribution
            random_sample = np.random.triangular(*triangular_bounds[layer_name])
            stochastic_residuals.append(random_sample)

        profile_run = compute_tds_profile(initial_tds=35000.0, layer_residuals=stochastic_residuals)
        all_profiles.append(profile_run)

    return np.array(all_profiles)
