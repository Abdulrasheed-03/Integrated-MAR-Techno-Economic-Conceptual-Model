"""
Hydrogeological engine tracking salinity attenuation and soil amendment absorption.
"""
import numpy as np

def calculate_adjusted_residual(base_residual: float, biochar: float, zeolite: float, ac: float) -> float:
    """Computes the adjusted solute retention factor based on soil amendment fractions."""
    res = base_residual
    res -= biochar * (0.10 / 0.05)
    res -= zeolite * (0.20 / 0.05)
    res -= ac * (0.10 / 0.05)
    return max(res, 0.05)

def compute_tds_profile(initial_tds: float, layer_residuals: list) -> list:
    """Calculates downward sequential Total Dissolved Solids concentrations per boundary."""
    current_concentration = initial_tds
    tds_profile = [initial_tds]
    for r in layer_residuals:
        current_concentration *= r
        tds_profile.append(current_concentration)
    return tds_profile
