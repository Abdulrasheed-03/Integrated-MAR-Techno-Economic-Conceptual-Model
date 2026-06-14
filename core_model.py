"""
Core reservoir engineering volumetric calculations.
"""
import numpy as np


def calculate_volumes(area: float, thickness: np.ndarray, porosity: np.ndarray, sw: np.ndarray) -> dict:
    """
    Calculates Bulk Volume, Pore Volume, OOIP, and FWIP for reservoir zones.

    Parameters:
        area (float): Reservoir area in square meters.
        thickness (np.ndarray): Zone thicknesses in meters.
        porosity (np.ndarray): Zone porosities as fractions.
        sw (np.ndarray): Zone water saturations as fractions.
    """
    so = 1.0 - sw

    v_bulk = area * thickness
    v_pore = v_bulk * porosity
    ooip = v_pore * so
    fwip = v_pore * sw

    return {
        "v_bulk": v_bulk,
        "v_pore": v_pore,
        "ooip": ooip,
        "fwip": fwip,
    }
