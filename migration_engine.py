"""
Numerical migration engine tracking water saturation updates over time.
"""
import numpy as np


def calculate_water_outflow(zone: dict, delta_P: float, c_r: float, c_w: float, c_o: float, dt_frac: float) -> float:
    """Computes the fractional water outflow volume for a specific zone per time step.

    dt_frac is the fraction of the total simulation steps (1 / n_steps). Multiplying
    by dt_frac distributes the total compressibility-driven volume change evenly across
    all time steps so that the cumulative depletion over the full simulation equals
    the theoretical maximum (c_t * Vpore * delta_P).
    """
    effective_pressure = max(0.0, delta_P - zone["Pc"])
    if effective_pressure <= 0.0:
        return 0.0

    phi = zone["phi"]
    total_compressibility = c_r + (phi * c_w) + ((1.0 - phi) * c_o)

    delta_v = total_compressibility * zone["Vpore"] * effective_pressure * dt_frac
    return min(delta_v, zone["Vw"])


def run_migration_simulation(
    zones: dict,
    n_steps: int,
    delta_P: float,
    c_r: float,
    c_w: float,
    c_o: float,
    drain_zones: list = None,
    receive_zones: list = None,
) -> dict:
    """Executes the time-step loop across all layers and records fluid history.

    Parameters:
        zones (dict): Zone property dictionaries (must include 'phi', 'Sw', 'h', 'k_m2', 'Pc').
        n_steps (int): Total number of time steps.
        delta_P (float): Total pressure differential driving fluid migration (Pa).
        c_r (float): Rock compressibility (Pa⁻¹).
        c_w (float): Water compressibility (Pa⁻¹).
        c_o (float): Oil compressibility (Pa⁻¹).
        drain_zones (list, optional): Keys of zones that lose water. Defaults to first 3 zones.
        receive_zones (list, optional): Keys of zones that gain water. Defaults to remaining zones.
    """
    zone_keys = list(zones.keys())
    if drain_zones is None:
        drain_zones = zone_keys[:3]
    if receive_zones is None:
        receive_zones = zone_keys[3:]

    # Initialise dynamic volume trackers without mutating the caller's base zones
    area = 2500 * 2500  # m² — shared reservoir cell area
    for z in zones:
        zones[z]["Vw_init"] = zones[z]["phi"] * zones[z]["Sw"] * zones[z]["h"] * area
        zones[z]["Vpore"]   = zones[z]["phi"] * zones[z]["h"] * area
        zones[z]["Vw"]      = zones[z]["Vw_init"]

    history = {z: [] for z in zones}
    dt_frac = 1.0 / n_steps  # fractional depletion per step

    for _ in range(n_steps):
        # Compute outflow from each draining zone
        v_out = {}
        for z in drain_zones:
            v_out[z] = calculate_water_outflow(zones[z], delta_P, c_r, c_w, c_o, dt_frac)
            zones[z]["Vw"] -= v_out[z]

        # Distribute total influx to receiving zones weighted by permeability
        total_influx = sum(v_out.values())
        k_values = {z: zones[z]["k_m2"] for z in receive_zones}
        k_total = sum(k_values.values())
        for z in receive_zones:
            zones[z]["Vw"] += total_influx * k_values[z] / k_total

        for z in zones:
            history[z].append(zones[z]["Vw"])

    return {z: np.array(history[z]) for z in zones}
