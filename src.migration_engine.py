"""
Numerical migration engine tracking water saturation updates over time.
"""
import numpy as np

def calculate_water_outflow(zone: dict, delta_P: float, c_r: float, c_w: float, c_o: float, dt_frac: float) -> float:
    """Computes the stochastically constrained water outflow volume for a specific zone."""
    effective_pressure = max(0.0, delta_P - zone["Pc"])
    if effective_pressure <= 0.0:
        return 0.0
        
    phi = zone["phi"]
    total_compressibility = c_r + (phi * c_w) + ((1.0 - phi) * c_o)
    
    delta_v = total_compressibility * zone["Vpore"] * effective_pressure * dt_frac
    return min(delta_v, zone["Vw"])

def run_migration_simulation(zones: dict, n_steps: int, delta_P: float, c_r: float, c_w: float, c_o: float) -> dict:
    """Executes the time-step loop across all layers and records fluid history."""
    # Initialize dynamic trackers inside code logic to keep base zones clean
    for z in zones:
        zones[z]["Vw_init"] = zones[z]["phi"] * zones[z]["Sw"] * zones[z]["h"] * (2500 * 2500)
        zones[z]["Vpore"] = zones[z]["phi"] * zones[z]["h"] * (2500 * 2500)
        zones[z]["Vw"] = zones[z]["Vw_init"]

    history = {z: [] for z in zones}

    for step in range(n_steps):
        v_out = {}
        for z in ["Zone1", "Zone2", "Zone3"]:
            v_out[z] = calculate_water_outflow(zones[z], delta_P, c_r, c_w, c_o, dt_frac=1.0/n_steps)
            zones[z]["Vw"] -= v_out[z]

        total_influx = sum(v_out.values())
        k4, k5 = zones["Zone4"]["k_m2"], zones["Zone5"]["k_m2"]
        
        zones["Zone4"]["Vw"] += total_influx * k4 / (k4 + k5)
        zones["Zone5"]["Vw"] += total_influx * k5 / (k4 + k5)

        for z in zones:
            history[z].append(zones[z]["Vw"])
            
    return {z: np.array(history[z]) for z in zones}
