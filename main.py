"""
Unified Execution Controller for the Integrated Subsurface Engineering,
Desalination and NPV Valuation Suite.

Usage:
    python main.py --mode volumetrics   # Module 1: Baseline OOIP/FWIP volumetrics
    python main.py --mode migration     # Module 2: 4D time-dependent fluid migration
    python main.py --mode salinity      # Module 3: Solute transport through filter layers
    python main.py --mode statistics    # Module 4: Sobol sensitivity + Monte Carlo uncertainty
    python main.py --mode financial     # Module 5: NPV distributions + Tornado analysis
"""
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Module 1 — Volumetrics
# ---------------------------------------------------------------------------

def run_volumetrics():
    """Calculates baseline deterministic OOIP and FWIP fluid distributions."""
    from src.core_model import calculate_volumes
    from src.visualizer import plot_fluid_volumes

    zones             = ["5", "4", "3", "2", "1"]
    thickness         = np.array([37, 43, 25, 22, 33])
    porosity          = np.array([18.5, 15.1, 12.3, 10.6, 17]) / 100
    saturation_water  = np.array([23, 51, 100, 100, 100]) / 100
    area              = 25000 * 25000   # m²

    volumes = calculate_volumes(area, thickness, porosity, saturation_water)
    print("[SUCCESS] Baseline Volumetrics Computed Successfully.")
    plot_fluid_volumes(zones, thickness, volumes["ooip"], volumes["fwip"])


# ---------------------------------------------------------------------------
# Module 2 — 4D Fluid Migration
# ---------------------------------------------------------------------------

def run_migration():
    """Models 4D fluid boundary migration over a 20-year production timeline."""
    from src.migration_engine import run_migration_simulation
    from src.visualizer import generate_4d_animation

    mD_to_m2 = 9.869e-16
    zones_data = {
        "Zone1": {"h": 33, "phi": 0.17,  "k": 227, "Sw": 1.0,  "Pc": 0},
        "Zone2": {"h": 22, "phi": 0.106, "k": 25,  "Sw": 1.0,  "Pc": 0},
        "Zone3": {"h": 25, "phi": 0.123, "k": 115, "Sw": 1.0,  "Pc": 0},
        "Zone4": {"h": 43, "phi": 0.151, "k": 133, "Sw": 0.51, "Pc": 5.1e3},
        "Zone5": {"h": 37, "phi": 0.185, "k": 68,  "Sw": 0.23, "Pc": 7.9e3},
    }
    for z in zones_data:
        zones_data[z]["k_m2"] = zones_data[z]["k"] * mD_to_m2

    print("[INFO] Initializing time-dependent 4D migration simulation...")
    n_steps    = 20 * 12                         # 240 monthly steps over 20 years
    time_years = np.arange(n_steps) * (1 / 12)  # fractional year per step

    history_results = run_migration_simulation(
        zones=zones_data,
        n_steps=n_steps,
        delta_P=(13.18e6 * 0.73),
        c_r=2.175e-9,
        c_w=6.554e-10,
        c_o=1.8125e-9,
    )

    generate_4d_animation(time_years, list(zones_data.keys()), history_results, n_steps)


# ---------------------------------------------------------------------------
# Module 3 — Salinity Transport
# ---------------------------------------------------------------------------

def run_salinity():
    """Simulates downward solute transport through engineered filtration layers."""
    from src.salinity_engine import calculate_adjusted_residual, compute_tds_profile
    from src.visualizer import render_static_salinity_plot

    initial_tds = 35000.0
    layers = [
        {"name": "A_topsoil",        "thickness_m": 0.5, "residual": 0.60},
        {"name": "B_clay_silt",       "thickness_m": 1.3, "residual": 0.50},
        {"name": "C_siltstone",       "thickness_m": 2.2, "residual": 0.60},
        {"name": "D_fine_sandstone",  "thickness_m": 2.5, "residual": 0.50},
        {"name": "E_med_coarse_sand", "thickness_m": 2.3, "residual": 0.65},
        {"name": "F_gravels",         "thickness_m": 3.0, "residual": 0.95},
    ]
    amendments = [
        {"biochar_vv": 0.05, "zeolite_vv": 0.0,  "ac_vv": 0.0},
        {"biochar_vv": 0.03, "zeolite_vv": 0.05, "ac_vv": 0.0},
        {"biochar_vv": 0.02, "zeolite_vv": 0.0,  "ac_vv": 0.02},
        {"biochar_vv": 0.02, "zeolite_vv": 0.0,  "ac_vv": 0.0},
        {"biochar_vv": 0.0,  "zeolite_vv": 0.05, "ac_vv": 0.05},
        {"biochar_vv": 0.0,  "zeolite_vv": 0.0,  "ac_vv": 0.0},
    ]

    base_res = [l["residual"] for l in layers]
    amend_res = [
        calculate_adjusted_residual(l["residual"], a["biochar_vv"], a["zeolite_vv"], a["ac_vv"])
        for l, a in zip(layers, amendments)
    ]

    tds_base  = compute_tds_profile(initial_tds, base_res)
    tds_amend = compute_tds_profile(initial_tds, amend_res)

    print(f"[SUCCESS] Hydrogeology Baseline Complete. Discharge Concentration: {tds_amend[-1]:.1f} mg/L")
    render_static_salinity_plot(layers, initial_tds, tds_base, tds_amend)


# ---------------------------------------------------------------------------
# Module 4 — Sensitivity & Uncertainty
# ---------------------------------------------------------------------------

def run_statistics():
    """Performs variance-based global sensitivity analysis and Monte Carlo uncertainty."""
    from src.statistics_engine import execute_sobol_analysis, execute_monte_carlo_uncertainty
    from src.visualizer import plot_statistical_publication_panel

    print("[INFO] Setting random seed for statistical consistency...")
    np.random.seed(42)

    layers = [
        "A_topsoil", "B_clay_silt", "C_siltstone",
        "D_fine_sandstone", "E_med_coarse_sand", "F_gravels",
    ]
    sobol_bounds = [
        [0.4, 0.8], [0.3, 0.7], [0.4, 0.8],
        [0.3, 0.7], [0.45, 0.85], [0.90, 0.99],
    ]
    triangular_distributions = {
        "A_topsoil":         (0.4,  0.6,  0.8),
        "B_clay_silt":       (0.3,  0.5,  0.7),
        "C_siltstone":       (0.4,  0.6,  0.8),
        "D_fine_sandstone":  (0.3,  0.5,  0.7),
        "E_med_coarse_sand": (0.45, 0.65, 0.85),
        "F_gravels":         (0.90, 0.95, 0.99),
    }
    depths = np.cumsum(np.insert(np.array([0.5, 1.3, 2.2, 2.5, 2.3, 3.0]), 0, 0))

    print("[INFO] Computing Sobol Indices (Saltelli Sampling)...")
    sobol_results = execute_sobol_analysis(layers, sobol_bounds, sample_size=2000)

    print("[INFO] Launching 15,000 Monte Carlo Simulation runs...")
    monte_carlo_matrix = execute_monte_carlo_uncertainty(layers, triangular_distributions, iterations=15000)

    print("[INFO] Outputting publication figures...")
    plot_statistical_publication_panel(sobol_results, monte_carlo_matrix, depths)


# ---------------------------------------------------------------------------
# Module 5 — Financial Valuation
# ---------------------------------------------------------------------------

def run_financial():
    """Evaluates project risk profiles using NPV distributions and Tornado chart sensitivity."""
    from src.financial_engine import run_financial_monte_carlo, run_tornado_sensitivity
    from src.visualizer import plot_financial_valuation_panel

    print("[INFO] Launching economic simulation processing loop...")
    np.random.seed(42)

    financial_parameters = {
        "CAPEX": [2.01, 6.43], "OPEX": [0.06, 0.74],
        "DSW":   [3.89, 8.72], "AC":   [9.40, 64.09],
        "WQ":    [0.34, 0.85], "EN":   [0.77, 3.82],
        "ST":    [1.22, 6.11], "AG":   [0.51, 3.85],
        "CO2":   [0.64, 4.49], "ECO":  [0.15, 0.51],
    }

    fin_npvs, ext_npvs = run_financial_monte_carlo(financial_parameters, iterations=10000)
    tornado_results    = run_tornado_sensitivity(financial_parameters, base_rate=0.05)

    plot_financial_valuation_panel(fin_npvs, ext_npvs, tornado_results)
    print("[SUCCESS] Financial validation execution panel compiled successfully.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

_RUNNERS = {
    "volumetrics": run_volumetrics,
    "migration":   run_migration,
    "salinity":    run_salinity,
    "statistics":  run_statistics,
    "financial":   run_financial,
}

def run_all():
    print("=========================================")
    print("Running Module 1: Volumetrics")
    run_volumetrics()
    print("\n=========================================")
    print("Running Module 2: Migration")
    run_migration()
    print("\n=========================================")
    print("Running Module 3: Salinity")
    run_salinity()
    print("\n=========================================")
    print("Running Module 4: Statistics")
    run_statistics()
    print("\n=========================================")
    print("Running Module 5: Financial")
    run_financial()

_RUNNERS["all"] = run_all


def main():
    parser = argparse.ArgumentParser(
        description="Unified Engineering Suite Simulation Package Gateway.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="financial",
        choices=list(_RUNNERS.keys()),
        help="Selects which simulation module to run (default: financial).",
    )
    args = parser.parse_args()

    print(f"[INFO] Running mode: {args.mode}")
    _RUNNERS[args.mode]()


if __name__ == "__main__":
    main()
