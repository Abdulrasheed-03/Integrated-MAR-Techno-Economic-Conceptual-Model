"""
Financial engine tracking project value via cash flow and sensitivity calculations.
"""
import numpy as np


def calculate_annuity_factor(rate: float, periods: int) -> float:
    """Computes standard uniform series present worth annuity discount multipliers."""
    return (1.0 - (1.0 + rate) ** -periods) / rate


def compute_deterministic_npv(variables: dict, rate: float, include_externalities: bool, periods: int = 20) -> float:
    """Calculates project net present value based on internal and environmental cash flows."""
    benefits = sum(variables[k] for k in ["WQ", "EN", "ST", "AG", "CO2", "ECO"]) if include_externalities else 0.0
    annual_cash_flow = (variables["DSW"] + variables["AC"] - variables["OPEX"]) + benefits

    annuity_factor = calculate_annuity_factor(rate, periods)
    return -variables["CAPEX"] + (annual_cash_flow * annuity_factor)


def run_financial_monte_carlo(parameters: dict, iterations: int = 10000) -> tuple:
    """Simulates multi-variable pricing risk profiles using uniform sample distributions."""
    financial_npvs = []
    extended_npvs = []

    for _ in range(iterations):
        stochastic_sample = {k: np.random.uniform(*v) for k, v in parameters.items()}
        random_rate = np.random.uniform(0.03, 0.08)

        financial_npvs.append(compute_deterministic_npv(stochastic_sample, random_rate, include_externalities=False))
        extended_npvs.append(compute_deterministic_npv(stochastic_sample, random_rate, include_externalities=True))

    return np.array(financial_npvs), np.array(extended_npvs)


def run_tornado_sensitivity(parameters: dict, base_rate: float = 0.05) -> tuple:
    """Calculates local parameter swing variations relative to static baseline averages."""
    base_values = {k: np.mean(v) for k, v in parameters.items()}
    baseline_npv = compute_deterministic_npv(base_values, base_rate, include_externalities=True)

    sensitivity_bounds = {}
    for key, (low, high) in parameters.items():
        sample_low, sample_high = base_values.copy(), base_values.copy()

        sample_low[key] = low
        sample_high[key] = high

        sensitivity_bounds[key] = (
            compute_deterministic_npv(sample_low, base_rate, include_externalities=True) - baseline_npv,
            compute_deterministic_npv(sample_high, base_rate, include_externalities=True) - baseline_npv,
        )

    # Sort parameter deviations based on absolute impact range
    sorted_items = sorted(sensitivity_bounds.items(), key=lambda item: abs(item[1][1] - item[1][0]), reverse=True)
    labels, value_tuples = zip(*sorted_items)

    low_deviations = np.array([v[0] for v in value_tuples])
    high_deviations = np.array([v[1] for v in value_tuples])

    return labels, low_deviations, high_deviations
