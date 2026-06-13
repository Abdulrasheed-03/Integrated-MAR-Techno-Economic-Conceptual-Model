"""
Plotting utilities for reservoir data presentation.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_fluid_volumes(zones: list, thickness: np.ndarray, ooip: np.ndarray, fwip: np.ndarray):
    """Generates a professional stacked bar chart of oil and water volumes."""
    # Normalize bar width by thickness
    widths = thickness / np.max(thickness) * 0.8
    positions = np.cumsum(np.insert(widths[:-1], 0, 0)) + widths / 2
    labels = [f"Zone {z}\n({t:.0f} m)" for z, t in zip(zones, thickness)]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Stacked bar generation (converted to Mm³)
    ax.bar(positions, fwip/1e6, color='skyblue', label='Water', width=widths)
    ax.bar(positions, ooip/1e6, bottom=fwip/1e6, color='green', label='Oil', width=widths)

    # Label data annotations
    for i in range(len(zones)):
        if fwip[i] > 0:
            ax.text(positions[i], fwip[i]/2/1e6, f"{fwip[i]/1e6:.1f}", ha='center', va='center', color='black', fontsize=10)
        if ooip[i] > 0:
            ax.text(positions[i], (fwip[i]+ooip[i]/2)/1e6, f"{ooip[i]/1e6:.1f}", ha='center', va='center', color='white', fontsize=10)

    ax.set_xlabel('Reservoir Zone', fontsize=12)
    ax.set_ylabel('Volume (Mm³)', fontsize=12)
    ax.set_title('Initial Oil and Water in Place per Zone', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()


"""
Plotting utilities for reservoir data presentation (Matplotlib & Plotly).
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def generate_4d_animation(time_years: np.ndarray, zones_order: list, history_arr: dict, n_steps: int):
    """Generates an interactive HTML 4D surface plot animation for external presentation."""
    n_zones = len(zones_order)
    water_change = np.zeros((n_zones, n_steps))
    
    for i, z in enumerate(zones_order):
        water_change[i, :] = history_arr[z] - history_arr[z][0]

    cum_influx_4 = np.cumsum(np.maximum(0, np.gradient(water_change[3, :])))
    cum_influx_5 = np.cumsum(np.maximum(0, np.gradient(water_change[4, :])))
    
    pressure_norm = np.linspace(13.18, 9.2, n_steps)
    pressure_norm = (pressure_norm - 9.2) / (13.18 - 9.2) * np.max(water_change)

    sw_init, sw_final = np.array([1.0, 1.0, 1.0, 0.51, 0.23]), np.array([0.9, 0.9, 0.9, 0.95, 0.98])
    sw = np.zeros((n_zones, n_steps))
    for i in range(n_zones):
        sw[i, :] = sw_init[i] + (sw_final[i] - sw_init[i]) * (time_years / time_years[-1])

    time_mesh, zones_mesh = np.meshgrid(time_years, np.arange(n_zones))

    frames = []
    for t in range(0, n_steps, 2):
        frame_data = [
            go.Surface(
                z=water_change[:, :t+1], x=time_mesh[:, :t+1], y=zones_mesh[:, :t+1],
                surfacecolor=sw[:, :t+1], colorscale='Viridis', cmin=0.0, cmax=1.0,
                colorbar=dict(title='Sw', len=0.5), showscale=(t == n_steps - 2)
            ),
            go.Scatter3d(x=time_years[:t+1], y=[3]*len(time_years[:t+1]), z=cum_influx_4[:t+1], mode='lines', line=dict(color='red', width=4), name='Zone4 Influx'),
            go.Scatter3d(x=time_years[:t+1], y=[4]*len(time_years[:t+1]), z=cum_influx_5[:t+1], mode='lines', line=dict(color='orange', width=4), name='Zone5 Influx'),
            go.Scatter3d(x=time_years[:t+1], y=[-1]*len(time_years[:t+1]), z=pressure_norm[:t+1], mode='lines', line=dict(color='cyan', width=3, dash='dash'), name='Pressure Decline')
        ]
        frames.append(go.Frame(data=frame_data, name=f'{t}'))

    fig = go.Figure(data=frames[0].data, frames=frames)
    
    # Apply configurations
    fig.update_layout(
        title='4D Interactive Reservoir Visualization: Water Migration',
        scene=dict(xaxis_title='Time (years)', yaxis_title='Zones', zaxis_title='Δ Water Volume (m³)',
                   yaxis=dict(tickmode='array', tickvals=np.arange(n_zones), ticktext=zones_order)),
        width=1400, height=800, template='plotly_dark',
        updatemenus=[{'buttons': [{'args': [None, {'frame': {'duration': 120, 'redraw': True}, 'fromcurrent': True}], 'label': '▶ Play', 'method': 'animate'},
                                  {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}], 'label': '⏸ Pause', 'method': 'animate'}],
                      'direction': 'left', 'type': 'buttons', 'x': 0.1, 'y': 0}]
    )
    
    pio.write_html(fig, file="Reservoir_4D_Animation.html", auto_open=False)
    print("[SUCCESS] 4D Interactive animation file saved as 'Reservoir_4D_Animation.html'.")


"""
Plotting utilities for reservoir and hydrogeological presentations.
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from src.salinity_engine import calculate_adjusted_residual, compute_tds_profile

def render_static_salinity_plot(layers: list, initial_tds: float, base_residuals: list, adjusted_residuals: list):
    """Generates the primary performance chart mapping solute decline against depth."""
    depths = [0] + list(np.cumsum([l["thickness_m"] for l in layers]))
    
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(depths, base_residuals, marker='o', color='#1f77b4', linewidth=2, label='Baseline TDS')
    plt.plot(depths, adjusted_residuals, marker='s', color='#2ca02c', linewidth=2, label='With Engineered Amendments')
    
    plt.xticks(depths, ["Surface"] + [l['name'] for l in layers], rotation=35, ha='right')
    plt.xlabel("Soil Profile Boundary Structure")
    plt.ylabel("Total Dissolved Solids (TDS) [mg/L]")
    plt.title("Solute Concentration Decline Along Vertical Profile Matrix", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

def run_interactive_salinity_dashboard(layers: list, initial_tds: float, amendments_baseline: list):
    """Constructs the interactive slider control layout for web-based UI instances."""
    
    def interactive_proxy(**kwargs):
        residuals_base = [kwargs[f"{l['name']}_residual"] for l in layers]
        residuals_with_amend = []
        
        for i, l in enumerate(layers):
            res_adj = calculate_adjusted_residual(
                residuals_base[i],
                kwargs[f"{l['name']}_biochar"],
                kwargs[f"{l['name']}_zeolite"],
                kwargs[f"{l['name']}_ac"]
            )
            residuals_with_amend.append(res_adj)
            
        tds_base = compute_tds_profile(initial_tds, residuals_base)
        tds_amend = compute_tds_profile(initial_tds, residuals_with_amend)
        
        render_static_salinity_plot(layers, initial_tds, tds_base, tds_amend)
        print(f"Final Discharge TDS (Baseline): {tds_base[-1]:.2f} mg/L")
        print(f"Final Discharge TDS (Amended):  {tds_amend[-1]:.2f} mg/L")

    # Generate dynamic UI control objects
    sliders = {}
    for i, l in enumerate(layers):
        sliders[f"{l['name']}_residual"] = FloatSlider(value=l['residual'], min=0.05, max=1.0, step=0.05, description=f"{l['name'][:5]} Res")
        sliders[f"{l['name']}_biochar"] = FloatSlider(value=amendments_baseline[i]["biochar_vv"], min=0, max=0.2, step=0.01, description="↳ Biochar")
        sliders[f"{l['name']}_zeolite"] = FloatSlider(value=amendments_baseline[i]["zeolite_vv"], min=0, max=0.2, step=0.01, description="↳ Zeolite")
        sliders[f"{l['name']}_ac"] = FloatSlider(value=amendments_baseline[i]["ac_vv"], min=0, max=0.2, step=0.01, description="↳ Act Carbon")

    interact(interactive_proxy, **sliders)


"""
Plotting utilities for reservoir, hydrogeological, and statistical presentations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_statistical_publication_panel(sobol_df: pd.DataFrame, profiles_matrix: np.ndarray, depths: np.ndarray):
    """Generates a side-by-side technical layout matching academic journal standards."""
    # Compute confidence intervals
    p10 = np.percentile(profiles_matrix, 10, axis=0)
    p50 = np.percentile(profiles_matrix, 50, axis=0)
    p90 = np.percentile(profiles_matrix, 90, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot A: Sobol Indices
    ax_a = axes[0]
    ax_a.barh(sobol_df["Layer"], sobol_df["ST"], color="steelblue", edgecolor='black', height=0.6)
    ax_a.set_xlabel("Total-order Sensitivity Index ($S_T$)", fontsize=11)
    ax_a.set_title("Global Variance Sensitivity Analysis", fontsize=12, fontweight="bold")
    ax_a.grid(axis='x', linestyle=':', alpha=0.6)
    ax_a.text(-0.08, -0.09, "(a)", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

    # Subplot B: Uncertainty Envelope
    ax_b = axes[1]
    ax_b.plot(depths, p50, color="black", linewidth=2, label="Median Stochastic Fit")
    ax_b.fill_between(depths, p10, p90, color="blue", alpha=0.15, label="Uncertainty Range (P10–P90)")
    ax_b.set_xlabel("Profile Vertical Depth (m)", fontsize=11)
    ax_b.set_ylabel("Total Dissolved Solids [mg/L]", fontsize=11)
    ax_b.set_title("Stochastic Salinity Transport Envelope", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle=':', alpha=0.6)
    ax_b.legend(frameon=True, loc="upper right")
    ax_b.text(-0.08, -0.09, "(b)", transform=ax_b.transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


"""
Master Plotting Utilities Layout Interface for Reservoir, Hydrogeological, and Financial Suites.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_financial_valuation_panel(financial_arr: np.ndarray, extended_arr: np.ndarray, tornado_data: tuple):
    """Generates the combined side-by-side verification chart for financial reviews."""
    sns.set_style("whitegrid")
    labels, low_bounds, high_bounds = tornado_data
    
    fin_percentiles = np.percentile(financial_arr, [10, 50, 90])
    ext_percentiles = np.percentile(extended_arr, [10, 50, 90])
    
    loss_probability_financial = np.mean(financial_arr < 0)
    loss_probability_extended = np.mean(extended_arr < 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Left Subplot: Distributions
    ax_a = axes[0]
    sns.kdeplot(financial_arr, label=f"Financial Framework (P50={fin_percentiles[1]:.2f})", linewidth=2, color="#1f77b4", ax=ax_a)
    sns.kdeplot(extended_arr, label=f"Extended Framework (P50={ext_percentiles[1]:.2f})", linewidth=2, color="#ff7f0e", ax=ax_a)
    
    for pf, pe in zip(fin_percentiles, ext_percentiles):
        ax_a.axvline(pf, color="#1f77b4", linestyle="--", alpha=0.4)
        ax_a.axvline(pe, color="#ff7f0e", linestyle="--", alpha=0.4)
    ax_a.axvline(0, color="black", linewidth=1.2)
    
    ax_a.set_title("Stochastic Investment NPV Distributions", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Net Present Value ($ Millions)")
    ax_a.set_ylabel("Probability Vector Density")
    
    stat_box_text = f"Risk Threshold P(NPV < 0):\nPure Financial: {loss_probability_financial:.1%}\nESG Extended:   {loss_probability_extended:.1%}"
    ax_a.text(0.03, 0.95, stat_box_text, transform=ax_a.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax_a.legend(loc="upper right")
    ax_a.text(-0.08, -0.11, "(a)", transform=ax_a.transAxes, fontsize=12, fontweight="bold")
    
    # Right Subplot: Tornado Chart
    ax_b = axes[1]
    y_positions = np.arange(len(labels))
    ax_b.barh(y_positions, high_bounds, color="#2ca02c", alpha=0.75, label="High Param Value Bounds")
    ax_b.barh(y_positions, low_bounds, color="#d62728", alpha=0.75, label="Low Param Value Bounds")
    ax_b.axvline(0, color='black', linewidth=1.2)
    
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels(labels)
    ax_b.invert_yaxis()
    ax_b.set_title("Local Sensitivity Matrix (Tornado Analysis Impact Deviation)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Net Present Value Shift Target Range ($ Millions)")
    ax_b.legend(loc="lower right")
    ax_b.text(-0.08, -0.11, "(b)", transform=ax_b.transAxes, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("NPV_SideBySide_Publication.png", dpi=600, bbox_inches="tight")
    plt.show()

