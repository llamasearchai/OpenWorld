"""
Visualization utilities for PhysicsGPT.

This module provides comprehensive plotting functions for the various physics modules,
including battery, solar cell, and general physics visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Union, Optional

# Updated import paths for OpenWorld structure
from ..utils.units import ureg, get_value 
from ..utils.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Battery Visualization Functions
# -----------------------------------------------------------------------------

def plot_voltage_current_soc(results: Dict, figsize: Tuple[float, float] = (10, 8)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot voltage, current, and SOC from battery simulation results.
    
    Args:
        results: Dictionary containing simulation results
        figsize: Figure size (width, height)
        
    Returns:
        Figure and list of axes
    """
    time_h = get_value(results.get('time', [])) / 3600  # Convert to hours
    voltage_v = get_value(results.get('voltage', []))
    current_a = get_value(results.get('current', []))
    soc_pct = get_value(results.get('soc', [])) * 100  # Convert to percentage
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig)
    
    # Voltage plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_h, voltage_v, 'b-')
    ax1.set_ylabel('Voltage [V]')
    ax1.grid(True)
    
    # Current plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_h, current_a, 'r-')
    ax2.set_ylabel('Current [A]')
    ax2.grid(True)
    
    # SOC plot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_h, soc_pct, 'g-')
    ax3.set_xlabel('Time [h]')
    ax3.set_ylabel('SOC [%]')
    ax3.grid(True)
    
    plt.tight_layout()
    
    return fig, [ax1, ax2, ax3]

def plot_battery_concentration_profiles(conc_data: Dict, time_idx: int = -1, figsize: Tuple[float, float] = (12, 10)) -> plt.Figure:
    """
    Plot concentration profiles in battery electrodes and electrolyte.
    
    Args:
        conc_data: Dictionary with concentration profiles
        time_idx: Time index to plot (-1 for final time)
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plots
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # Extract data
    x_n = get_value(conc_data.get('x_n', []))
    x_p = get_value(conc_data.get('x_p', []))
    x_electrolyte = get_value(conc_data.get('x_electrolyte', []))
    r_n = get_value(conc_data.get('r_n', []))
    r_p = get_value(conc_data.get('r_p', []))
    
    c_s_n = get_value(conc_data.get('c_s_n', []))[time_idx]
    c_s_p = get_value(conc_data.get('c_s_p', []))[time_idx]
    c_e = get_value(conc_data.get('c_e', []))[time_idx]
    
    # Negative electrode concentration
    ax1 = fig.add_subplot(gs[0, 0])
    for i, x in enumerate(x_n):
        if i % max(1, len(x_n) // 5) == 0:  # Plot a subset of locations
            ax1.plot(r_n, c_s_n[i], label=f'x = {x:.2f}')
    ax1.set_xlabel('Radial position [m]')
    ax1.set_ylabel('Concentration [mol/m³]')
    ax1.set_title('Negative Electrode Solid Concentration')
    ax1.legend()
    ax1.grid(True)
    
    # Positive electrode concentration
    ax2 = fig.add_subplot(gs[0, 1])
    for i, x in enumerate(x_p):
        if i % max(1, len(x_p) // 5) == 0:  # Plot a subset of locations
            ax2.plot(r_p, c_s_p[i], label=f'x = {x:.2f}')
    ax2.set_xlabel('Radial position [m]')
    ax2.set_ylabel('Concentration [mol/m³]')
    ax2.set_title('Positive Electrode Solid Concentration')
    ax2.legend()
    ax2.grid(True)
    
    # Electrolyte concentration
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x_electrolyte, c_e, 'b-')
    ax3.set_xlabel('Cell position [m]')
    ax3.set_ylabel('Electrolyte concentration [mol/m³]')
    ax3.set_title('Electrolyte Concentration Profile')
    
    # Add separator boundaries
    if 'L_n' in conc_data and 'L_s' in conc_data:
        L_n = get_value(conc_data['L_n'])
        L_s = get_value(conc_data['L_s'])
        ax3.axvline(x=L_n, color='k', linestyle='--', label='Neg/Sep Boundary')
        ax3.axvline(x=L_n + L_s, color='r', linestyle='--', label='Sep/Pos Boundary')
        ax3.legend()
    
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

def plot_battery_discharge_curves(results_list: List[Dict], labels: List[str], figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot discharge curves for multiple battery simulations.
    
    Args:
        results_list: List of simulation result dictionaries
        labels: Labels for each simulation
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plots
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, results in enumerate(results_list):
        capacity = get_value(results.get('capacity', []))
        voltage = get_value(results.get('voltage', []))
        
        ax.plot(capacity, voltage, label=labels[i])
    
    ax.set_xlabel('Capacity [Ah]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title('Discharge Curves')
    ax.grid(True)
    ax.legend()
    
    return fig

# -----------------------------------------------------------------------------
# Solar Cell Visualization Functions
# -----------------------------------------------------------------------------

def plot_jv_curve(jv_data: Dict, metrics: Dict = None, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot J-V curve for solar cell simulation.
    
    Args:
        jv_data: Dictionary with voltage and current density data
        metrics: Optional dictionary with performance metrics
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plot
    """
    voltage = get_value(jv_data.get('voltage', []))
    current = get_value(jv_data.get('current_density', []))
    power = voltage * current
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot J-V curve
    ax1.plot(voltage, current, 'b-', label='J-V Curve')
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current Density [mA/cm²]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Add power curve on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(voltage, power, 'r--', label='Power')
    ax2.set_ylabel('Power [mW/cm²]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add performance metrics if provided
    if metrics:
        # Find max power point
        mpp_v = metrics.get('mpp_voltage', 0)
        mpp_j = metrics.get('mpp_current_density', 0)
        
        # Plot markers for key points
        ax1.plot([0], [metrics.get('jsc', 0)], 'bo', label='Jsc')
        ax1.plot([metrics.get('voc', 0)], [0], 'bo', label='Voc')
        ax1.plot([mpp_v], [mpp_j], 'ro', label='MPP')
        
        # Add metrics as text
        text = (
            f"PCE: {metrics.get('pce', 0):.2f}%\n"
            f"Voc: {metrics.get('voc', 0):.3f} V\n"
            f"Jsc: {metrics.get('jsc', 0):.2f} mA/cm²\n"
            f"FF: {metrics.get('ff', 0):.3f}"
        )
        ax1.text(0.05, 0.05, text, transform=ax1.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Solar Cell J-V Characteristics')
    plt.grid(True)
    plt.tight_layout()
    
    return fig

def plot_carrier_profiles(carrier_data: Dict, position_idx: int = None, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Plot carrier concentration and potential profiles in solar cell.
    
    Args:
        carrier_data: Dictionary with carrier concentration data
        position_idx: Optional position index for plotting specific location
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plots
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig)
    
    # Extract data
    position = get_value(carrier_data.get('position', []))
    electron_conc = get_value(carrier_data.get('electron_concentration', []))
    hole_conc = get_value(carrier_data.get('hole_concentration', []))
    potential = get_value(carrier_data.get('electric_potential', []))
    
    # For band diagram
    conduction_band = get_value(carrier_data.get('conduction_band', []))
    valence_band = get_value(carrier_data.get('valence_band', []))
    
    # Plot carrier concentrations
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(position * 1e9, electron_conc, 'b-', label='Electrons')
    ax1.semilogy(position * 1e9, hole_conc, 'r-', label='Holes')
    ax1.set_xlabel('Position [nm]')
    ax1.set_ylabel('Carrier Concentration [cm⁻³]')
    ax1.set_title('Carrier Concentration Profiles')
    ax1.legend()
    ax1.grid(True)
    
    # Plot band diagram or potential
    ax2 = fig.add_subplot(gs[1])
    
    if conduction_band is not None and valence_band is not None:
        # Plot band diagram
        ax2.plot(position * 1e9, conduction_band, 'b-', label='Conduction Band')
        ax2.plot(position * 1e9, valence_band, 'r-', label='Valence Band')
        ax2.set_ylabel('Energy [eV]')
        ax2.set_title('Band Diagram')
    else:
        # Plot potential
        ax2.plot(position * 1e9, potential, 'g-')
        ax2.set_ylabel('Electric Potential [V]')
        ax2.set_title('Electric Potential Profile')
    
    ax2.set_xlabel('Position [nm]')
    ax2.grid(True)
    
    # Add layer boundaries if available
    if 'layer_boundaries' in carrier_data:
        boundaries = get_value(carrier_data['layer_boundaries'])
        for boundary in boundaries:
            ax1.axvline(x=boundary * 1e9, color='k', linestyle='--')
            ax2.axvline(x=boundary * 1e9, color='k', linestyle='--')
    
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Physics Visualization Functions
# -----------------------------------------------------------------------------

def plot_projectile_trajectory(trajectory_data: Dict, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot projectile motion trajectory.
    
    Args:
        trajectory_data: Dictionary with trajectory data
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plot
    """
    x = get_value(trajectory_data.get('x', []))
    y = get_value(trajectory_data.get('y', []))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trajectory
    ax.plot(x, y, 'b-')
    
    # Mark start, end, and apex points
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Find apex (highest point)
    apex_idx = np.argmax(y)
    ax.plot(x[apex_idx], y[apex_idx], 'bo', markersize=10, label='Apex')
    
    ax.set_xlabel('x position [m]')
    ax.set_ylabel('y position [m]')
    ax.set_title('Projectile Motion Trajectory')
    ax.grid(True)
    ax.legend()
    
    # Set aspect ratio to equal for realistic trajectory
    ax.set_aspect('equal', adjustable='box')
    
    return fig

def plot_stress_strain_curve(data: Dict, figsize: Tuple[float, float] = (8, 6)) -> plt.Figure:
    """
    Plot stress-strain curve for material mechanics.
    
    Args:
        data: Dictionary with stress and strain data
        figsize: Figure size (width, height)
        
    Returns:
        Figure with plot
    """
    strain = get_value(data.get('strain', []))
    stress = get_value(data.get('stress', []))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(strain, stress, 'b-')
    
    # Mark elastic limit if available
    if 'elastic_limit' in data:
        elastic_limit = get_value(data['elastic_limit'])
        elastic_strain = strain[strain <= elastic_limit]
        elastic_stress = stress[strain <= elastic_limit]
        
        # Find stress at elastic limit
        if len(elastic_strain) > 0:
            elastic_stress_val = elastic_stress[-1]
            ax.plot(elastic_limit, elastic_stress_val, 'ro', markersize=8, 
                   label='Elastic Limit')
            
            # Add dashed line for elastic region
            x_elastic = np.linspace(0, elastic_limit, 100)
            y_elastic = (elastic_stress_val / elastic_limit) * x_elastic
            ax.plot(x_elastic, y_elastic, 'r--', label='Elastic Region')
    
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress [MPa]')
    ax.set_title('Stress-Strain Curve')
    ax.grid(True)
    ax.legend()
    
    return fig

# -----------------------------------------------------------------------------
# Interactive Plotting (Plotly)
# -----------------------------------------------------------------------------

def create_interactive_battery_plot(results: Dict, title: str = "Battery Simulation Results") -> go.Figure:
    """
    Create an interactive plotly figure for battery simulation results.
    
    Args:
        results: Dictionary with simulation results
        title: Plot title
        
    Returns:
        Plotly figure
    """
    time_h = get_value(results.get('time', [])) / 3600  # Convert to hours
    voltage_v = get_value(results.get('voltage', []))
    current_a = get_value(results.get('current', []))
    soc_pct = get_value(results.get('soc', [])) * 100  # Convert to percentage
    
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=('Voltage', 'Current', 'State of Charge'),
                        vertical_spacing=0.1)
    
    # Add voltage trace
    fig.add_trace(
        go.Scatter(x=time_h, y=voltage_v, mode='lines', name='Voltage [V]',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add current trace
    fig.add_trace(
        go.Scatter(x=time_h, y=current_a, mode='lines', name='Current [A]',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    # Add SOC trace
    fig.add_trace(
        go.Scatter(x=time_h, y=soc_pct, mode='lines', name='SOC [%]',
                  line=dict(color='green')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Set x-axis title only on the bottom plot
    fig.update_xaxes(title_text="Time [h]", row=3, col=1)
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_yaxes(title_text="Current [A]", row=2, col=1)
    fig.update_yaxes(title_text="SOC [%]", row=3, col=1)
    
    return fig

def create_interactive_jv_plot(jv_data: Dict, metrics: Dict = None, title: str = "Solar Cell J-V Characteristics") -> go.Figure:
    """
    Create an interactive plotly figure for solar cell J-V curve.
    
    Args:
        jv_data: Dictionary with voltage and current density data
        metrics: Optional dictionary with performance metrics
        title: Plot title
        
    Returns:
        Plotly figure
    """
    voltage = get_value(jv_data.get('voltage', []))
    current = get_value(jv_data.get('current_density', []))
    power = voltage * current
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add J-V curve
    fig.add_trace(
        go.Scatter(x=voltage, y=current, mode='lines', name='J-V Curve',
                  line=dict(color='blue')),
        secondary_y=False
    )
    
    # Add power curve
    fig.add_trace(
        go.Scatter(x=voltage, y=power, mode='lines', name='Power',
                  line=dict(color='red', dash='dash')),
        secondary_y=True
    )
    
    # Add performance metrics if provided
    if metrics:
        # Find max power point
        mpp_v = metrics.get('mpp_voltage', 0)
        mpp_j = metrics.get('mpp_current_density', 0)
        mpp_p = mpp_v * mpp_j
        
        # Add key points
        fig.add_trace(
            go.Scatter(x=[0], y=[metrics.get('jsc', 0)], mode='markers',
                      marker=dict(color='blue', size=10), name='Jsc'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=[metrics.get('voc', 0)], y=[0], mode='markers',
                      marker=dict(color='blue', size=10), name='Voc'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=[mpp_v], y=[mpp_j], mode='markers',
                      marker=dict(color='red', size=10), name='MPP'),
            secondary_y=False
        )
        
        # Add metrics as annotations
        fig.add_annotation(
            x=0.05, y=0.05,
            xref="paper", yref="paper",
            text=(f"PCE: {metrics.get('pce', 0):.2f}%<br>"
                  f"Voc: {metrics.get('voc', 0):.3f} V<br>"
                  f"Jsc: {metrics.get('jsc', 0):.2f} mA/cm²<br>"
                  f"FF: {metrics.get('ff', 0):.3f}"),
            showarrow=False,
            bgcolor="white",
            opacity=0.8
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        width=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Set axes titles
    fig.update_xaxes(title_text="Voltage [V]")
    fig.update_yaxes(title_text="Current Density [mA/cm²]", secondary_y=False)
    fig.update_yaxes(title_text="Power [mW/cm²]", secondary_y=True)
    
    return fig 