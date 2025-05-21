#!/usr/bin/env python
"""
OpenWorld Battery Simulation Example
===================================

This example demonstrates how to use the OpenWorld battery simulation module
to simulate a lithium-ion battery during charge and discharge cycles.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from openworld.core.battery import BatteryParameters
from openworld.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

def run_example():
    """Run a basic battery simulation example"""
    print("OpenWorld Battery Simulation Example")
    print("=====================================")
    
    # 1. Create battery parameters for different chemistries
    print("\nInitializing battery parameters for different chemistries...")
    
    # Create battery parameter objects for different chemistries
    battery_params = {
        "NMC": BatteryParameters(param_set_name="graphite_nmc"),
        "LFP": BatteryParameters(param_set_name="graphite_lfp"),
        "High-Energy NMC811": BatteryParameters(param_set_name="nmc811"),
        "Silicon-NMC": BatteryParameters(param_set_name="silicon_nmc"),
        "High-Power LFP": BatteryParameters(param_set_name="graphite_lfp_high_power")
    }
    
    # 2. Print key parameters for each chemistry
    print("\nComparing key parameters for different chemistries:")
    print(f"{'Chemistry':<20} {'Capacity (Ah)':<15} {'Negative Diff (m²/s)':<20} {'Positive Diff (m²/s)':<20}")
    print("-" * 80)
    
    for name, params in battery_params.items():
        # Scale all batteries to 3Ah for fair comparison
        params.scale_capacity(3.0)
        
        # Print key parameters
        print(f"{name:<20} {params.nominal_capacity_Ah.magnitude:<15.2f} "
              f"{params.D_s_n.magnitude:<20.2e} {params.D_s_p.magnitude:<20.2e}")
    
    # 3. Simulate a charge-discharge cycle (placeholder)
    print("\nSimulating battery cycles (placeholder)...")
    
    # In the full implementation, you would import DFNModel and BatterySimulation
    # and run an actual simulation. For now, this is a placeholder.
    
    # Generate dummy data for visualization
    time_points = np.linspace(0, 4*3600, 1000)  # 4 hours
    
    # Create C/2 charge, C/2 discharge profile with rest periods
    current_profile = np.zeros_like(time_points)
    
    # 1h C/2 charge
    idx_1h = int(len(time_points) * 1/4)
    current_profile[:idx_1h] = 1.5  # 1.5A is C/2 for a 3Ah cell
    
    # 1h rest
    idx_2h = int(len(time_points) * 2/4)
    
    # 1h C/2 discharge
    idx_3h = int(len(time_points) * 3/4)
    current_profile[idx_2h:idx_3h] = -1.5
    
    # Final rest
    
    # Generate dummy SOC profile
    soc_profile = np.zeros_like(time_points)
    soc_profile[0] = 0.3  # Start at 30% SOC
    
    # Calculate SOC changes based on current
    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i-1]
        dsoc = current_profile[i-1] * dt / (3.0 * 3600)  # SOC change = I*dt/capacity
        soc_profile[i] = min(max(soc_profile[i-1] + dsoc, 0.0), 1.0)  # Clamp to 0-1
    
    # Generate voltage profile based on SOC
    voltage_profile = 3.2 + 0.8 * soc_profile + 0.2 * np.random.randn(len(time_points)) * 0.01
    
    # Limit voltage during CC-CV charging
    for i in range(len(time_points)):
        if current_profile[i] > 0 and voltage_profile[i] > 4.2:
            voltage_profile[i] = 4.2
            
        if soc_profile[i] < 0.05:
            voltage_profile[i] = 3.2  # Low voltage cutoff
    
    # 4. Plot results
    print("\nGenerating plots...")
    
    # Current and SOC plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot current
    ax1.plot(time_points/3600, current_profile, 'b-', linewidth=2)
    ax1.set_ylabel('Current (A)')
    ax1.set_title('Battery Cycle Simulation (Placeholder)')
    ax1.grid(True, alpha=0.3)
    
    # Plot SOC
    ax2.plot(time_points/3600, soc_profile * 100, 'g-', linewidth=2)
    ax2.set_ylabel('SOC (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot voltage
    ax3.plot(time_points/3600, voltage_profile, 'r-', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig("battery_cycle_simulation.png", dpi=300)
    
    # 5. Compare chemistry performance (placeholder)
    print("\nComparing chemistry performance (placeholder)...")
    
    # Plot dummy discharge curves for different chemistries
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Placeholder capacity range (0-100%)
    capacity_pct = np.linspace(0, 100, 100)
    
    # Characteristic voltage curves for different chemistries
    voltage_curves = {
        "NMC": 3.0 + 1.2 * np.exp(-capacity_pct/20) * np.exp(-(capacity_pct-90)**2/500),
        "LFP": 3.2 + 0.2 * np.exp(-capacity_pct/10) - 0.2 * np.exp((capacity_pct-95)/5),
        "High-Energy NMC811": 3.0 + 1.3 * np.exp(-capacity_pct/25) * np.exp(-(capacity_pct-85)**2/450),
        "Silicon-NMC": 2.8 + 1.5 * np.exp(-capacity_pct/30) * np.exp(-(capacity_pct-80)**2/500),
        "High-Power LFP": 3.15 + 0.25 * np.exp(-capacity_pct/5) - 0.2 * np.exp((capacity_pct-90)/10)
    }
    
    # Plot each chemistry
    colors = ['b', 'g', 'r', 'c', 'm']
    for (name, voltage), color in zip(voltage_curves.items(), colors):
        ax.plot(capacity_pct, voltage, color=color, linewidth=2, label=name)
    
    ax.set_xlabel('Discharge Capacity (%)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Discharge Curves for Different Battery Chemistries (Placeholder)')
    ax.set_xlim(0, 100)
    ax.set_ylim(2.5, 4.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("battery_chemistry_comparison.png", dpi=300)
    
    print("\nExample complete! Output files:")
    print("- battery_cycle_simulation.png")
    print("- battery_chemistry_comparison.png")
    print("\nIn a full implementation, connect this to the DFNModel and BatterySimulation classes")

if __name__ == "__main__":
    run_example() 