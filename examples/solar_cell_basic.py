#!/usr/bin/env python
"""
OpenWorld Solar Cell Example
==========================

This example demonstrates how to use the OpenWorld solar cell module
to simulate and analyze semiconductor solar cells.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from openworld.core.solar.material import MaterialProperties, COMMON_MATERIALS
from openworld.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

def run_example():
    """Run a basic solar cell simulation example"""
    print("OpenWorld Solar Cell Example")
    print("===========================")
    
    # 1. Create and analyze materials for solar cells
    print("\nInitializing semiconductor materials for solar cells...")
    
    # Create a list of materials to compare
    materials = {
        "Silicon": COMMON_MATERIALS["silicon"],
        "GaAs": COMMON_MATERIALS["gaas"],
        "CdTe": COMMON_MATERIALS["cdte"],
        "CIGS": COMMON_MATERIALS["cigs"],
        "Perovskite": COMMON_MATERIALS["perovskite"]
    }
    
    # 2. Compare key material properties
    print("\nComparing key properties for different materials:")
    print(f"{'Material':<12} {'Band Gap (eV)':<15} {'Electron Mobility (cm²/Vs)':<25} {'Hole Mobility (cm²/Vs)':<23}")
    print("-" * 80)
    
    for name, material in materials.items():
        print(f"{name:<12} {material.band_gap:<15.2f} {material.mu_n:<25.1f} {material.mu_p:<23.1f}")
    
    # 3. Calculate and compare diffusion lengths
    print("\nCalculating minority carrier diffusion lengths (µm):")
    print(f"{'Material':<12} {'Electron (µm)':<15} {'Hole (µm)':<15}")
    print("-" * 45)
    
    for name, material in materials.items():
        # Create n-type version
        n_type = material.copy_with_modifications(doping_type="n", doping_concentration=1e16)
        
        # Create p-type version
        p_type = material.copy_with_modifications(doping_type="p", doping_concentration=1e16)
        
        # Calculate diffusion lengths (convert from cm to µm)
        electron_diffusion_length = n_type.get_diffusion_length(carrier_type="minority") * 1e4
        hole_diffusion_length = p_type.get_diffusion_length(carrier_type="minority") * 1e4
        
        print(f"{name:<12} {electron_diffusion_length:<15.2f} {hole_diffusion_length:<15.2f}")
    
    # 4. Generate wavelength-dependent absorption for materials
    print("\nCalculating absorption coefficients vs wavelength...")
    
    # Create wavelength range from 300 to 1200 nm
    wavelengths = np.linspace(300, 1200, 100)
    
    # Calculate corresponding photon energies (in eV)
    photon_energies = 1240 / wavelengths
    
    # Calculate absorption coefficients for each material
    absorption_data = {}
    
    for name, material in materials.items():
        abs_coeffs = []
        for wl in wavelengths:
            abs_coeffs.append(material.get_absorption_at_wavelength(wl))
        absorption_data[name] = np.array(abs_coeffs)
    
    # 5. Plot absorption coefficients
    print("\nGenerating absorption coefficient plot...")
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for (name, abs_coeffs), color in zip(absorption_data.items(), colors):
        plt.semilogy(wavelengths, abs_coeffs, color=color, linewidth=2, label=name)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption Coefficient (cm⁻¹)')
    plt.title('Absorption Coefficients for Solar Cell Materials')
    plt.xlim(300, 1200)
    plt.ylim(1e2, 1e6)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("solar_materials_absorption.png", dpi=300)
    
    # 6. Calculate maximum theoretical solar cell efficiencies (placeholder)
    print("\nCalculating maximum theoretical efficiencies (placeholder)...")
    
    # Create solar spectrum (simplified AM1.5G approximation) - placeholder
    solar_irradiance = 100  # mW/cm²
    
    # Theoretical detailed balance limit efficiencies (simplified model)
    # These approximate the Shockley-Queisser limit 
    shoq_eff = {}
    for name, material in materials.items():
        # Simplified efficiency calculation based on band gap
        # This is a polynomial fit to the S-Q limit, not an exact calculation
        Eg = material.band_gap
        if Eg < 0.5 or Eg > 2.0:
            eff = 5.0  # Very low for small or large band gaps
        else:
            # Polynomial approximation of S-Q limit curve
            eff = -34.1 * Eg**2 + 97.0 * Eg - 31.3
        
        # Cap at reasonable values
        eff = max(0, min(eff, 33.7))
        shoq_eff[name] = eff
    
    # 7. Plot theoretical efficiency vs band gap
    print("\nGenerating theoretical efficiency plot...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot theoretical curve
    band_gaps = np.linspace(0.5, 2.5, 100)
    sq_curve = -34.1 * band_gaps**2 + 97.0 * band_gaps - 31.3
    sq_curve = np.maximum(0, np.minimum(sq_curve, 33.7))
    plt.plot(band_gaps, sq_curve, 'k-', linewidth=2, label='Theoretical Limit')
    
    # Plot materials
    for (name, material), color in zip(materials.items(), colors):
        plt.plot(material.band_gap, shoq_eff[name], color + 'o', markersize=10, label=name)
    
    plt.xlabel('Band Gap (eV)')
    plt.ylabel('Maximum Theoretical Efficiency (%)')
    plt.title('Shockley-Queisser Limit for Solar Cell Materials')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("solar_theoretical_efficiency.png", dpi=300)
    
    # 8. Create recombination rates vs carrier concentration plot
    print("\nCalculating recombination mechanisms...")
    
    # Create carrier concentration range
    carrier_concentrations = np.logspace(14, 18, 100)  # cm⁻³
    
    # Calculate recombination rates for silicon
    silicon = materials["Silicon"]
    recombination_rates = {
        "SRH": [],
        "radiative": [],
        "Auger": [],
        "total": []
    }
    
    for n in carrier_concentrations:
        # Assume equal electron and hole concentrations
        rates = silicon.get_recombination_rate(n, n)
        for mech, rate in rates.items():
            recombination_rates[mech].append(rate)
    
    # Convert to numpy arrays
    for mech in recombination_rates:
        recombination_rates[mech] = np.array(recombination_rates[mech])
    
    # Plot recombination rates
    print("\nGenerating recombination mechanisms plot...")
    
    plt.figure(figsize=(10, 6))
    
    plt.loglog(carrier_concentrations, recombination_rates["SRH"], 'b-', 
               linewidth=2, label='SRH (defect)')
    plt.loglog(carrier_concentrations, recombination_rates["radiative"], 'g-', 
               linewidth=2, label='Radiative')
    plt.loglog(carrier_concentrations, recombination_rates["Auger"], 'r-', 
               linewidth=2, label='Auger')
    plt.loglog(carrier_concentrations, recombination_rates["total"], 'k--', 
               linewidth=2, label='Total')
    
    plt.xlabel('Carrier Concentration (cm⁻³)')
    plt.ylabel('Recombination Rate (cm⁻³/s)')
    plt.title('Recombination Mechanisms in Silicon')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("silicon_recombination_mechanisms.png", dpi=300)
    
    print("\nExample complete! Output files:")
    print("- solar_materials_absorption.png")
    print("- solar_theoretical_efficiency.png")
    print("- silicon_recombination_mechanisms.png")
    print("\nIn a full implementation, use SolarCellDevice and DriftDiffusionModel")

if __name__ == "__main__":
    run_example() 