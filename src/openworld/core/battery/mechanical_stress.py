"""
Mechanical stress model for battery electrodes.

This module implements a mechanical stress model for battery electrodes,
calculating diffusion-induced stresses, tracking crack growth, and predicting
capacity fade based on mechanical degradation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from pint import Quantity

# Updated import paths for OpenWorld structure
from ...utils.units import ureg 
from ...utils.logging import get_logger

logger = get_logger(__name__)

class MechanicalStressModel:
    """
    Class for calculating mechanical stress in battery electrode particles.
    
    Implements the Christensen-Newman model for diffusion-induced stress in
    spherical electrode particles, with extensions for tracking cumulative
    damage and crack growth using Paris law.
    """
    
    def __init__(self, 
                 particle_radius: Union[float, Quantity],
                 youngs_modulus: Union[float, Quantity],
                 poissons_ratio: float,
                 partial_molar_volume: Union[float, Quantity],
                 initial_crack_size: Union[float, Quantity] = None,
                 paris_law_prefactor: float = 1e-12,
                 paris_law_exponent: float = 3.0,
                 max_concentration: Union[float, Quantity] = None):
        """
        Initialize the mechanical stress model.
        
        Args:
            particle_radius: Radius of the electrode particle [m]
            youngs_modulus: Young's modulus of the electrode material [Pa]
            poissons_ratio: Poisson's ratio of the electrode material [dimensionless]
            partial_molar_volume: Partial molar volume of lithium in the host [m³/mol]
            initial_crack_size: Initial size of cracks in the material [m]
            paris_law_prefactor: Prefactor A in Paris law (da/dN = A * ΔK^m) [m/cycle/(MPa*√m)^m]
            paris_law_exponent: Exponent m in Paris law [dimensionless]
            max_concentration: Maximum concentration of lithium in the material [mol/m³]
        """
        # Convert inputs to correct units
        self.R = self._ensure_quantity(particle_radius, ureg.meter)
        self.E = self._ensure_quantity(youngs_modulus, ureg.pascal)
        self.nu = poissons_ratio
        self.Omega = self._ensure_quantity(partial_molar_volume, ureg.meter**3 / ureg.mol)
        
        # Crack growth parameters
        self.a_0 = self._ensure_quantity(initial_crack_size, ureg.meter) if initial_crack_size is not None else 1e-9 * ureg.meter
        self.paris_A = paris_law_prefactor
        self.paris_m = paris_law_exponent
        
        # State variables
        self.max_concentration = max_concentration
        self.cumulative_damage = 0.0  # Dimensionless damage parameter
        self.crack_length = self.a_0  # Current crack length
        self.max_tensile_stress_history = []  # To track stress over time/cycles
        self.cycle_count = 0
        
        logger.info(f"Initialized mechanical stress model with R={self.R:.2e~P}, E={self.E:.2e~P}")
    
    def _ensure_quantity(self, value, unit):
        """Convert value to a Quantity with the specified unit if it's not already."""
        if isinstance(value, Quantity):
            return value.to(unit)
        else:
            return ureg.Quantity(value, unit)
    
    def calculate_stress(self, 
                         concentration_profile: np.ndarray, 
                         radial_positions: np.ndarray,
                         avg_concentration: Union[float, Quantity] = None) -> Dict[str, np.ndarray]:
        """
        Calculate radial and tangential stress in a spherical particle.
        
        Implements the Christensen-Newman model for diffusion-induced stress.
        
        Args:
            concentration_profile: Lithium concentration at each radial position [mol/m³]
            radial_positions: Radial positions from center to surface [m]
            avg_concentration: Average concentration in the particle [mol/m³]
            
        Returns:
            Dictionary with radial and tangential stress arrays [Pa]
        """
        # Convert inputs to correct units
        r_vals = np.array(radial_positions)
        c_vals = np.array(concentration_profile)
        
        if hasattr(r_vals, 'magnitude'):
            r_vals = r_vals.magnitude
        if hasattr(c_vals, 'magnitude'):
            c_vals = c_vals.magnitude
            
        # Ensure increasing r values (center to surface)
        if r_vals[0] > r_vals[-1]:
            r_vals = r_vals[::-1]
            c_vals = c_vals[::-1]
            
        # Calculate average concentration if not provided
        if avg_concentration is None:
            # Volumetric average accounting for spherical geometry
            r_centers = 0.5 * (r_vals[:-1] + r_vals[1:])
            vol_shells = 4/3 * np.pi * (r_vals[1:]**3 - r_vals[:-1]**3)
            c_shells = 0.5 * (c_vals[:-1] + c_vals[1:])
            avg_c = np.sum(c_shells * vol_shells) / (4/3 * np.pi * r_vals[-1]**3)
        else:
            avg_c = avg_concentration.magnitude if hasattr(avg_concentration, 'magnitude') else avg_concentration
            
        # Calculate stresses using Christensen-Newman solution
        R = self.R.magnitude
        E = self.E.magnitude
        nu = self.nu
        Omega = self.Omega.magnitude
        
        # Normalize radial positions
        r_norm = r_vals / R
        
        # Radial and tangential stress arrays
        sigma_r = np.zeros_like(r_vals)
        sigma_t = np.zeros_like(r_vals)
        
        # Modulus factor in stress equation
        mod_factor = E * Omega / (3 * (1 - nu))
        
        # Calculate stress at each radial position
        for i, r in enumerate(r_norm):
            if r == 0:  # Special case at the center
                # Both radial and tangential stress are equal at center
                term1 = avg_c
                # Integration from 0 to R (using trapezoidal rule on discretized data)
                term2 = np.trapz(c_vals * r_norm**2, r_norm)
                sigma_r[i] = sigma_t[i] = mod_factor * (term1 - term2)
            else:
                # Radial stress
                # Integration from r to R (using trapezoidal rule on slice of data)
                idx_r_to_R = r_norm >= r
                if np.sum(idx_r_to_R) > 1:
                    term = np.trapz(c_vals[idx_r_to_R] * r_norm[idx_r_to_R]**2, r_norm[idx_r_to_R])
                    sigma_r[i] = 2 * mod_factor * term / r**3
                else:
                    sigma_r[i] = 0  # At surface, radial stress is zero
                
                # Tangential stress
                term1 = avg_c
                # Integration from 0 to r
                idx_0_to_r = r_norm <= r
                if np.sum(idx_0_to_r) > 1:
                    term2 = np.trapz(c_vals[idx_0_to_r] * r_norm[idx_0_to_r]**2, r_norm[idx_0_to_r])
                else:
                    term2 = 0
                # Integration from r to R
                term3 = np.trapz(c_vals[idx_r_to_R] * r_norm[idx_r_to_R]**2, r_norm[idx_r_to_R])
                sigma_t[i] = mod_factor * (term1 + term2/r**3 - term3/r**3 - c_vals[i])
        
        # Convert back to quantities with proper units
        sigma_r_q = sigma_r * ureg.pascal
        sigma_t_q = sigma_t * ureg.pascal
        
        # Store maximum tensile stress for damage calculation
        max_tensile_stress = np.max(sigma_t_q)
        self.max_tensile_stress_history.append(max_tensile_stress)
        
        return {
            "radial_stress": sigma_r_q,
            "tangential_stress": sigma_t_q,
            "max_tensile_stress": max_tensile_stress
        }
    
    def calculate_stress_intensity_factor(self, 
                                         max_stress: Union[float, Quantity], 
                                         crack_length: Union[float, Quantity] = None) -> Quantity:
        """
        Calculate the stress intensity factor for a surface crack.
        
        Args:
            max_stress: Maximum tensile stress [Pa]
            crack_length: Crack length [m], defaults to current crack length
            
        Returns:
            Stress intensity factor [Pa·√m]
        """
        # Use current crack length if none provided
        a = crack_length if crack_length is not None else self.crack_length
        a = self._ensure_quantity(a, ureg.meter)
        
        # Convert stress to proper units
        stress = self._ensure_quantity(max_stress, ureg.pascal)
        
        # Geometric factor for surface crack
        Y = 1.12  # Simplified factor for surface crack
        
        # Calculate stress intensity factor: K = Y * σ * √(π * a)
        K = Y * stress * np.sqrt(np.pi * a)
        
        return K.to(ureg.pascal * ureg.meter**0.5)
    
    def update_crack_growth(self, stress_range: Union[float, Quantity]) -> Tuple[Quantity, float]:
        """
        Update crack growth using Paris law.
        
        Args:
            stress_range: Stress range for the cycle [Pa]
            
        Returns:
            Tuple of (new_crack_length, damage_increment)
        """
        self.cycle_count += 1
        
        # Calculate stress intensity factor range
        delta_K = self.calculate_stress_intensity_factor(stress_range)
        
        # Convert to MPa*√m for Paris law
        delta_K_MPa_m = delta_K.to(ureg.megapascal * ureg.meter**0.5)
        
        # Apply Paris law: da/dN = A * (ΔK)^m
        da_dN = self.paris_A * (delta_K_MPa_m.magnitude ** self.paris_m)
        crack_increment = da_dN * ureg.meter
        
        # Update crack length
        self.crack_length += crack_increment
        
        # Calculate damage increment
        # Damage is defined as ratio of current to critical crack length
        critical_length = self.R  # Simplification: critical length is particle radius
        damage_increment = crack_increment / critical_length
        
        # Update cumulative damage
        self.cumulative_damage += damage_increment.magnitude
        
        return self.crack_length, damage_increment.magnitude
    
    def calculate_capacity_fade(self, initial_capacity: Union[float, Quantity]) -> Dict[str, Any]:
        """
        Calculate capacity fade based on mechanical damage.
        
        Args:
            initial_capacity: Initial capacity [Ah]
            
        Returns:
            Dictionary with capacity fade metrics
        """
        # Convert capacity to quantity if needed
        initial_capacity_q = self._ensure_quantity(initial_capacity, ureg.ampere_hour)
        
        # Calculate capacity fade based on cumulative damage
        # Simplified model: capacity fade proportional to damage
        capacity_fade_factor = self.cumulative_damage
        
        # Limit maximum fade to 30% due to mechanical effects
        capacity_fade_factor = min(capacity_fade_factor, 0.3)
        
        # Calculate remaining capacity
        remaining_capacity = initial_capacity_q * (1 - capacity_fade_factor)
        capacity_fade = initial_capacity_q * capacity_fade_factor
        
        return {
            "capacity_fade_factor": capacity_fade_factor,
            "capacity_fade": capacity_fade,
            "remaining_capacity": remaining_capacity,
            "capacity_fade_percent": capacity_fade_factor * 100
        }
    
    def get_damage_metrics(self) -> Dict[str, Any]:
        """
        Get current damage metrics.
        
        Returns:
            Dictionary with damage metrics
        """
        # Calculate critical stress intensity factor (fracture toughness)
        # Typical value for graphite: ~1 MPa*√m
        K_IC = 1.0 * ureg.megapascal * ureg.meter**0.5
        
        # Calculate current stress intensity factor if stress history exists
        current_K = None
        safety_factor = None
        if self.max_tensile_stress_history:
            max_stress = max(self.max_tensile_stress_history)
            current_K = self.calculate_stress_intensity_factor(max_stress)
            safety_factor = K_IC / current_K if current_K.magnitude > 0 else float('inf')
        
        return {
            "cumulative_damage": self.cumulative_damage,
            "crack_length": self.crack_length,
            "initial_crack_size": self.a_0,
            "cycle_count": self.cycle_count,
            "current_K": current_K,
            "fracture_toughness": K_IC,
            "safety_factor": safety_factor
        }
    
    def plot_stress_profiles(self, 
                           radial_positions: np.ndarray, 
                           stresses: Dict[str, np.ndarray],
                           save_path: str = None) -> plt.Figure:
        """
        Plot radial and tangential stress profiles.
        
        Args:
            radial_positions: Radial positions array [m]
            stresses: Dictionary with stress arrays from calculate_stress()
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        r_vals = np.array(radial_positions)
        if hasattr(r_vals, 'magnitude'):
            r_vals = r_vals.magnitude
        
        # Convert to μm for better readability
        r_microns = r_vals * 1e6
        
        # Get stress values and convert to MPa
        sigma_r = stresses["radial_stress"].to(ureg.megapascal).magnitude
        sigma_t = stresses["tangential_stress"].to(ureg.megapascal).magnitude
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot stresses
        ax.plot(r_microns, sigma_r, 'b-', linewidth=2, label='Radial Stress')
        ax.plot(r_microns, sigma_t, 'r-', linewidth=2, label='Tangential Stress')
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Radial Position (μm)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Stress Distribution in Electrode Particle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add max stress annotation
        max_idx = np.argmax(np.abs(sigma_t))
        max_val = sigma_t[max_idx]
        ax.annotate(f'Max: {max_val:.2f} MPa', 
                   xy=(r_microns[max_idx], max_val),
                   xytext=(r_microns[max_idx] + 0.1*r_microns[-1], max_val),
                   arrowprops=dict(arrowstyle='->'))
        
        # Tight layout
        fig.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved stress profile plot to {save_path}")
        
        return fig
    
    def plot_damage_evolution(self, save_path: str = None) -> plt.Figure:
        """
        Plot damage evolution over cycles.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.cycle_count == 0:
            logger.warning("No cycle data available for damage evolution plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No cycle data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot crack length evolution
        cycles = np.arange(1, self.cycle_count + 1)
        crack_length_microns = self.crack_length.to(ureg.micrometer).magnitude
        initial_crack_microns = self.a_0.to(ureg.micrometer).magnitude
        
        # Extrapolate crack length history (simplified)
        crack_lengths = np.linspace(initial_crack_microns, crack_length_microns, self.cycle_count)
        
        ax1.plot(cycles, crack_lengths, 'b-', linewidth=2)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Crack Length (μm)')
        ax1.set_title('Crack Growth')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative damage
        # Extrapolate damage history (simplified)
        damages = np.linspace(0, self.cumulative_damage, self.cycle_count)
        
        ax2.plot(cycles, damages, 'r-', linewidth=2)
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Cumulative Damage')
        ax2.set_title('Damage Accumulation')
        ax2.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved damage evolution plot to {save_path}")
        
        return fig
    
    @classmethod
    def from_material_properties(cls, 
                               material_name: str, 
                               particle_radius: Union[float, Quantity],
                               max_concentration: Union[float, Quantity] = None) -> 'MechanicalStressModel':
        """
        Create a stress model from predefined material properties.
        
        Args:
            material_name: Material name ('graphite', 'nmc', 'lfp', etc.)
            particle_radius: Particle radius [m]
            max_concentration: Maximum lithium concentration [mol/m³]
            
        Returns:
            Initialized MechanicalStressModel instance
        """
        # Material properties database (can be expanded)
        materials = {
            'graphite': {
                'youngs_modulus': 10e9,  # Pa
                'poissons_ratio': 0.3,
                'partial_molar_volume': 3.1e-6,  # m³/mol
                'initial_crack_size': 1e-8,  # m
                'paris_law_prefactor': 9e-12,
                'paris_law_exponent': 3.0
            },
            'nmc': {
                'youngs_modulus': 140e9,  # Pa
                'poissons_ratio': 0.3,
                'partial_molar_volume': 2.1e-6,  # m³/mol
                'initial_crack_size': 0.5e-8,  # m
                'paris_law_prefactor': 8e-13,
                'paris_law_exponent': 3.5
            },
            'lfp': {
                'youngs_modulus': 120e9,  # Pa
                'poissons_ratio': 0.28,
                'partial_molar_volume': 2.9e-6,  # m³/mol
                'initial_crack_size': 1.2e-8,  # m
                'paris_law_prefactor': 1.5e-12,
                'paris_law_exponent': 3.2
            },
            # Add more materials as needed
        }
        
        # Check if material exists
        material = materials.get(material_name.lower())
        if material is None:
            logger.warning(f"Material '{material_name}' not found in database. Using graphite properties.")
            material = materials['graphite']
        
        # Create model with material properties
        return cls(
            particle_radius=particle_radius,
            youngs_modulus=material['youngs_modulus'] * ureg.pascal,
            poissons_ratio=material['poissons_ratio'],
            partial_molar_volume=material['partial_molar_volume'] * ureg.meter**3 / ureg.mol,
            initial_crack_size=material['initial_crack_size'] * ureg.meter,
            paris_law_prefactor=material['paris_law_prefactor'],
            paris_law_exponent=material['paris_law_exponent'],
            max_concentration=max_concentration
        ) 