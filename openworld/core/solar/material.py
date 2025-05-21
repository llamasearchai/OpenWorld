"""
Material properties for solar cell simulations in OpenWorld.

This module provides classes and functions for defining material
properties needed for solar cell drift-diffusion simulations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Callable
import numpy as np
import copy

from ...utils.logging import get_logger
from ...utils.units import ureg, u

logger = get_logger(__name__)

@dataclass
class MaterialProperties:
    """
    Class representing material properties for semiconductor simulation.
    
    This contains all the physical parameters needed for drift-diffusion
    simulations of semiconductor devices.
    """
    
    # Basic properties
    name: str
    band_gap: float  # eV
    electron_affinity: float  # eV
    epsilon_r: float  # Relative permittivity
    
    # Transport properties
    mu_n: float  # Electron mobility [cm²/(V·s)]
    mu_p: float  # Hole mobility [cm²/(V·s)]
    
    # Recombination properties
    tau_n: float = 1e-6  # Electron lifetime [s]
    tau_p: float = 1e-6  # Hole lifetime [s]
    
    # Intrinsic properties
    n_i: float = 1e10  # Intrinsic carrier concentration [cm⁻³]
    
    # Doping properties
    doping_type: Optional[str] = None  # 'n' or 'p'
    doping_concentration: float = 0.0  # [cm⁻³]
    
    # Optical properties
    absorption_coefficient: Optional[Union[np.ndarray, List[float]]] = None
    absorption_wavelengths: Optional[Union[np.ndarray, List[float]]] = None
    
    # Recombination mechanisms
    auger_coefficient_n: float = 1e-30  # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p: float = 1e-30  # Auger coefficient for holes [cm⁶/s]
    radiative_coefficient: float = 1e-10  # Radiative recombination coefficient [cm³/s]
    
    # Interface properties
    surface_recombination_velocity_n: float = 1e3  # SRV for electrons [cm/s]
    surface_recombination_velocity_p: float = 1e3  # SRV for holes [cm/s]
    
    # Thermal properties
    thermal_conductivity: float = 1.0  # [W/(m·K)]
    heat_capacity: float = 1.0  # [J/(g·K)]
    
    # Temperature dependence (optional functions)
    bandgap_temperature_function: Optional[Callable[[float], float]] = None
    mobility_temperature_function_n: Optional[Callable[[float], float]] = None
    mobility_temperature_function_p: Optional[Callable[[float], float]] = None
    
    # Additional properties (for extensibility)
    additional_properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the provided properties."""
        # Basic validation
        if self.band_gap <= 0:
            raise ValueError(f"Band gap must be positive, got {self.band_gap}")
        
        if self.epsilon_r <= 0:
            raise ValueError(f"Relative permittivity must be positive, got {self.epsilon_r}")
        
        if self.mu_n <= 0:
            raise ValueError(f"Electron mobility must be positive, got {self.mu_n}")
        
        if self.mu_p <= 0:
            raise ValueError(f"Hole mobility must be positive, got {self.mu_p}")
        
        # Optional parameter validation
        if self.doping_type is not None and self.doping_type not in ('n', 'p'):
            raise ValueError(f"Doping type must be 'n' or 'p', got {self.doping_type}")
        
        if self.doping_concentration < 0:
            raise ValueError(f"Doping concentration cannot be negative, got {self.doping_concentration}")
        
        # Convert absorption coefficient data to numpy arrays if needed
        if self.absorption_coefficient is not None and not isinstance(self.absorption_coefficient, np.ndarray):
            self.absorption_coefficient = np.array(self.absorption_coefficient)
            
        if self.absorption_wavelengths is not None and not isinstance(self.absorption_wavelengths, np.ndarray):
            self.absorption_wavelengths = np.array(self.absorption_wavelengths)
            
        # Verify matching lengths for optical properties
        if (self.absorption_coefficient is not None and self.absorption_wavelengths is not None and
                len(self.absorption_coefficient) != len(self.absorption_wavelengths)):
            raise ValueError(f"Absorption coefficient and wavelengths arrays must have the same length, "
                             f"got {len(self.absorption_coefficient)} and {len(self.absorption_wavelengths)}")
    
    def get_work_function(self) -> float:
        """
        Calculate the work function of the material.
        
        For semiconductors, this is related to electron affinity, band gap, and doping.
        
        Returns:
            Work function in eV
        """
        # Simple approximation based on electron affinity and band gap
        if self.doping_type == 'n':
            # n-type material: work function is close to electron affinity
            work_function = self.electron_affinity + 0.1  # Simplified approximation
        elif self.doping_type == 'p':
            # p-type material: work function is closer to ionization energy (electron affinity + band gap)
            work_function = self.electron_affinity + self.band_gap - 0.1  # Simplified approximation
        else:
            # Intrinsic material: work function is roughly mid-gap
            work_function = self.electron_affinity + self.band_gap / 2
        
        return work_function
    
    def get_debye_length(self, temperature_K: float = 300.0) -> float:
        """
        Calculate the Debye length in the material.
        
        Args:
            temperature_K: Temperature in Kelvin
            
        Returns:
            Debye length in cm
        """
        if self.doping_concentration == 0:
            # For intrinsic material, use intrinsic carrier concentration
            carrier_concentration = self.n_i
        else:
            # For doped material, use doping concentration
            carrier_concentration = self.doping_concentration
        
        # Thermal voltage in eV
        k_B = 8.617333262e-5  # Boltzmann constant in eV/K
        thermal_voltage = k_B * temperature_K
        
        # Elementary charge in C
        q = 1.602176634e-19
        
        # Permittivity of free space in F/cm
        epsilon_0 = 8.8541878128e-14
        
        # Calculate Debye length in cm
        debye_length = np.sqrt((self.epsilon_r * epsilon_0 * thermal_voltage) / 
                              (q * carrier_concentration))
        
        return debye_length
    
    def get_diffusion_length(self, carrier_type: str = 'minority', temperature_K: float = 300.0) -> float:
        """
        Calculate the diffusion length for specified carrier type.
        
        Args:
            carrier_type: 'minority', 'electron', or 'hole'
            temperature_K: Temperature in Kelvin
            
        Returns:
            Diffusion length in cm
        """
        # Determine which carrier is of interest
        if carrier_type == 'minority':
            if self.doping_type == 'n':
                # In n-type, holes are minority carriers
                mobility = self.mu_p
                lifetime = self.tau_p
            elif self.doping_type == 'p':
                # In p-type, electrons are minority carriers
                mobility = self.mu_n
                lifetime = self.tau_n
            else:
                # In intrinsic material, use average
                mobility = (self.mu_n + self.mu_p) / 2
                lifetime = (self.tau_n + self.tau_p) / 2
        elif carrier_type == 'electron':
            mobility = self.mu_n
            lifetime = self.tau_n
        elif carrier_type == 'hole':
            mobility = self.mu_p
            lifetime = self.tau_p
        else:
            raise ValueError(f"Invalid carrier type: {carrier_type}. Must be 'minority', 'electron', or 'hole'")
        
        # Apply temperature dependence if function is provided
        if carrier_type == 'electron' or (carrier_type == 'minority' and self.doping_type == 'p'):
            if self.mobility_temperature_function_n is not None:
                mobility = self.mobility_temperature_function_n(temperature_K)
        elif carrier_type == 'hole' or (carrier_type == 'minority' and self.doping_type == 'n'):
            if self.mobility_temperature_function_p is not None:
                mobility = self.mobility_temperature_function_p(temperature_K)
        
        # Calculate diffusion coefficient (Einstein relation)
        k_B = 8.617333262e-5  # Boltzmann constant in eV/K
        thermal_voltage = k_B * temperature_K
        
        # Diffusion coefficient in cm²/s
        diffusion_coefficient = mobility * thermal_voltage
        
        # Diffusion length in cm
        diffusion_length = np.sqrt(diffusion_coefficient * lifetime)
        
        return diffusion_length
    
    def get_absorption_at_wavelength(self, wavelength_nm: float) -> float:
        """
        Get absorption coefficient at a specific wavelength.
        
        Args:
            wavelength_nm: Wavelength in nanometers
            
        Returns:
            Absorption coefficient in cm⁻¹
        """
        if self.absorption_coefficient is None or self.absorption_wavelengths is None:
            # If no absorption data available, estimate using band gap
            # This is a very rough approximation
            photon_energy_eV = 1240 / wavelength_nm  # E = hc/λ
            
            if photon_energy_eV < self.band_gap:
                # Below band gap - minimal absorption
                return 1e2  # Residual absorption, 1/cm
            else:
                # Above band gap - absorption increases with energy
                excess_energy = photon_energy_eV - self.band_gap
                # Simple power law model, very approximate
                return 1e4 * (1 + excess_energy)  # 1/cm
        
        else:
            # Interpolate from provided data
            if wavelength_nm < np.min(self.absorption_wavelengths) or wavelength_nm > np.max(self.absorption_wavelengths):
                # Return closest value if outside range
                idx = np.argmin(np.abs(self.absorption_wavelengths - wavelength_nm))
                return self.absorption_coefficient[idx]
            else:
                # Linear interpolation
                idx_low = np.max(np.where(self.absorption_wavelengths <= wavelength_nm)[0])
                idx_high = np.min(np.where(self.absorption_wavelengths >= wavelength_nm)[0])
                
                if idx_low == idx_high:
                    return self.absorption_coefficient[idx_low]
                
                # Interpolate
                wl_low = self.absorption_wavelengths[idx_low]
                wl_high = self.absorption_wavelengths[idx_high]
                abs_low = self.absorption_coefficient[idx_low]
                abs_high = self.absorption_coefficient[idx_high]
                
                # Linear interpolation
                return abs_low + (abs_high - abs_low) * (wavelength_nm - wl_low) / (wl_high - wl_low)
    
    def get_recombination_rate(self, n: float, p: float, n_i: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate recombination rates for different mechanisms.
        
        Args:
            n: Electron concentration [cm⁻³]
            p: Hole concentration [cm⁻³]
            n_i: Intrinsic carrier concentration [cm⁻³] (uses self.n_i if None)
            
        Returns:
            Dictionary with recombination rates for different mechanisms [cm⁻³/s]
        """
        if n_i is None:
            n_i = self.n_i
            
        # SRH recombination (uses lifetimes)
        n0 = p0 = n_i  # Simplified - trap level at intrinsic level
        r_srh = (n * p - n_i**2) / (self.tau_p * (n + n0) + self.tau_n * (p + p0))
        
        # Radiative recombination
        r_rad = self.radiative_coefficient * (n * p - n_i**2)
        
        # Auger recombination
        r_auger = (self.auger_coefficient_n * n + self.auger_coefficient_p * p) * (n * p - n_i**2)
        
        # Total recombination rate
        r_total = r_srh + r_rad + r_auger
        
        return {
            "SRH": r_srh,
            "radiative": r_rad,
            "Auger": r_auger,
            "total": r_total
        }
    
    def get_bandgap_at_temperature(self, temperature_K: float = 300.0) -> float:
        """
        Get the band gap of the material at a specific temperature.
        
        Args:
            temperature_K: Temperature in Kelvin
            
        Returns:
            Band gap in eV
        """
        if self.bandgap_temperature_function is not None:
            return self.bandgap_temperature_function(temperature_K)
        else:
            # Default Varshni formula for temperature dependence if no custom function
            # Parameters for silicon are used as default
            alpha = 4.73e-4  # eV/K
            beta = 636.0     # K
            return self.band_gap - (alpha * temperature_K**2) / (temperature_K + beta)
    
    def copy_with_modifications(self, **kwargs) -> 'MaterialProperties':
        """
        Create a new MaterialProperties instance with modifications.
        
        Args:
            **kwargs: Parameters to update in the new instance
            
        Returns:
            New MaterialProperties instance with updated parameters
        """
        # Create a deep copy of the current instance
        new_properties = copy.deepcopy(self)
        
        # Update with specified parameters
        for key, value in kwargs.items():
            if hasattr(new_properties, key):
                setattr(new_properties, key, value)
            else:
                new_properties.additional_properties[key] = value
        
        # Re-run validation
        new_properties.__post_init__()
        
        return new_properties


# Common semiconductor materials (with enhanced parameters)
SILICON = MaterialProperties(
    name="Silicon",
    band_gap=1.12,            # eV
    electron_affinity=4.05,   # eV
    epsilon_r=11.7,           # Relative permittivity
    mu_n=1400,                # Electron mobility [cm²/(V·s)]
    mu_p=450,                 # Hole mobility [cm²/(V·s)]
    n_i=1e10,                 # Intrinsic carrier concentration [cm⁻³]
    tau_n=1e-6,               # Electron lifetime [s]
    tau_p=1e-6,               # Hole lifetime [s]
    radiative_coefficient=1e-14,  # Radiative recombination coefficient [cm³/s]
    auger_coefficient_n=2.8e-31,  # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p=9.9e-32,  # Auger coefficient for holes [cm⁶/s]
    thermal_conductivity=149.0,   # [W/(m·K)]
    heat_capacity=0.7         # [J/(g·K)]
)

GALLIUM_ARSENIDE = MaterialProperties(
    name="GaAs",
    band_gap=1.42,            # eV
    electron_affinity=4.07,   # eV
    epsilon_r=12.9,           # Relative permittivity
    mu_n=8500,                # Electron mobility [cm²/(V·s)]
    mu_p=400,                 # Hole mobility [cm²/(V·s)]
    n_i=2e6,                  # Intrinsic carrier concentration [cm⁻³]
    tau_n=1e-8,               # Electron lifetime [s]
    tau_p=1e-8,               # Hole lifetime [s]
    radiative_coefficient=7e-10,  # Radiative recombination coefficient [cm³/s]
    auger_coefficient_n=5e-30,    # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p=5e-30,    # Auger coefficient for holes [cm⁶/s]
    thermal_conductivity=55.0,    # [W/(m·K)]
    heat_capacity=0.33        # [J/(g·K)]
)

CADMIUM_TELLURIDE = MaterialProperties(
    name="CdTe",
    band_gap=1.5,             # eV
    electron_affinity=4.28,   # eV
    epsilon_r=10.2,           # Relative permittivity
    mu_n=1050,                # Electron mobility [cm²/(V·s)]
    mu_p=100,                 # Hole mobility [cm²/(V·s)]
    n_i=8e5,                  # Intrinsic carrier concentration [cm⁻³]
    tau_n=1e-9,               # Electron lifetime [s]
    tau_p=1e-9,               # Hole lifetime [s]
    radiative_coefficient=1e-10,  # Radiative recombination coefficient [cm³/s]
    auger_coefficient_n=1e-29,    # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p=1e-29,    # Auger coefficient for holes [cm⁶/s]
    thermal_conductivity=6.0,     # [W/(m·K)]
    heat_capacity=0.21        # [J/(g·K)]
)

CIGS = MaterialProperties(
    name="CIGS",
    band_gap=1.15,            # eV (can vary from 1.0-1.7 depending on composition)
    electron_affinity=4.3,    # eV
    epsilon_r=13.6,           # Relative permittivity
    mu_n=100,                 # Electron mobility [cm²/(V·s)]
    mu_p=25,                  # Hole mobility [cm²/(V·s)]
    n_i=2e9,                  # Intrinsic carrier concentration [cm⁻³]
    tau_n=1e-9,               # Electron lifetime [s]
    tau_p=1e-9,               # Hole lifetime [s]
    radiative_coefficient=5e-11,  # Radiative recombination coefficient [cm³/s]
    auger_coefficient_n=1e-29,    # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p=1e-29,    # Auger coefficient for holes [cm⁶/s]
    thermal_conductivity=4.0,     # [W/(m·K)]
    heat_capacity=0.3         # [J/(g·K)]
)

PEROVSKITE = MaterialProperties(
    name="Perovskite",
    band_gap=1.55,            # eV (typical value, can vary)
    electron_affinity=3.9,    # eV
    epsilon_r=24.1,           # Relative permittivity
    mu_n=1.6,                 # Electron mobility [cm²/(V·s)]
    mu_p=0.8,                 # Hole mobility [cm²/(V·s)]
    n_i=1e9,                  # Intrinsic carrier concentration [cm⁻³]
    tau_n=1e-7,               # Electron lifetime [s]
    tau_p=1e-7,               # Hole lifetime [s]
    radiative_coefficient=1e-9,   # Radiative recombination coefficient [cm³/s]
    auger_coefficient_n=1e-28,    # Auger coefficient for electrons [cm⁶/s]
    auger_coefficient_p=1e-28,    # Auger coefficient for holes [cm⁶/s]
    thermal_conductivity=0.5,     # [W/(m·K)]
    heat_capacity=0.27        # [J/(g·K)]
)

# Dictionary of common materials for easy access
COMMON_MATERIALS = {
    "silicon": SILICON,
    "gaas": GALLIUM_ARSENIDE,
    "cdte": CADMIUM_TELLURIDE,
    "cigs": CIGS,
    "perovskite": PEROVSKITE
} 