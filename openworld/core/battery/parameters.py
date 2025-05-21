"""
Battery parameters module for OpenWorld.

This module provides parameter handling for battery simulations, supporting
multiple chemistries and configurations with proper unit handling.
"""

import numpy as np
from dataclasses import dataclass, field, fields
from typing import Dict, Any as TypingAny, Optional, Callable, Union # Renamed Any to TypingAny
import copy # For deepcopying

from ...utils.logging import get_logger
from ...utils.units import ureg, u, convert_to_base_units, ensure_quantity # Assuming ureg and u are defined here
from ...utils.exceptions import ConfigurationError, BatteryError

logger = get_logger(__name__)

# Define some common OCV functions as placeholders
# In a real scenario, these would be more complex, often table-based or polynomial.
def ocv_graphite(stoichiometry: float, T_K: float) -> Quantity:
    """Placeholder OCV for graphite anode."""
    # Simple Nernst-like behavior, not physically accurate but provides a curve.
    U0 = 0.1 * u.volt
    return U0 + 0.05 * u.volt * np.log((1 - stoichiometry + 1e-9) / (stoichiometry + 1e-9))

def ocv_nmc(stoichiometry: float, T_K: float) -> Quantity:
    """Placeholder OCV for NMC cathode."""
    U0 = 3.7 * u.volt
    return U0 - 0.1 * u.volt * np.log((1 - stoichiometry + 1e-9) / (stoichiometry + 1e-9))

def ocv_lfp(stoichiometry: float, T_K: float) -> Quantity:
    """Placeholder OCV for LFP cathode (flat profile)."""
    return 3.4 * u.volt # LFP has a very flat OCV

@dataclass
class BatteryParameters:
    """
    Parameters for battery models including physical, thermal, and electrochemical properties.
    This class handles both dimensional and non-dimensional parameters with proper unit conversion.
    It can load predefined parameter sets for common battery chemistries.
    """
    
    param_set_name: str = "custom"
    
    # Cell Geometry & Capacity
    nominal_capacity_Ah: Quantity = field(default_factory=lambda: 3.0 * u.ampere_hour)
    initial_soc: float = 0.5 # Initial state of charge (0 to 1)
    electrode_area_cm2: Quantity = field(default_factory=lambda: 100.0 * u.cm**2) # Electrode cross-sectional area
    L_n: Quantity = field(default_factory=lambda: 75e-4 * u.cm)  # Negative electrode thickness (cm)
    L_s: Quantity = field(default_factory=lambda: 20e-4 * u.cm)   # Separator thickness (cm)
    L_p: Quantity = field(default_factory=lambda: 75e-4 * u.cm)  # Positive electrode thickness (cm)

    # Particle Properties
    R_p_n_um: Quantity = field(default_factory=lambda: 5.0 * u.micrometer)  # Negative active material particle radius (um)
    R_p_p_um: Quantity = field(default_factory=lambda: 5.0 * u.micrometer)  # Positive active material particle radius (um)
    eps_s_n: float = 0.5  # Volume fraction of solid phase in negative electrode
    eps_s_p: float = 0.5  # Volume fraction of solid phase in positive electrode
    eps_e_n: float = 0.3  # Volume fraction of electrolyte in negative electrode
    eps_e_s: float = 0.4  # Volume fraction of electrolyte in separator
    eps_e_p: float = 0.3  # Volume fraction of electrolyte in positive electrode

    # Transport Properties (Electrolyte)
    c_e_init_mol_L: Quantity = field(default_factory=lambda: 1.0 * u.mol / u.liter) # Initial electrolyte concentration (mol/L)
    D_e_cm2_s: Quantity = field(default_factory=lambda: 2.7e-6 * u.cm**2 / u.second) # Electrolyte diffusion coefficient (cm^2/s)
    t_plus: float = 0.363   # Cation transference number (Li+)
    kappa_S_cm: Quantity = field(default_factory=lambda: 0.01 * u.siemens / u.cm) # Electrolyte conductivity (S/cm)

    # Transport Properties (Solid Phase)
    D_s_n_cm2_s: Quantity = field(default_factory=lambda: 3.9e-10 * u.cm**2 / u.second) # Solid diffusion coeff in neg electrode (cm^2/s)
    D_s_p_cm2_s: Quantity = field(default_factory=lambda: 1.0e-10 * u.cm**2 / u.second) # Solid diffusion coeff in pos electrode (cm^2/s)
    sigma_n_S_cm: Quantity = field(default_factory=lambda: 1.0 * u.siemens / u.cm)    # Solid phase conductivity in neg electrode (S/cm)
    sigma_p_S_cm: Quantity = field(default_factory=lambda: 0.1 * u.siemens / u.cm)    # Solid phase conductivity in pos electrode (S/cm)

    # Stoichiometry & Concentration Limits
    c_s_n_max_mol_cm3: Quantity = field(default_factory=lambda: 3.0e-2 * u.mol / u.cm**3) # Max Li conc in neg electrode (mol/cm^3)
    c_s_p_max_mol_cm3: Quantity = field(default_factory=lambda: 5.0e-2 * u.mol / u.cm**3) # Max Li conc in pos electrode (mol/cm^3)
    theta_n_min: float = 0.01 # Min stoichiometry for negative electrode
    theta_n_max: float = 0.99 # Max stoichiometry for negative electrode
    theta_p_min: float = 0.01 # Min stoichiometry for positive electrode
    theta_p_max: float = 0.99 # Max stoichiometry for positive electrode
    
    # Derived initial stoichiometries based on initial_soc
    theta_n_init: float = field(init=False)
    theta_p_init: float = field(init=False)

    # Reaction Kinetics
    k_n_cm_s: Quantity = field(default_factory=lambda: 1.0e-7 * u.cm / u.second) # Reaction rate constant for neg electrode (cm/s)
    k_p_cm_s: Quantity = field(default_factory=lambda: 3.0e-7 * u.cm / u.second) # Reaction rate constant for pos electrode (cm/s)
    alpha_a: float = 0.5 # Anodic transfer coefficient
    alpha_c: float = 0.5 # Cathodic transfer coefficient

    # Open Circuit Voltage (OCV) - functions of stoichiometry and temperature
    ocv_function_n: Callable[[float, float], Quantity] = field(default=ocv_graphite)
    ocv_function_p: Callable[[float, float], Quantity] = field(default=ocv_nmc)

    # Thermal Parameters
    T_init_K: Quantity = field(default_factory=lambda: 298.15 * u.kelvin)  # Initial temperature (K)
    T_ambient_K: Quantity = field(default_factory=lambda: 298.15 * u.kelvin) # Ambient temperature (K)
    h_cell_W_cm2_K: Quantity = field(default_factory=lambda: 1e-3 * u.watt / (u.cm**2 * u.kelvin)) # Heat transfer coefficient (W/cm^2K)
    rho_cell_g_cm3: Quantity = field(default_factory=lambda: 2.0 * u.g / u.cm**3) # Cell density (g/cm^3)
    Cp_cell_J_g_K: Quantity = field(default_factory=lambda: 1.0 * u.joule / (u.g * u.kelvin)) # Cell specific heat capacity (J/gK)
    delta_S_n_J_mol_K: Quantity = field(default_factory=lambda: 0.0 * u.joule / (u.mol * u.kelvin)) # Entropic coefficient for neg electrode
    delta_S_p_J_mol_K: Quantity = field(default_factory=lambda: 0.0 * u.joule / (u.mol * u.kelvin)) # Entropic coefficient for pos electrode
    
    # Discretization (can be overridden by model options)
    N_x_n: int = 10  # Mesh points in negative electrode
    N_x_s: int = 5   # Mesh points in separator
    N_x_p: int = 10  # Mesh points in positive electrode
    N_r_n: int = 5   # Radial mesh points in negative particles
    N_r_p: int = 5   # Radial mesh points in positive particles

    def __post_init__(self):
        """Initialize or load parameters after construction, convert units, validate."""
        if self.param_set_name != "custom":
            self._load_parameter_set(self.param_set_name)
        
        self._convert_all_to_base_units()
        self._set_initial_stoichiometries()
        self._validate_parameters()
        logger.info(f"BatteryParameters initialized: '{self.param_set_name}'. Initial SOC: {self.initial_soc:.2%}")

    def _set_initial_stoichiometries(self):
        """Set initial stoichiometries based on initial_soc and min/max values."""
        # Positive electrode lithiates on discharge (SOC decreases, theta_p increases)
        # Negative electrode de-lithiates on discharge (SOC decreases, theta_n decreases)
        self.theta_p_init = self.theta_p_min + self.initial_soc * (self.theta_p_max - self.theta_p_min)
        self.theta_n_init = self.theta_n_max - self.initial_soc * (self.theta_n_max - self.theta_n_min)
        # Ensure they are within bounds, initial_soc should be validated before this.
        self.theta_p_init = np.clip(self.theta_p_init, self.theta_p_min, self.theta_p_max)
        self.theta_n_init = np.clip(self.theta_n_init, self.theta_n_min, self.theta_n_max)

    def _convert_all_to_base_units(self):
        """Convert all Quantity fields to their base units for internal consistency."""
        # Define target base units for categories of parameters if needed, or let pint handle it.
        # For DFN, common units are cm, s, mol, A, V, K, J, S (Siemens)
        # This ensures all calculations within models use a consistent unit system.
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Quantity):
                try:
                    # No specific base units, just convert to a consistent system if possible
                    # Pint will simplify units like Ah to C, cm^2/s to m^2/s depending on context
                    # For DFN, it's often useful to work in cm, g, s or cm, mol, s system.
                    # We assume that the default units are chosen to be sensible for DFN.
                    setattr(self, f.name, val.to_base_units()) 
                except Exception as e:
                    logger.warning(f"Could not convert {f.name} ({val}) to base units: {e}. Using as is.")
                    setattr(self, f.name, val) # Keep original if conversion fails

    def _load_parameter_set(self, param_set_name: str):
        """Load a predefined parameter set based on chemistry."""
        # This is a simplified example. A real implementation would have extensive parameter sets.
        logger.info(f"Loading parameter set: {param_set_name}")
        if param_set_name == "graphite_nmc":
            self.L_n = 75e-4 * u.cm; self.L_s = 20e-4 * u.cm; self.L_p = 75e-4 * u.cm
            self.R_p_n_um = 5 * u.um; self.R_p_p_um = 5 * u.um
            self.eps_s_n = 0.58; self.eps_s_p = 0.50
            self.eps_e_n = 0.30; self.eps_e_s = 0.40; self.eps_e_p = 0.30
            self.c_e_init_mol_L = 1.0 * u.mol / u.liter
            self.D_e_cm2_s= 2.7e-6 * u.cm**2 / u.s
            self.t_plus = 0.38
            self.kappa_S_cm = 0.01 * u.S / u.cm
            self.D_s_n_cm2_s = 3.9e-10 * u.cm**2 / u.s # Graphite
            self.D_s_p_cm2_s = 1.0e-10 * u.cm**2 / u.s # NMC
            self.sigma_n_S_cm = 1.0 * u.S / u.cm
            self.sigma_p_S_cm = 0.1 * u.S / u.cm
            self.c_s_n_max_mol_cm3 = 0.031370 * u.mol / u.cm**3 # Graphite (e.g. 2272 mAh/g, 2.2 g/cm3 -> ~0.03 mol/cm3)
            self.c_s_p_max_mol_cm3 = 0.051410 * u.mol / u.cm**3 # NMC (e.g. 160 mAh/g, 4.8 g/cm3 -> ~0.05 mol/cm3)
            self.theta_n_min = 0.01; self.theta_n_max = 0.90 # Graphite effective range
            self.theta_p_min = 0.20; self.theta_p_max = 0.98 # NMC effective range
            self.k_n_cm_s = 2e-7 * u.cm / u.s
            self.k_p_cm_s = 5e-7 * u.cm / u.s
            self.ocv_function_n = ocv_graphite
            self.ocv_function_p = ocv_nmc
        elif param_set_name == "graphite_lfp":
            self.L_n = 60e-4 * u.cm; self.L_s = 25e-4 * u.cm; self.L_p = 100e-4 * u.cm
            self.R_p_n_um = 8 * u.um; self.R_p_p_um = 0.5 * u.um # LFP particles are small
            self.eps_s_n = 0.55; self.eps_s_p = 0.45
            self.eps_e_n = 0.32; self.eps_e_s = 0.40; self.eps_e_p = 0.35
            self.D_s_n_cm2_s = 2.0e-10 * u.cm**2 / u.s # Graphite
            self.D_s_p_cm2_s = 2.0e-13 * u.cm**2 / u.s # LFP (LiFePO4 is a poor ionic conductor)
            self.c_s_n_max_mol_cm3 = 0.030 * u.mol / u.cm**3 # Graphite
            self.c_s_p_max_mol_cm3 = 0.022806 * u.mol / u.cm**3 # LFP (e.g. 170 mAh/g, 3.6 g/cm3 -> ~0.0228 mol/cm3)
            self.theta_n_min = 0.01; self.theta_n_max = 0.85
            self.theta_p_min = 0.02; self.theta_p_max = 0.98 # LFP has wide flat plateau
            self.k_n_cm_s = 1.5e-7 * u.cm / u.s
            self.k_p_cm_s = 8e-8 * u.cm / u.s # LFP kinetics can be slower
            self.ocv_function_n = ocv_graphite
            self.ocv_function_p = ocv_lfp
        else:
            logger.warning(f"Unknown predefined parameter set: '{param_set_name}'. Using default/custom values.")
            self.param_set_name = "custom" # Revert to custom if set not found

    def _validate_parameters(self):
        """Validate parameters for consistency and physical plausibility."""
        if not (0.0 <= self.initial_soc <= 1.0):
            raise ConfigurationError(f"Initial SOC must be between 0 and 1, got {self.initial_soc}")
        # Volume fractions
        if not (0 < self.eps_s_n < 1 and 0 < self.eps_e_n < 1 and (self.eps_s_n + self.eps_e_n) < 1.0):
            raise ConfigurationError("Invalid volume fractions for negative electrode.")
        if not (0 < self.eps_s_p < 1 and 0 < self.eps_e_p < 1 and (self.eps_s_p + self.eps_e_p) < 1.0):
            raise ConfigurationError("Invalid volume fractions for positive electrode.")
        if not (0 < self.eps_e_s < 1):
            raise ConfigurationError("Invalid electrolyte volume fraction for separator.")
        # Stoichiometry limits
        if not (0.0 <= self.theta_n_min < self.theta_n_max <= 1.0):
            raise ConfigurationError("Invalid stoichiometry range for negative electrode.")
        if not (0.0 <= self.theta_p_min < self.theta_p_max <= 1.0):
            raise ConfigurationError("Invalid stoichiometry range for positive electrode.")
        # Check that initial stoichiometries (derived from SOC) are within their respective limits
        if not (self.theta_n_min <= self.theta_n_init <= self.theta_n_max):
             logger.warning(f"Initial negative stoichiometry {self.theta_n_init:.3f} is outside bounds [{self.theta_n_min:.3f}, {self.theta_n_max:.3f}]. Clamping.")
             self.theta_n_init = np.clip(self.theta_n_init, self.theta_n_min, self.theta_n_max)
        if not (self.theta_p_min <= self.theta_p_init <= self.theta_p_max):
             logger.warning(f"Initial positive stoichiometry {self.theta_p_init:.3f} is outside bounds [{self.theta_p_min:.3f}, {self.theta_p_max:.3f}]. Clamping.")
             self.theta_p_init = np.clip(self.theta_p_init, self.theta_p_min, self.theta_p_max)
        
        # Check positive definiteness of quantities that should be positive
        for f_name in ["nominal_capacity_Ah", "electrode_area_cm2", 
                       "L_n", "L_s", "L_p", "R_p_n_um", "R_p_p_um",
                       "D_e_cm2_s", "kappa_S_cm", "D_s_n_cm2_s", "D_s_p_cm2_s",
                       "sigma_n_S_cm", "sigma_p_S_cm", "c_s_n_max_mol_cm3", "c_s_p_max_mol_cm3",
                       "k_n_cm_s", "k_p_cm_s", "T_init_K", "T_ambient_K"]:
            val = getattr(self, f_name)
            if isinstance(val, Quantity) and val.magnitude <= 0:
                raise ConfigurationError(f"Parameter '{f_name}' must be positive, got {val}.")
            elif isinstance(val, (int, float)) and val <=0: # For plain numbers like epsilons
                 pass # Already checked for epsilons, N_x etc.

    def scale_capacity(self, new_capacity_Ah: float):
        """
        Scale the battery capacity. This typically involves scaling electrode area or thickness.
        Here, we scale electrode_area_cm2, assuming thicknesses remain constant.
        
        Args:
            new_capacity_Ah: New target capacity in Ampere-hours.
        """
        if new_capacity_Ah <= 0:
            raise ValueError("New capacity must be positive.")
        
        current_cap_Ah = self.nominal_capacity_Ah.to(u.Ah).magnitude
        if current_cap_Ah == 0:
            raise BatteryError("Cannot scale capacity from zero. Set a non-zero initial capacity.")
            
        scaling_factor = new_capacity_Ah / current_cap_Ah
        
        self.nominal_capacity_Ah = new_capacity_Ah * u.Ah
        self.electrode_area_cm2 *= scaling_factor
        self._convert_all_to_base_units() # Re-convert after modification
        logger.info(f"Battery capacity scaled to {self.nominal_capacity_Ah.to_compact()}. Electrode area scaled to {self.electrode_area_cm2.to_compact()}.")

    def adjust_initial_soc(self, new_soc: float):
        """
        Adjust the initial state of charge (SOC) and recalculate initial stoichiometries.
        
        Args:
            new_soc: Target SOC (0 to 1).
        """
        if not (0.0 <= new_soc <= 1.0):
            raise ConfigurationError(f"Target SOC must be between 0 and 1, got {new_soc}")
        
        self.initial_soc = new_soc
        self._set_initial_stoichiometries() # Recalculate based on new SOC
        self._validate_parameters() # Re-validate, especially the initial thetas
        logger.info(f"Initial SOC adjusted to {self.initial_soc:.2%}. Theta_n_init={self.theta_n_init:.3f}, Theta_p_init={self.theta_p_init:.3f}")

    def to_dict(self, strip_decorators: bool = True) -> Dict[str, TypingAny]:
        """Convert parameters to a dictionary, optionally stripping pint @decorators for plain magnitudes."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Quantity):
                # Store as dict with magnitude and unit string
                d[f.name] = {"magnitude": val.magnitude, "unit": str(val.units)}
            elif callable(val) and val.__name__.startswith("ocv_"):
                 d[f.name] = val.__name__ # Store function name
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, TypingAny]) -> 'BatteryParameters':
        """Create a BatteryParameters instance from a dictionary."""
        # Create a mutable copy of the input data
        init_data = copy.deepcopy(data)
        
        # Convert unit dicts back to Quantity objects
        for key, value in init_data.items():
            if isinstance(value, dict) and "magnitude" in value and "unit" in value:
                try:
                    init_data[key] = value["magnitude"] * ureg(value["unit"])
                except Exception as e:
                    logger.error(f"Error parsing quantity from dict for '{key}': {value}. Error: {e}. Using magnitude only or default.")
                    init_data[key] = value["magnitude"] # Fallback to magnitude if unit parsing fails
            elif isinstance(value, str) and value.startswith("ocv_"):
                # Attempt to map OCV function names back to functions
                ocv_func_map = {
                    "ocv_graphite": ocv_graphite,
                    "ocv_nmc": ocv_nmc,
                    "ocv_lfp": ocv_lfp
                }
                if value in ocv_func_map:
                    init_data[key] = ocv_func_map[value]
                else:
                    logger.warning(f"Unknown OCV function name '{value}' for key '{key}'. Using default.")
                    # Let dataclass use its default if key is missing or value is problematic
                    if key in init_data: del init_data[key]

        # Filter out keys not in dataclass fields to prevent TypeError on __init__
        valid_field_names = {f.name for f in fields(cls)}
        filtered_init_data = {k: v for k, v in init_data.items() if k in valid_field_names}
        
        try:
            instance = cls(**filtered_init_data)
        except TypeError as e:
            logger.error(f"Error creating BatteryParameters from dict: {e}. Data provided: {filtered_init_data}")            
            # Fallback to default instance if dict loading fails significantly
            logger.warning("Falling back to default BatteryParameters due to instantiation error.")
            instance = cls()
            # Attempt to apply whatever valid data was passed, if possible
            for key, value in filtered_init_data.items():
                if hasattr(instance, key):
                    try:
                        setattr(instance, key, value)
                    except: # Broad except for setattr issues
                        pass 
            instance.__post_init__() # Ensure post_init runs even on fallback
        return instance 