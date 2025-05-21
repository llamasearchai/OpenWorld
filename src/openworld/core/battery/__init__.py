"""Core battery modeling and simulation components for OpenWorld."""

# This directory will house battery-specific models, parameters, and simulation logic.
# For example:
# - Electrochemical models (e.g., DFN, SPM)
# - Thermal models
# - Degradation models
# - Parameter sets for different chemistries

# Import from moved PhysicsGPT modules
from .mechanical_stress import MechanicalStressModel

# Example (if you were to add specific battery modules):
# from .dfn_model import DFNModel
# from .battery_parameters import BatteryParametersNMC
# from .thermal_model import BatteryThermalModel

__all__ = [
    "MechanicalStressModel",
    # "DFNModel",
    # "BatteryParametersNMC",
    # "BatteryThermalModel",
] 