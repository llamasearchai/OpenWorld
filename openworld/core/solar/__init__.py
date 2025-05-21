"""
Solar cell simulation module for the OpenWorld platform.

This module provides physics-based simulations of solar photovoltaic devices, including:
- Drift-diffusion based device simulations
- Semiconductor material properties management
- Solar spectrum handling (placeholder)
- Photon absorption and carrier generation (simplified)
- Optical modeling (placeholder for advanced models)
- Device characterization (J-V curves, quantum efficiency - simplified)
- Advanced visualization (placeholder)
"""

from .material import MaterialProperties
from .device import SolarCellDevice, Layer
from .drift_diffusion import DriftDiffusionModel
from .simulation import SolarSimulation

# Common types of simulation results (can be expanded)
# from .simulation import (
#     IVSimulationResult,
#     QESimulationResult,
#     TransientSimulationResult
# )

# Visualization utilities (to be implemented)
# from .visualization import (
#     plot_jv_curve,
#     plot_band_diagram,
#     plot_carrier_densities,
#     plot_generation_profile,
#     plot_quantum_efficiency
# )

__all__ = [
    # Core models and classes
    'MaterialProperties',
    'Layer',
    'SolarCellDevice',
    'DriftDiffusionModel',
    'SolarSimulation',
    
    # Specific simulation result types (to be defined if needed)
    # 'IVSimulationResult',
    # 'QESimulationResult',
    # 'TransientSimulationResult',
    
    # Visualization functions (to be implemented)
    # 'plot_jv_curve',
    # 'plot_band_diagram',
    # 'plot_carrier_densities',
    # 'plot_generation_profile',
    # 'plot_quantum_efficiency'
] 