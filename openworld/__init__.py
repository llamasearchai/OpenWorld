"""
OpenWorld - Unified Physics Simulation and AI Reasoning Platform
"""

__version__ = "1.0.0"

# Import core components for easy access
from .agent import WorldModelAgent, MetaWorldModelAgent
from .core.physics import PhysicsWorld, Particle
from .core.battery import DFNModel, BatterySimulation
from .core.solar import DriftDiffusionModel, SolarSimulation
from .api import OpenWorldAPI

# Define what's available via import *
__all__ = [
    'WorldModelAgent', 'MetaWorldModelAgent',
    'PhysicsWorld', 'Particle',
    'DFNModel', 'BatterySimulation',
    'DriftDiffusionModel', 'SolarSimulation',
    'OpenWorldAPI'
] 