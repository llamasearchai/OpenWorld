"""Core Physics Simulation components for OpenWorld."""

from .engine import PhysicsEngine # Expose the main engine
# from ..config import PhysicsConfig # Config is usually accessed via a global settings object or passed during instantiation

# Placeholder for other core physics concepts if they become separate files
# For example:
# from .particle import Particle, ParticleSystem
# from .continuum import ContinuumMechanics
# from .electromagnetism import EMField
# from .quantum import QuantumSystem

__all__ = [
    "PhysicsEngine",
    # "PhysicsConfig", # Usually not exposed here directly
    # "Particle",
    # "ParticleSystem",
] 