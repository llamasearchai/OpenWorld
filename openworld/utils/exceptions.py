"""
Custom exceptions for the OpenWorld platform.

This module defines the exception hierarchy used throughout the platform.
"""

class OpenWorldError(Exception):
    """Base class for all OpenWorld exceptions."""
    pass

class ConfigurationError(OpenWorldError):
    """Error in configuration settings."""
    pass

class SimulationError(OpenWorldError):
    """Error during simulation execution."""
    pass

class APIError(OpenWorldError):
    """Error in API operation."""
    pass

class AIError(OpenWorldError):
    """Error in AI reasoning or model execution."""
    pass

class ValidationError(OpenWorldError):
    """Error in data validation."""
    pass

class PhysicsError(SimulationError):
    """Error in physics simulation."""
    pass

class BatteryError(SimulationError):
    """Error in battery simulation."""
    pass

class SolarError(SimulationError):
    """Error in solar simulation."""
    pass

class NumericalError(SimulationError):
    """Error in numerical computation."""
    pass 