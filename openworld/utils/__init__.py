"""
Utility modules for OpenWorld.

This package contains utility modules used throughout the OpenWorld platform.
"""

from .units import ureg, u, convert_to_base_units, strip_units, ensure_quantity
from .logging import get_logger, configure_logging
from .exceptions import (
    OpenWorldError, ConfigurationError, SimulationError, 
    APIError, AIError, ValidationError,
    PhysicsError, BatteryError, SolarError, NumericalError
)

__all__ = [
    # Units
    'ureg', 'u', 'convert_to_base_units', 'strip_units', 'ensure_quantity',
    
    # Logging
    'get_logger', 'configure_logging',
    
    # Exceptions
    'OpenWorldError', 'ConfigurationError', 'SimulationError',
    'APIError', 'AIError', 'ValidationError',
    'PhysicsError', 'BatteryError', 'SolarError', 'NumericalError'
] 