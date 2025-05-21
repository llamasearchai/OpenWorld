"""
Units handling utilities for OpenWorld.

This module provides a consistent way to handle physical units
throughout the OpenWorld platform using Pint.
"""

import pint
import numpy as np

# Create a unit registry to be used throughout the application
ureg = pint.UnitRegistry()

# Define commonly used units for convenience
class u:
    """Shorthand for common units."""
    # Length
    meter = ureg.meter
    m = meter
    cm = ureg.centimeter
    mm = ureg.millimeter
    um = ureg.micrometer
    nm = ureg.nanometer
    
    # Time
    second = ureg.second
    s = second
    minute = ureg.minute
    hour = ureg.hour
    
    # Mass
    kilogram = ureg.kilogram
    kg = kilogram
    gram = ureg.gram
    
    # Force
    newton = ureg.newton
    N = newton
    
    # Energy
    joule = ureg.joule
    J = joule
    electronvolt = ureg.electron_volt
    eV = electronvolt
    
    # Electric
    volt = ureg.volt
    V = volt
    ampere = ureg.ampere
    A = ampere
    ohm = ureg.ohm
    
    # Battery specific
    ampere_hour = ureg.ampere * ureg.hour
    Ah = ampere_hour
    
    # Temperature
    kelvin = ureg.kelvin
    K = kelvin
    celsius = ureg.degC
    C = celsius
    
    # Pressure
    pascal = ureg.pascal
    Pa = pascal
    
    # Derived
    watt = ureg.watt
    W = watt

# Enable pint to handle numpy arrays
ureg.setup_matplotlib()

# Register application-specific units if needed
# Example: ureg.define('soc = [] = SOC')  # State of Charge unit

def convert_to_base_units(quantity):
    """Convert a quantity to base SI units."""
    return quantity.to_base_units()

def strip_units(quantity):
    """Return magnitude of quantity without units."""
    return quantity.magnitude

def ensure_quantity(value, unit):
    """Ensure a value is a quantity with the specified unit."""
    if isinstance(value, pint.Quantity):
        return value.to(unit)
    return value * unit 