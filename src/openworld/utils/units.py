"""
Units module for PhysicsGPT.

This module provides consistent unit handling across the codebase using the pint library.
It defines standard units and conversions for physical quantities used in simulations.
"""

import pint

# Initialize unit registry
ureg = pint.UnitRegistry()
ureg.setup_matplotlib()

# Define convenient quantities
Q_ = ureg.Quantity

# Define standard units for different domains
# Battery units
class BatteryUnits:
    # Capacity
    Ah = ureg.ampere_hour
    mAh = ureg.milliampere_hour
    
    # Voltage, current
    V = ureg.volt
    A = ureg.ampere
    
    # Time
    h = ureg.hour
    min = ureg.minute
    s = ureg.second
    
    # Concentration
    mol_m3 = ureg.mol / ureg.meter**3
    
    # Diffusion coefficient
    m2_s = ureg.meter**2 / ureg.second
    
    # Stress
    Pa = ureg.pascal
    MPa = ureg.megapascal
    
    # Temperature
    K = ureg.kelvin
    C = ureg.celsius
    
    # Geometry
    m = ureg.meter
    cm = ureg.centimeter
    mm = ureg.millimeter
    um = ureg.micrometer
    nm = ureg.nanometer
    
    # Energy
    J = ureg.joule
    kJ = ureg.kilojoule
    Wh = ureg.watt * ureg.hour

# Solar cell units
class SolarUnits:
    # Current density
    mA_cm2 = ureg.milliampere / ureg.centimeter**2
    
    # Length
    nm = ureg.nanometer
    um = ureg.micrometer
    
    # Energy
    eV = ureg.electron_volt
    
    # Band gap related
    kb_eV = 8.617333262e-5 * ureg.electron_volt / ureg.kelvin  # Boltzmann in eV/K
    
    # Efficiency
    percent = ureg.percent
    
    # Illumination
    sun = ureg.dimensionless  # 1 sun = standard AM1.5G illumination
    
    # Mobility
    cm2_Vs = ureg.centimeter**2 / (ureg.volt * ureg.second)
    
    # Carrier concentration
    cm_minus3 = 1 / ureg.centimeter**3
    
    # Carrier lifetime
    ns = ureg.nanosecond
    us = ureg.microsecond

# Physics units
class PhysicsUnits:
    # Mechanics
    N = ureg.newton
    kg = ureg.kilogram
    g = ureg.gram
    
    # Time
    s = ureg.second
    ms = ureg.millisecond
    
    # Length
    m = ureg.meter
    cm = ureg.centimeter
    mm = ureg.millimeter
    
    # Angle
    rad = ureg.radian
    deg = ureg.degree
    
    # Energy
    J = ureg.joule
    kJ = ureg.kilojoule
    
    # Velocity, acceleration
    m_s = ureg.meter / ureg.second
    m_s2 = ureg.meter / ureg.second**2
    
    # Pressure
    Pa = ureg.pascal
    kPa = ureg.kilopascal
    atm = ureg.atmosphere

# Get dimensionless value from quantity
def get_value(quantity):
    """
    Extract dimensionless value from a Pint quantity.
    
    Args:
        quantity: Pint quantity or regular number
        
    Returns:
        Dimensionless value (float or numpy array)
    """
    if isinstance(quantity, ureg.Quantity):
        return quantity.magnitude
    return quantity

# Convert a value to specified units
def convert_to(quantity, target_unit):
    """
    Convert a quantity to the specified units.
    
    Args:
        quantity: Pint quantity to convert
        target_unit: Target unit to convert to
        
    Returns:
        Converted quantity
    """
    if isinstance(quantity, ureg.Quantity):
        return quantity.to(target_unit)
    else:
        # If not a quantity, wrap it with target unit
        return quantity * target_unit 