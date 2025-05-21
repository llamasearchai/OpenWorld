"""
Particle class representing a point mass in the physics simulation.
"""

import numpy as np
from pint import Quantity
from typing import Any, Dict, Optional, List, Union, Tuple

from ...utils.logging import get_logger
from ...utils.units import ureg, u
from ..base.physical_object import PhysicalObject

logger = get_logger(__name__)

class Particle(PhysicalObject):
    """
    A particle representing a point mass with position, velocity, and forces.
    
    This class implements a simple particle model with various integration methods.
    """
    
    def __init__(self, 
                position: Any,
                mass: Any,
                velocity: Optional[Any] = None,
                radius: Optional[Any] = None,  # For visualization and collision
                charge: Optional[Any] = None,  # For EM forces
                obj_id: Optional[str] = None,
                tags: Optional[List[str]] = None,
                integration_method: str = "verlet",
                default_pos_unit: str = u.meter,
                default_mass_unit: str = u.kilogram,
                default_vel_unit: Optional[str] = None):
        """
        Initialize a particle.
        
        Args:
            position: Initial position vector
            mass: Particle mass
            velocity: Initial velocity vector (default: zero)
            radius: Particle radius (default: 0.1 meters)
            charge: Particle electric charge (default: 0 coulombs)
            obj_id: Optional ID (randomly generated if not provided)
            tags: Optional tags for categorizing/grouping particles
            integration_method: Numerical method for updating position/velocity
            default_pos_unit: Default unit for position
            default_mass_unit: Default unit for mass
            default_vel_unit: Default unit for velocity
        """
        super().__init__(
            position=position,
            mass=mass,
            velocity=velocity,
            obj_id=obj_id,
            tags=tags,
            default_pos_unit=default_pos_unit,
            default_mass_unit=default_mass_unit,
            default_vel_unit=default_vel_unit
        )
        
        # Set particle-specific properties
        default_radius = 0.1 * ureg.meter if radius is None else radius
        self.add_property("radius", ureg.Quantity(default_radius, ureg.meter))
        
        default_charge = 0.0 * ureg.coulomb if charge is None else charge
        self.add_property("charge", ureg.Quantity(default_charge, ureg.coulomb))
        
        # Set integration method
        valid_methods = ["euler", "verlet", "rk4"]
        if integration_method not in valid_methods:
            logger.warning(f"Invalid integration method '{integration_method}'. Using 'verlet' instead.")
            self._integration_method = "verlet"
        else:
            self._integration_method = integration_method
        
        # For integration methods that need history
        self._prev_acceleration = None
        
        logger.debug(f"Created particle {self.id} with integration method: {self._integration_method}")
    
    @property
    def radius(self) -> Quantity:
        """Particle radius."""
        return self.get_property("radius")
    
    @radius.setter
    def radius(self, value: Any):
        """Set particle radius."""
        self.add_property("radius", ureg.Quantity(value, ureg.meter))
    
    @property
    def charge(self) -> Quantity:
        """Particle electric charge."""
        return self.get_property("charge")
    
    @charge.setter
    def charge(self, value: Any):
        """Set particle electric charge."""
        self.add_property("charge", ureg.Quantity(value, ureg.coulomb))
    
    def update_state(self, acceleration: Quantity, dt: Quantity):
        """
        Update position and velocity based on current acceleration.
        
        Implements several integration methods for updating the particle state.
        
        Args:
            acceleration: Current acceleration vector
            dt: Time step duration
        """
        dt_sec = dt.to(ureg.second).magnitude
        
        # Ensure acceleration has compatible units with velocity/time
        accel_units = self.velocity.units / ureg.second
        acceleration = acceleration.to(accel_units)
        
        # Choose integration method
        if self._integration_method == "euler":
            # Basic Euler integration (least accurate)
            self.velocity = self.velocity + acceleration * dt
            self.position = self.position + self.velocity * dt
            
        elif self._integration_method == "verlet":
            # Velocity Verlet integration (good for energy conservation)
            # Position update
            self.position = self.position + self.velocity * dt + 0.5 * acceleration * dt**2
            
            # Store current acceleration for next step
            if self._prev_acceleration is None:
                # First step, just use current acceleration
                new_velocity = self.velocity + acceleration * dt
            else:
                # Velocity verlet update using previous and current acceleration
                new_velocity = self.velocity + 0.5 * (self._prev_acceleration + acceleration) * dt
            
            self.velocity = new_velocity
            self._prev_acceleration = acceleration
            
        elif self._integration_method == "rk4":
            # 4th-order Runge-Kutta method (most accurate but expensive)
            # This is a simplification since we're not recalculating forces/accelerations
            # A full implementation would need access to the force calculation
            
            pos_0 = self.position
            vel_0 = self.velocity
            
            # K1
            k1_v = acceleration * dt_sec
            k1_x = vel_0 * dt_sec
            
            # K2 (midpoint)
            vel_mid = vel_0 + 0.5 * k1_v
            k2_v = acceleration * dt_sec  # Simplified - should use updated force
            k2_x = vel_mid * dt_sec
            
            # K3 (midpoint)
            vel_mid2 = vel_0 + 0.5 * k2_v
            k3_v = acceleration * dt_sec  # Simplified - should use updated force
            k3_x = vel_mid2 * dt_sec
            
            # K4 (endpoint)
            vel_end = vel_0 + k3_v
            k4_v = acceleration * dt_sec  # Simplified - should use updated force
            k4_x = vel_end * dt_sec
            
            # Final updates
            self.velocity = vel_0 + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
            self.position = pos_0 + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
        
        else:
            # Default to Euler if method not recognized
            self.velocity = self.velocity + acceleration * dt
            self.position = self.position + self.velocity * dt
    
    def get_state_snapshot(self, time: Quantity) -> Dict[str, Any]:
        """
        Return a particle-specific state snapshot.
        
        Args:
            time: Current simulation time
            
        Returns:
            Dictionary with state information
        """
        # Start with base state
        base_state = super().get_state_snapshot(time).dict()
        
        # Add particle-specific properties
        particle_state = {
            **base_state,
            "radius": self.radius.to(ureg.meter).magnitude,
            "charge": self.charge.to(ureg.coulomb).magnitude,
            "integration_method": self._integration_method
        }
        
        return particle_state 