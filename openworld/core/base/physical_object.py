"""
Physical object class for OpenWorld physics simulations.

This module defines the PhysicalObject class which represents any physical entity
in the simulation world, with properties like mass, position, velocity, etc.
"""

import uuid
from typing import Dict, List, Optional, Tuple, Union, Any as TypingAny # Renamed Any to TypingAny
import numpy as np
from pint import Quantity
from pydantic import BaseModel, Field, validator

from ...utils.logging import get_logger
from ...utils.units import ureg, u, convert_to_base_units, strip_units # Assuming ureg and u are defined here
from ...utils.exceptions import PhysicsError

logger = get_logger(__name__)

class ObjectState(BaseModel):
    """
    Immutable snapshot of an object's state at a specific time.
    """
    time: Quantity = Field(..., description="Simulation time when state was recorded")
    position: Quantity = Field(..., description="Position vector")
    velocity: Quantity = Field(..., description="Velocity vector")
    acceleration: Quantity = Field(..., description="Acceleration vector")
    orientation: np.ndarray = Field(..., description="Orientation quaternion (w, x, y, z)")
    angular_velocity: Quantity = Field(..., description="Angular velocity vector")
    angular_acceleration: Quantity = Field(..., description="Angular acceleration vector")
    kinetic_energy: Quantity = Field(..., description="Kinetic energy")
    potential_energy: Quantity = Field(..., description="Potential energy")
    applied_forces: List[Tuple[str, Quantity]] = Field(..., description="List of applied forces (name, force)")
    applied_torques: List[Tuple[str, Quantity]] = Field(..., description="List of applied torques (name, torque)")
    metadata: Dict[str, TypingAny] = Field(default_factory=dict, description="Additional state metadata")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Quantity: lambda q: {"value": q.magnitude, "units": str(q.units)},
            np.ndarray: lambda arr: arr.tolist() # Ensure numpy arrays are JSON serializable
        }

    def dict(self, *args, **kwargs) -> Dict[str, TypingAny]:
        """Override dict to properly handle quantities and numpy arrays"""
        d = super().dict(*args, **kwargs)
        for field_name in self.__fields__:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Quantity):
                d[field_name] = {"value": field_value.magnitude, "units": str(field_value.units)}
            elif isinstance(field_value, np.ndarray):
                d[field_name] = field_value.tolist()
        return d

class PhysicalObject:
    """
    Base class for all physical objects in the simulation.
    
    This class represents a physical entity with properties like mass, position,
    velocity, and methods for applying forces and updating state.
    """
    
    def __init__(self,
                 id: Optional[str] = None,
                 mass: Quantity = 1.0 * u.kilogram,
                 position: Quantity = np.zeros(3) * u.meter,
                 velocity: Quantity = np.zeros(3) * u.meter / u.second,
                 acceleration: Quantity = np.zeros(3) * u.meter / u.second**2,
                 orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternion (w, x, y, z)
                 angular_velocity: Quantity = np.zeros(3) * u.radian / u.second,
                 angular_acceleration: Quantity = np.zeros(3) * u.radian / u.second**2,
                 moment_of_inertia: Quantity = np.eye(3) * u.kilogram * u.meter**2,
                 fixed: bool = False,
                 properties: Optional[Dict[str, TypingAny]] = None):
        """
        Initialize a physical object.
        
        Args:
            id: Unique identifier (generated if None)
            mass: Mass of the object
            position: Initial position vector
            velocity: Initial velocity vector
            acceleration: Initial acceleration vector
            orientation: Initial orientation as quaternion (w, x, y, z)
            angular_velocity: Initial angular velocity vector
            angular_acceleration: Initial angular acceleration vector
            moment_of_inertia: Moment of inertia tensor (3x3 for 3D)
            fixed: Whether the object is fixed in space (infinite mass)
            properties: Additional physical properties (e.g., shape, material)
        """
        self.id = id or f"obj_{uuid.uuid4().hex[:8]}"
        self._mass = convert_to_base_units(mass)
        self._position = convert_to_base_units(position)
        self._velocity = convert_to_base_units(velocity)
        self._acceleration = convert_to_base_units(acceleration)
        
        if not isinstance(orientation, np.ndarray) or orientation.shape != (4,):
            raise PhysicsError("Orientation must be a 4-element numpy array (quaternion w,x,y,z)")
        self._orientation = orientation / np.linalg.norm(orientation) # Normalize
        
        self._angular_velocity = convert_to_base_units(angular_velocity)
        self._angular_acceleration = convert_to_base_units(angular_acceleration)
        self._moment_of_inertia = convert_to_base_units(moment_of_inertia)
        self.fixed = fixed
        self.properties = properties or {}
        
        self._history: List[ObjectState] = []
        self._applied_forces: Dict[str, Quantity] = {}
        self._applied_torques: Dict[str, Quantity] = {}
        
        self._validate_dimensions()
        
        logger.debug(f"Created PhysicalObject {self.id} with mass {self.mass.to_compact() if not self.fixed else 'infinite'}")
    
    def _validate_dimensions(self):
        """Validate that vector quantities have consistent dimensions."""
        dim = len(self._position)
        if dim not in (2, 3):
            raise PhysicsError(f"Position vector must be 2D or 3D, got {dim}D")

        for name, vec in [("velocity", self._velocity), 
                           ("acceleration", self._acceleration)]:
            if len(vec) != dim:
                raise PhysicsError(f"Dimension mismatch for {name}: expected {dim}D, got {len(vec)}D")

        if dim == 3: # Rotational properties are only meaningful in 3D
            if len(self._angular_velocity) != 3:
                raise PhysicsError(f"Angular velocity must be 3D, got {len(self._angular_velocity)}D")
            if len(self._angular_acceleration) != 3:
                raise PhysicsError(f"Angular acceleration must be 3D, got {len(self._angular_acceleration)}D")
            if self._moment_of_inertia.shape != (3,3):
                raise PhysicsError(f"Moment of inertia must be a 3x3 tensor, got {self._moment_of_inertia.shape}")
        elif len(self._angular_velocity) !=0 and np.any(self._angular_velocity.magnitude !=0): # Not zero vector
             logger.warning("Angular velocity is defined for a non-3D object. It will be ignored.")

    @property
    def mass(self) -> Quantity:
        """Get the object's mass."""
        return self._mass if not self.fixed else float('inf') * u.kilogram
    
    @mass.setter
    def mass(self, value: Quantity):
        """Set the object's mass."""
        if self.fixed:
            raise PhysicsError("Cannot set mass of a fixed object")
        self._mass = convert_to_base_units(value)
    
    @property
    def position(self) -> Quantity:
        """Get the object's position."""
        return self._position
    
    @position.setter
    def position(self, value: Quantity):
        """Set the object's position."""
        value = convert_to_base_units(value)
        if len(value) != len(self._position):
            raise PhysicsError(f"Position dimension mismatch: expected {len(self._position)}D, got {len(value)}D")
        self._position = value
    
    @property
    def velocity(self) -> Quantity:
        """Get the object's velocity."""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value: Quantity):
        """Set the object's velocity."""
        value = convert_to_base_units(value)
        if len(value) != len(self._velocity):
            raise PhysicsError(f"Velocity dimension mismatch: expected {len(self._velocity)}D, got {len(value)}D")
        self._velocity = value
    
    @property
    def acceleration(self) -> Quantity:
        """Get the object's acceleration."""
        return self._acceleration
    
    @acceleration.setter
    def acceleration(self, value: Quantity):
        """Set the object's acceleration."""
        value = convert_to_base_units(value)
        if len(value) != len(self._acceleration):
            raise PhysicsError(f"Acceleration dimension mismatch: expected {len(self._acceleration)}D, got {len(value)}D")
        self._acceleration = value
    
    @property
    def orientation(self) -> np.ndarray:
        """Get the object's orientation (quaternion w,x,y,z)."""
        return self._orientation
    
    @orientation.setter
    def orientation(self, value: np.ndarray):
        """Set the object's orientation (quaternion w,x,y,z)."""
        if not isinstance(value, np.ndarray) or value.shape != (4,):
            raise PhysicsError("Orientation must be a 4-element numpy array (quaternion w,x,y,z)")
        norm = np.linalg.norm(value)
        if norm == 0:
            raise PhysicsError("Orientation quaternion cannot be zero vector.")
        self._orientation = value / norm # Normalize
    
    @property
    def angular_velocity(self) -> Quantity:
        """Get the object's angular velocity (radians/s)."""
        return self._angular_velocity
    
    @angular_velocity.setter
    def angular_velocity(self, value: Quantity):
        """Set the object's angular velocity (radians/s)."""
        value = convert_to_base_units(value)
        if len(self._position) == 3 and len(value) != 3:
             raise PhysicsError(f"Angular velocity must be 3D for a 3D object, got {len(value)}D")
        self._angular_velocity = value
    
    @property
    def angular_acceleration(self) -> Quantity:
        """Get the object's angular acceleration (radians/s^2)."""
        return self._angular_acceleration
    
    @angular_acceleration.setter
    def angular_acceleration(self, value: Quantity):
        """Set the object's angular acceleration (radians/s^2)."""
        value = convert_to_base_units(value)
        if len(self._position) == 3 and len(value) != 3:
            raise PhysicsError(f"Angular acceleration must be 3D for a 3D object, got {len(value)}D")
        self._angular_acceleration = value
    
    @property
    def moment_of_inertia(self) -> Quantity:
        """Get the object's moment of inertia tensor (kg m^2)."""
        return self._moment_of_inertia
    
    @moment_of_inertia.setter
    def moment_of_inertia(self, value: Quantity):
        """Set the object's moment of inertia tensor (kg m^2)."""
        value = convert_to_base_units(value)
        if len(self._position) == 3 and value.shape != (3, 3):
            raise PhysicsError(f"Moment of inertia must be a 3x3 tensor for a 3D object, got {value.shape}")
        self._moment_of_inertia = value

    @property
    def kinetic_energy(self) -> Quantity:
        """Calculate the object's kinetic energy (Joules)."""
        linear_ke = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        angular_ke = 0.0 * u.joule
        if len(self._position) == 3 and self.mass != float('inf') * u.kilogram:
            # Ensure moment_of_inertia is a 3x3 matrix with correct units
            I = self.moment_of_inertia.to(u.kg * u.m**2).magnitude
            omega = self.angular_velocity.to(u.rad/u.s).magnitude
            if I.shape == (3,3) and omega.shape == (3,):
                 angular_ke_mag = 0.5 * np.dot(omega, np.dot(I, omega))
                 angular_ke = angular_ke_mag * u.joule
        return linear_ke + angular_ke

    @property
    def momentum(self) -> Quantity:
        """Calculate the object's linear momentum (kg m/s)."""
        return self.mass * self.velocity

    @property
    def angular_momentum(self) -> Quantity:
        """Calculate the object's angular momentum (kg m^2/s)."""
        if len(self._position) == 3 and self.mass != float('inf') * u.kilogram:
            I = self.moment_of_inertia.to(u.kg * u.m**2).magnitude
            omega = self.angular_velocity.to(u.rad/u.s).magnitude
            if I.shape == (3,3) and omega.shape == (3,):
                return np.dot(I, omega) * u.kg * u.m**2 / u.s
        return np.zeros(3) * u.kg * u.m**2 / u.s

    def apply_force(self, force: Quantity, name: Optional[str] = None, application_point: Optional[Quantity] = None) -> None:
        """
        Apply a force to the object.
        If application_point is provided (relative to object's CoM), it will also generate a torque.
        
        Args:
            force: Force vector to apply (Newtons)
            name: Optional name for the force (for tracking)
            application_point: Optional point of force application relative to CoM (meters)
        """
        if self.fixed:
            return
            
        force = convert_to_base_units(force)
        if len(force) != len(self._position):
            raise PhysicsError(f"Force dimension mismatch: expected {len(self._position)}D, got {len(force)}D")
            
        name = name or f"force_{len(self._applied_forces) + 1}"
        self._applied_forces[name] = force
        self._acceleration += force / self.mass

        if application_point is not None and len(self._position) == 3:
            application_point = convert_to_base_units(application_point)
            if len(application_point) != 3:
                raise PhysicsError("Application point must be a 3D vector for torque calculation")
            # Calculate torque: r x F
            torque = np.cross(application_point.to(u.m).magnitude, force.to(u.N).magnitude) * u.N * u.m
            self.apply_torque(torque, name=f"torque_from_{name}")

    def apply_torque(self, torque: Quantity, name: Optional[str] = None) -> None:
        """
        Apply a torque to the object (Newton meters).
        Only applicable to 3D objects.
        
        Args:
            torque: Torque vector to apply
            name: Optional name for the torque (for tracking)
        """
        if self.fixed or len(self._position) != 3:
            return # Torques are only for 3D non-fixed objects
            
        torque = convert_to_base_units(torque)
        if len(torque) != 3:
            raise PhysicsError("Torque must be a 3D vector")
            
        name = name or f"torque_{len(self._applied_torques) + 1}"
        self._applied_torques[name] = torque
        
        # Angular acceleration = I_inv * torque
        # Need to handle potential singularity if moment_of_inertia is not invertible
        try:
            I_inv = np.linalg.inv(self.moment_of_inertia.to(u.kg * u.m**2).magnitude)
            alpha_mag = np.dot(I_inv, torque.to(u.N*u.m).magnitude)
            self._angular_acceleration += alpha_mag * u.rad / u.s**2
        except np.linalg.LinAlgError:
            logger.warning(f"Moment of inertia for object {self.id} is singular. Torque will not be applied.")

    def clear_forces_and_torques(self) -> None:
        """Clear all applied forces and torques, and reset accelerations."""
        self._applied_forces.clear()
        self._applied_torques.clear()
        self._acceleration = np.zeros_like(self._acceleration) * u.m / u.s**2
        self._angular_acceleration = np.zeros_like(self._angular_acceleration) * u.rad / u.s**2

    def get_state_snapshot(self, time: Quantity) -> ObjectState:
        """
        Get a snapshot of the object's current state.
        
        Args:
            time: Current simulation time
            
        Returns:
            ObjectState containing the current state
        """
        return ObjectState(
            time=time,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            angular_acceleration=self.angular_acceleration,
            kinetic_energy=self.kinetic_energy,
            potential_energy=0 * u.joule,  # Potential energy is typically world-dependent
            applied_forces=list(self._applied_forces.items()),
            applied_torques=list(self._applied_torques.items()),
            metadata={
                "id": self.id,
                "fixed": self.fixed,
                **self.properties
            }
        )

    def record_state(self, time: Quantity) -> None:
        """Record the current state to history."""
        self._history.append(self.get_state_snapshot(time))

    def clear_history(self) -> None:
        """Clear the object's state history."""
        self._history.clear()

    @property
    def history(self) -> List[ObjectState]: # Return list of ObjectState Pydantic models
        """Get the object's state history as a list of ObjectState instances."""
        return self._history

    def update(self, dt: Quantity) -> None:
        """
        Update the object's state based on current forces and torques using Verlet integration for position.
        
        Args:
            dt: Time step (Quantity)
        """
        if self.fixed:
            self.clear_forces_and_torques() # Clear forces even if fixed, as they might have been applied
            return
            
        dt_sec = dt.to(ureg.second).magnitude
        
        # Update linear motion
        # Position update using Verlet integration (more stable for physics simulations)
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        self._position += self._velocity * dt + 0.5 * self._acceleration * dt**2
        # Velocity update: v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt
        # Since a(t+dt) is not known yet, use a(t) for now, then recalculate forces and a(t+dt) in world step.
        # Or, more simply: v(t+dt) = v(t) + a(t)*dt (Euler for velocity)
        self._velocity += self._acceleration * dt
        
        # Update angular motion if 3D
        if len(self._position) == 3:
            # Angular velocity update (Euler)
            self._angular_velocity += self._angular_acceleration * dt
            
            # Orientation update using quaternion integration
            # dQ/dt = 0.5 * Q * omega_q
            # where omega_q is the quaternion (0, omega_x, omega_y, omega_z)
            omega_vec = self._angular_velocity.to(u.rad/u.s).magnitude
            if np.any(omega_vec): # Only update if there's angular velocity
                # Quaternion derivative
                q_w, q_x, q_y, q_z = self._orientation
                omega_x, omega_y, omega_z = omega_vec
                
                # dQ/dt components
                dq_w = 0.5 * (-q_x * omega_x - q_y * omega_y - q_z * omega_z)
                dq_x = 0.5 * ( q_w * omega_x + q_y * omega_z - q_z * omega_y)
                dq_y = 0.5 * ( q_w * omega_y + q_z * omega_x - q_x * omega_z)
                dq_z = 0.5 * ( q_w * omega_z + q_x * omega_y - q_y * omega_x)
                
                # Update orientation
                self._orientation += np.array([dq_w, dq_x, dq_y, dq_z]) * dt_sec
                self._orientation /= np.linalg.norm(self._orientation) # Re-normalize
        
        # Forces and torques are applied for one time step, then cleared by the world or before next update.
        # The world will typically call clear_forces_and_torques() after calculating all interactions
        # and before the next `update` call.

    def __repr__(self) -> str:
        return f"PhysicalObject(id='{self.id}', mass={self.mass}, position={self.position})" 