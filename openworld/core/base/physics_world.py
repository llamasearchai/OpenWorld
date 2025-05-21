"""
Physics world class for managing physical objects and simulation.
"""

import uuid
import numpy as np
from typing import Dict, List, Callable, Any as TypingAny, Optional, Tuple, Union # Renamed Any to TypingAny
from pint import Quantity
import time
from collections import defaultdict

from ...utils.logging import get_logger
from ...utils.units import ureg, u, convert_to_base_units, strip_units # Assuming ureg and u are defined here
from ...utils.exceptions import ConfigurationError, PhysicsError
from .physical_object import PhysicalObject, ObjectState

logger = get_logger(__name__)

class PhysicsWorld:
    """
    Container for physical objects, forces, constraints, and simulation state.
    
    This class manages a collection of physical objects and provides
    methods for adding global forces, applying constraints, and retrieving
    simulation state. It also handles the main simulation loop.
    """
    
    def __init__(self,
                 dimensions: int = 3,
                 gravity: Optional[Union[Quantity, List[float], float]] = None,
                 world_id: Optional[str] = None,
                 enable_history: bool = True,
                 history_interval: Quantity = 0.1 * u.second,
                 solver_type: str = "euler"):
        """
        Initialize a physics world.
        
        Args:
            dimensions: Number of spatial dimensions (2 or 3).
            gravity: Gravity vector (Quantity or list) or scalar magnitude (float).
                     If scalar, applied in -y for 2D or -z for 3D.
                     Default units are m/s^2 if float provided.
            world_id: Custom world ID (generated if not provided).
            enable_history: Whether to record object state history.
            history_interval: Time interval between history recordings (Quantity).
            solver_type: Type of solver to use ('euler', 'verlet', etc. - currently only Euler is used for velocity).
        """
        if dimensions not in (2, 3):
            raise ConfigurationError("Physics world must be 2D or 3D")
            
        self.dimensions = dimensions
        self.world_id = world_id or f"world_{uuid.uuid4().hex[:8]}"
        self._objects: Dict[str, PhysicalObject] = {}
        self._global_forces: List[Callable[[PhysicalObject, float], Quantity]] = [] # Force func (obj, time_sec) -> force_vec
        self._constraints: List[Callable[[PhysicsWorld, float], None]] = [] # Constraint func (world, dt_sec) -> None
        self._time: Quantity = 0.0 * ureg.second
        self._default_gravity_accel: Optional[Quantity] = None
        
        self._enable_history = enable_history
        self._history_interval = convert_to_base_units(history_interval, u.second)
        self._next_history_time: Quantity = self._history_interval if self._enable_history else float('inf') * u.second
        
        self._collision_handlers: List[Callable[[PhysicalObject, PhysicalObject, TypingAny], None]] = [] # handler(obj1, obj2, collision_info)
        self._performance_stats = defaultdict(list)
        self.solver_type = solver_type # Currently illustrative, object update method handles integration.
        
        if gravity is not None:
            self.set_gravity(gravity)
        
        logger.info(f"Initialized PhysicsWorld (ID: {self.world_id}) with {dimensions} dimensions. Solver: {solver_type}")

    def set_gravity(self, gravity: Union[Quantity, List[float], float]) -> None:
        """
        Set the gravity for this world.
        
        Args:
            gravity: Gravity vector (Quantity or list) or scalar magnitude (float).
                     If scalar, applied in -y for 2D or -z for 3D.
                     Default units are m/s^2 if float provided.
        """
        # Remove existing gravity force if present
        # This requires identifying the specific _gravity_force method, which can be tricky if other identical methods exist.
        # A more robust way is to store a reference to the added gravity force function or flag it.
        # For now, assume it's the only one named _gravity_force or the last one added if multiple were.
        self._global_forces = [f for f in self._global_forces if getattr(f, '__name__', '') != '_gravity_force']
        
        if isinstance(gravity, (int, float)):
            # If scalar, create vector and assume m/s^2
            if self.dimensions == 2:
                gravity_vec = [0, -abs(gravity)] * u.meter / u.second**2
            else:  # 3D
                gravity_vec = [0, 0, -abs(gravity)] * u.meter / u.second**2
        elif isinstance(gravity, list):
            gravity_vec = ureg.Quantity(gravity, u.meter / u.second**2) # Assume m/s^2 if list of floats
        elif isinstance(gravity, Quantity):
            gravity_vec = gravity
        else:
            raise ConfigurationError("Invalid gravity type. Must be Quantity, list of floats, or float.")
            
        # Convert to base units for internal consistency
        self._default_gravity_accel = convert_to_base_units(gravity_vec, u.meter / u.second**2)
        
        if self._default_gravity_accel.shape != (self.dimensions,):
            raise ConfigurationError(
                f"Gravity vector dimension mismatch: {self._default_gravity_accel.shape} vs world {self.dimensions}D"
            )
        
        self.add_global_force(self._gravity_force)
        logger.info(f"Set gravity to {self._default_gravity_accel.to_compact()}")

    def _gravity_force(self, obj: PhysicalObject, time_sec: float) -> Quantity:
        """Internal function for applying default gravity."""
        if self._default_gravity_accel is not None and not obj.fixed:
            return obj.mass * self._default_gravity_accel
        return np.zeros(self.dimensions) * ureg.newton # No force if no gravity or object is fixed

    @property
    def time(self) -> Quantity:
        """Current simulation time."""
        return self._time

    @property
    def objects(self) -> Dict[str, PhysicalObject]:
        """Dictionary of physical objects in the world."""
        return self._objects

    def add_object(self, obj: PhysicalObject) -> None:
        """
        Add a physical object to the world.
        
        Args:
            obj: Physical object to add
        Raises:
            ValueError: If object dimensions don't match world dimensions or object ID exists.
        """
        if len(obj.position) != self.dimensions:
            raise ValueError(f"Object position dimension {len(obj.position)}D != World dimension {self.dimensions}D")
        
        if obj.id in self._objects:
            # To allow overwriting, one might remove the old object first or use a different method.
            # For strictness, we prevent overwriting by default.
            raise ValueError(f"Object with ID '{obj.id}' already exists. Overwriting is not allowed by add_object.")
        
        self._objects[obj.id] = obj
        logger.debug(f"Added object '{obj.id}' to world '{self.world_id}'")

    def remove_object(self, obj_id: str) -> None:
        """
        Remove an object by ID.
        Args:
            obj_id: ID of object to remove
        """
        if obj_id in self._objects:
            del self._objects[obj_id]
            logger.debug(f"Removed object '{obj_id}' from world '{self.world_id}'")
        else:
            logger.warning(f"Attempted to remove non-existent object ID: '{obj_id}'")

    def get_object(self, obj_id: str) -> Optional[PhysicalObject]:
        """
        Get an object by ID.
        Args:
            obj_id: ID of object to retrieve
        Returns:
            The physical object or None if not found.
        """
        return self._objects.get(obj_id)

    def add_global_force(self, force_func: Callable[[PhysicalObject, float], Quantity]) -> None:
        """
        Add a force function that potentially acts on all objects.
        Args:
            force_func: Force function taking (PhysicalObject, time_in_seconds) and returning force Quantity.
        """
        self._global_forces.append(force_func)
        logger.debug(f"Added global force function: {getattr(force_func, '__name__', 'anonymous_force')}")

    def add_constraint(self, constraint_func: Callable[[PhysicsWorld, float], None]) -> None:
        """
        Add a constraint function to the simulation.
        Args:
            constraint_func: Function that takes (PhysicsWorld, dt_in_seconds) and applies constraints.
        """
        self._constraints.append(constraint_func)
        logger.debug(f"Added constraint function: {getattr(constraint_func, '__name__', 'anonymous_constraint')}")

    def add_collision_handler(self, handler: Callable[[PhysicalObject, PhysicalObject, TypingAny], None]) -> None:
        """
        Add a collision detection/handling function.
        Args:
            handler: Function that takes (obj1, obj2, collision_info) and handles collision.
                     collision_info can be specific to the detection method (e.g., penetration depth, normal).
        """
        self._collision_handlers.append(handler)
        logger.debug(f"Added collision handler: {getattr(handler, '__name__', 'anonymous_collision_handler')}")

    def get_world_state(self) -> Dict[str, TypingAny]:
        """
        Return a snapshot of the world's current state.
        Returns:
            Dictionary with world state information.
        """
        return {
            "world_id": self.world_id,
            "time": self._time.to_dict() if hasattr(self._time, 'to_dict') else str(self._time), # For Pydantic Quantity serialization
            "dimensions": self.dimensions,
            "num_objects": len(self.objects),
            "objects": {
                obj_id: obj.get_state_snapshot(self._time).dict()
                for obj_id, obj in self.objects.items()
            },
            "performance_stats": {k: np.mean(v) if v else 0 for k, v in self._performance_stats.items()}
        }

    def get_object_histories(self) -> Dict[str, List[Dict[str, TypingAny]]]:
        """
        Return histories for all objects.
        Returns:
            Dictionary mapping object IDs to their history (list of ObjectState dicts).
        """
        histories = {}
        for obj_id, obj in self.objects.items():
            histories[obj_id] = [state.dict() for state in obj.history]
        return histories

    def clear_histories(self) -> None:
        """Clear history for all objects and reset history recording time."""
        for obj in self.objects.values():
            obj.clear_history()
        self._next_history_time = self._time + self._history_interval if self._enable_history else float('inf') * u.second
        logger.debug(f"Cleared histories for world {self.world_id}")

    def _apply_forces_and_torques(self) -> None:
        """Apply all global forces and torques to objects."""
        time_sec = self._time.to(ureg.second).magnitude
        for obj in self.objects.values():
            if obj.fixed: continue # Skip fixed objects
            obj.clear_forces_and_torques() # Clear from previous step before applying new ones
            for force_func in self._global_forces:
                try:
                    force = force_func(obj, time_sec)
                    if force is not None and isinstance(force, Quantity) and np.any(force.magnitude):
                        obj.apply_force(force, name=getattr(force_func, '__name__', 'global_force'))
                except Exception as e:
                    logger.error(f"Error in global force function '{getattr(force_func, '__name__', '')}' for obj '{obj.id}': {e}")
            # Note: Specific interaction forces (e.g., collisions, springs) should be handled separately
            # or as part of global_forces if they fit that model.

    def _apply_constraints(self, dt_sec: float) -> None:
        """Apply all constraints to the world."""
        for constraint_func in self._constraints:
            try:
                constraint_func(self, dt_sec)
            except Exception as e:
                logger.error(f"Error in constraint function '{getattr(constraint_func, '__name__', '')}': {e}")

    def _detect_and_handle_collisions(self, dt_sec: float) -> None:
        """
        Detect and handle collisions between objects.
        This is a placeholder for a proper collision detection and response system.
        A full implementation would involve broadphase and narrowphase detection,
        and then resolving collisions using impulses or penalty forces.
        """
        # Example: Naive O(N^2) collision check (very inefficient for many objects)
        # This should be replaced with a more sophisticated collision detection system.
        object_list = list(self.objects.values())
        for i in range(len(object_list)):
            for j in range(i + 1, len(object_list)):
                obj1 = object_list[i]
                obj2 = object_list[j]
                
                if obj1.fixed and obj2.fixed: continue

                # Placeholder for collision detection logic
                # e.g., if isinstance(obj1.properties.get('shape'), Sphere) and ...
                # collision_info = check_collision(obj1, obj2)
                collision_info = None # Placeholder
                
                if collision_info:
                    for handler in self._collision_handlers:
                        try:
                            handler(obj1, obj2, collision_info)
                        except Exception as e:
                            logger.error(f"Error in collision handler '{getattr(handler, '__name__', '')}': {e}")

    def _record_object_states(self) -> None:
        """Record state of all objects if history is enabled and interval is met."""
        if self._enable_history and self._time >= self._next_history_time:
            for obj in self.objects.values():
                obj.record_state(self._time)
            self._next_history_time += self._history_interval

    def step(self, dt: Quantity) -> None:
        """
        Advance the simulation by one time step.
        
        Args:
            dt: Time step (Quantity).
        """
        step_start_time = time.perf_counter()
        dt_base = convert_to_base_units(dt, u.second)
        dt_sec = dt_base.magnitude

        # 1. Apply forces and torques based on current state
        force_start_time = time.perf_counter()
        self._apply_forces_and_torques()
        self._performance_stats['force_calc_time_ms'].append((time.perf_counter() - force_start_time) * 1000)
        
        # 2. Update object states (position, velocity, orientation, etc.)
        update_start_time = time.perf_counter()
        for obj in self.objects.values():
            if not obj.fixed:
                obj.update(dt_base)
        self._performance_stats['object_update_time_ms'].append((time.perf_counter() - update_start_time) * 1000)

        # 3. Detect and handle collisions / Apply constraints
        # Collision detection can be complex. A simple placeholder is used.
        # Constraints might adjust positions/velocities after the update step.
        collision_constraint_start_time = time.perf_counter()
        self._detect_and_handle_collisions(dt_sec) # This might modify accelerations/velocities for next step or apply impulses
        self._apply_constraints(dt_sec) # This might directly modify positions/velocities
        self._performance_stats['collision_constraint_time_ms'].append((time.perf_counter() - collision_constraint_start_time) * 1000)
        
        # 4. Record state history if enabled and interval is met
        record_start_time = time.perf_counter()
        self._record_object_states()
        self._performance_stats['history_record_time_ms'].append((time.perf_counter() - record_start_time) * 1000)
        
        # 5. Advance simulation time
        self._time += dt_base
        
        self._performance_stats['total_step_time_ms'].append((time.perf_counter() - step_start_time) * 1000)
        
        # Keep only last N performance stats for averaging
        for key in self._performance_stats:
            if len(self._performance_stats[key]) > 100:
                self._performance_stats[key] = self._performance_stats[key][-100:]

    def run_simulation(self, 
                      duration: Quantity, 
                      dt: Quantity = 0.01 * u.second,
                      progress_callback: Optional[Callable[[float, Quantity], None]] = None) -> Dict[str, TypingAny]:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Total simulation time (Quantity).
            dt: Time step size (Quantity).
            progress_callback: Optional callback function for progress updates (receives progress fraction, current_time).
            
        Returns:
            Dictionary containing simulation results.
        """
        sim_start_time = time.perf_counter()
        
        duration_base = convert_to_base_units(duration, u.second)
        dt_base = convert_to_base_units(dt, u.second)
        
        if dt_base.magnitude <= 0:
            raise ValueError("Time step dt must be positive.")
        if duration_base.magnitude < 0: # Allow zero duration for initial state snapshot
             raise ValueError("Simulation duration must be non-negative.")

        num_steps = int(np.ceil(duration_base / dt_base)) if duration_base.magnitude > 0 else 0
        
        logger.info(f"Starting simulation for {duration_base.to_compact()} with dt={dt_base.to_compact()} ({num_steps} steps)")
        
        if self._enable_history:
            self.clear_histories() # Clear previous history before new run
            self._record_object_states() # Record initial state at t=0
        
        for step_num in range(num_steps):
            self.step(dt_base)
            
            if progress_callback is not None and step_num % max(1, num_steps // 100) == 0:
                current_progress = (step_num + 1) / num_steps
                try:
                    progress_callback(current_progress, self._time)
                except Exception as e:
                    logger.warning(f"Error in progress_callback: {e}")
        
        # Ensure final state is recorded if history is enabled
        if self._enable_history and num_steps > 0 and self._time < self._next_history_time : # If last step didn't trigger recording
             self._record_object_states() 

        sim_end_time = time.perf_counter()
        elapsed_real_time_sec = sim_end_time - sim_start_time
        
        results = self.get_world_state() # Get final state of the world
        if self._enable_history:
            results["histories"] = self.get_object_histories()
        
        results["simulation_config"] = {
            "duration_requested": str(duration_base.to_compact()),
            "dt_requested": str(dt_base.to_compact()),
            "num_steps_calculated": num_steps
        }
        results["performance_summary"] = {
            "total_real_time_seconds": elapsed_real_time_sec,
            "steps_per_second_real_time": num_steps / elapsed_real_time_sec if elapsed_real_time_sec > 0 else float('inf'),
            "avg_step_time_ms": np.mean(self._performance_stats['total_step_time_ms']) if self._performance_stats['total_step_time_ms'] else 0
        }
        
        logger.info(f"Simulation completed in {elapsed_real_time_sec:.3f} real seconds. Final sim time: {self._time.to_compact()}")
        return results

    def __repr__(self) -> str:
        return f"PhysicsWorld(id='{self.world_id}', time={self.time.to_compact()}, objects={len(self._objects)})" 