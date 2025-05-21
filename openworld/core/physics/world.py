import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ...utils.logging import get_logger
import uuid
from scipy.spatial.transform import Rotation
from .collision import GJK
from .contact import ContactResolver
from .broadphase import SweepAndPrune, BVHTree, BruteForce

logger = get_logger(__name__)

@dataclass
class PhysicsObject:
    object_id: str
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    geometry_type: str = 'sphere'
    geometry_params: dict = field(default_factory=lambda: {'radius': 1.0})
    restitution: float = 0.5  # Coefficient of restitution
    friction: float = 0.5     # Coefficient of friction
    linear_damping: float = 0.05
    angular_damping: float = 0.05
    kinematic: bool = False   # Whether object is kinematic (not affected by physics)
    sleep_threshold: float = 0.1
    is_sleeping: bool = False
    user_data: Any = None  # For custom user data
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.inertia = np.asarray(self.inertia, dtype=np.float64)
        
        # Calculate inertia tensor if not provided
        if np.array_equal(self.inertia, np.eye(3)):
            self._calculate_inertia()
        
        # Initialize geometry params if not provided
        if self.geometry_type == 'sphere' and 'radius' not in self.geometry_params:
            self.geometry_params['radius'] = 1.0
        elif self.geometry_type == 'box' and 'size' not in self.geometry_params:
            self.geometry_params['size'] = [1.0, 1.0, 1.0]

    def _calculate_inertia(self):
        """Calculate inertia tensor based on geometry"""
        if self.geometry_type == 'sphere':
            r = self.geometry_params.get('radius', 1.0)
            i = 0.4 * self.mass * r * r
            self.inertia = np.diag([i, i, i])
        elif self.geometry_type == 'box':
            size = np.array(self.geometry_params.get('size', [1.0, 1.0, 1.0]))
            i = self.mass * (size ** 2) / 12.0
            self.inertia = np.diag(i)

class PhysicsWorld:
    def __init__(self, dimensions: int = 3, gravity: List[float] = [0, -9.81, 0],
                 time_step: float = 0.01, solver_iterations: int = 10,
                 broadphase_type: str = "SAP"):
        self.id = str(uuid.uuid4())
        self.dimensions = dimensions
        self.gravity = np.array(gravity[:dimensions])
        self.time_step = time_step
        self.solver_iterations = solver_iterations
        self.objects: Dict[str, PhysicsObject] = {}
        self.time_elapsed = 0.0
        self.contact_resolver = ContactResolver(
            velocity_iterations=solver_iterations,
            position_iterations=solver_iterations
        )
        self.broadphase = self._create_broadphase(broadphase_type)
        
    def _create_broadphase(self, broadphase_type: str):
        """Create broadphase collision detection system"""
        if broadphase_type == "SAP":
            return SweepAndPrune()
        elif broadphase_type == "BVH":
            return BVHTree()
        else:
            return BruteForce()
            
    def add_object(self, config: Dict[str, Any]) -> str:
        obj = PhysicsObject(
            object_id=config['object_id'],
            position=np.array(config['position']),
            orientation=np.array(config['orientation']),
            mass=config['mass'],
            geometry_type=config.get('geometry_type', 'sphere'),
            geometry_params=config.get('geometry_params', {'radius': 1.0})
        )
        self.objects[obj.object_id] = obj
        return obj.object_id
        
    def run_simulation(self, duration: float) -> Dict[str, Any]:
        steps = int(duration / self.time_step)
        for _ in range(steps):
            self._step()
        return self.get_simulation_state()
        
    def _step(self):
        # Apply forces and integrate
        for obj in self.objects.values():
            if not obj.kinematic:
                obj.velocity += self.gravity * self.time_step
                obj.velocity *= (1 - obj.linear_damping)
                obj.angular_velocity *= (1 - obj.angular_damping)
                
            obj.position += obj.velocity * self.time_step
            if np.any(obj.angular_velocity != 0):
                rotation = Rotation.from_rotvec(obj.angular_velocity * self.time_step)
                obj.orientation = (rotation * Rotation.from_quat(obj.orientation)).as_quat()
        
        # Collision detection and response
        contacts = self._detect_collisions()
        self.contact_resolver.resolve(contacts, self.time_step)
        self.time_elapsed += self.time_step

    def _detect_collisions(self) -> List[Contact]:
        """Optimized collision detection with broadphase"""
        # Broadphase: Find potential collision pairs
        potential_pairs = self.broadphase.update(self.objects.values())
        
        # Narrowphase: Exact collision detection
        contacts = []
        for obj1, obj2 in potential_pairs:
            if GJK.check_collision(obj1, obj2):
                contacts.append(Contact(obj1, obj2))
                
        return contacts

    def get_simulation_state(self) -> Dict[str, Any]:
        return {
            'time_elapsed': self.time_elapsed,
            'objects': {oid: {
                'position': obj.position.tolist(),
                'orientation': obj.orientation.tolist(),
                'velocity': obj.velocity.tolist(),
                'angular_velocity': obj.angular_velocity.tolist()
            } for oid, obj in self.objects.items()}
        }

    def update_object_bounds(self, obj: PhysicsObject):
        """Update broadphase bounds when object moves"""
        self.broadphase.update_object(obj) 