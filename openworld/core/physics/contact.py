import numpy as np
from typing import List
from ..physics.world import PhysicsObject

class Contact:
    def __init__(self, obj1: PhysicsObject, obj2: PhysicsObject):
        self.obj1 = obj1
        self.obj2 = obj2
        self.contact_normal = self._calculate_contact_normal()
        self.penetration = self._calculate_penetration()
        self.contact_point = self._calculate_contact_point()
        self.restitution = min(obj1.restitution, obj2.restitution)
        self.friction = np.sqrt(obj1.friction * obj2.friction)
        
    def _calculate_contact_normal(self) -> np.ndarray:
        """Calculate contact normal based on object positions and shapes"""
        if hasattr(self.obj1, 'geometry_type') and hasattr(self.obj2, 'geometry_type'):
            # For sphere-sphere collision
            if self.obj1.geometry_type == 'sphere' and self.obj2.geometry_type == 'sphere':
                normal = self.obj2.position - self.obj1.position
                return normal / np.linalg.norm(normal)
            # Add other geometry-specific normal calculations
        return np.array([0, 1, 0])  # Default normal
        
    def _calculate_penetration(self) -> float:
        """Calculate penetration depth based on object geometries"""
        if self.obj1.geometry_type == 'sphere' and self.obj2.geometry_type == 'sphere':
            r1 = self.obj1.geometry_params['radius']
            r2 = self.obj2.geometry_params['radius']
            distance = np.linalg.norm(self.obj2.position - self.obj1.position)
            return r1 + r2 - distance
        # Add other geometry-specific penetration calculations
        return 0.0
        
    def _calculate_contact_point(self) -> np.ndarray:
        """Calculate contact point between objects"""
        if self.obj1.geometry_type == 'sphere' and self.obj2.geometry_type == 'sphere':
            r1 = self.obj1.geometry_params['radius']
            r2 = self.obj2.geometry_params['radius']
            total_radius = r1 + r2
            if total_radius > 0:
                return self.obj1.position + (self.obj2.position - self.obj1.position) * (r1 / total_radius)
        return (self.obj1.position + self.obj2.position) / 2

class ContactResolver:
    def __init__(self, velocity_iterations: int = 10, position_iterations: int = 5):
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        
    def resolve(self, contacts: List[Contact], time_step: float):
        if not contacts:
            return
            
        # Resolve interpenetration
        for _ in range(self.position_iterations):
            max_penetration = 0
            worst_contact = None
            
            # Find worst penetration
            for contact in contacts:
                if contact.penetration > max_penetration:
                    max_penetration = contact.penetration
                    worst_contact = contact
                    
            if worst_contact:
                self._resolve_penetration(worst_contact)
                
        # Resolve velocity
        for _ in range(self.velocity_iterations):
            for contact in contacts:
                self._resolve_velocity(contact, time_step)

    def _resolve_penetration(self, contact: Contact):
        """Resolve positional penetration between objects"""
        move_per_mass = contact.penetration / (contact.obj1.mass + contact.obj2.mass)
        
        # Calculate movement vectors
        move1 = contact.contact_normal * (move_per_mass * contact.obj2.mass)
        move2 = contact.contact_normal * (-move_per_mass * contact.obj1.mass)
        
        # Apply position corrections
        contact.obj1.position += move1
        contact.obj2.position += move2

    def _resolve_velocity(self, contact: Contact, time_step: float):
        """Resolve collision velocities (impulse method)"""
        # Calculate relative velocity
        rv = (contact.obj2.velocity - contact.obj1.velocity)
        vel_along_normal = np.dot(rv, contact.contact_normal)
        
        # Do not resolve if objects are separating
        if vel_along_normal > 0:
            return
            
        # Calculate restitution
        e = min(contact.obj1.restitution, contact.obj2.restitution)
        
        # Calculate impulse scalar
        j = -(1 + e) * vel_along_normal
        j /= (1/contact.obj1.mass + 1/contact.obj2.mass)
        
        # Apply impulse
        impulse = j * contact.contact_normal
        contact.obj1.velocity -= impulse / contact.obj1.mass
        contact.obj2.velocity += impulse / contact.obj2.mass
        
        # Friction impulse
        tangent = rv - contact.contact_normal * vel_along_normal
        if np.linalg.norm(tangent) > 1e-6:
            tangent = tangent / np.linalg.norm(tangent)
            
            # Calculate friction impulse
            jt = -np.dot(rv, tangent)
            jt /= (1/contact.obj1.mass + 1/contact.obj2.mass)
            
            # Coulomb's law
            max_friction = contact.friction * abs(j)
            jt = np.clip(jt, -max_friction, max_friction)
            
            # Apply friction impulse
            friction_impulse = jt * tangent
            contact.obj1.velocity -= friction_impulse / contact.obj1.mass
            contact.obj2.velocity += friction_impulse / contact.obj2.mass 