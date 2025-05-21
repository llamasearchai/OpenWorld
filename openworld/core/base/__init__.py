"""
Base physics classes for the OpenWorld platform.

This module provides base classes used throughout the physics simulation modules.
"""

from .physical_object import PhysicalObject, ObjectState
from .physics_world import PhysicsWorld

__all__ = [
    'PhysicalObject',
    'ObjectState',
    'PhysicsWorld'
] 