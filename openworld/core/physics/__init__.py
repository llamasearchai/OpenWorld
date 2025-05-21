"""
Physics simulation module for the OpenWorld platform.

This module provides physics simulation capabilities for classical mechanics.
"""

from .particle import Particle
from .world import PhysicsWorld, PhysicsObject
from .collision import GJK
from .contact import Contact, ContactResolver
from .broadphase import Broadphase, BruteForce, SweepAndPrune, BVHTree

__all__ = [
    'Particle',
    'PhysicsWorld',
    'PhysicsObject',
    'GJK',
    'Contact',
    'ContactResolver',
    'Broadphase',
    'BruteForce',
    'SweepAndPrune',
    'BVHTree'
] 