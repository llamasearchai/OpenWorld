import numpy as np
from typing import List, Tuple
from ..physics.world import PhysicsObject

class GJK:
    @staticmethod
    def check_collision(obj1: PhysicsObject, obj2: PhysicsObject) -> bool:
        """Implementation of Gilbert-Johnson-Keerthi collision detection algorithm"""
        simplex = []
        direction = np.array([1.0, 0.0, 0.0])
        
        # Initial support point
        support = GJK._support(obj1, obj2, direction)
        simplex.append(support)
        direction = -support
        
        while True:
            support = GJK._support(obj1, obj2, direction)
            
            if np.dot(support, direction) < 0:
                return False
                
            simplex.append(support)
            
            if GJK._handle_simplex(simplex, direction):
                return True

    @staticmethod
    def _support(obj1: PhysicsObject, obj2: PhysicsObject, d: np.ndarray) -> np.ndarray:
        """Calculate support point in direction d for Minkowski difference"""
        p1 = GJK._get_furthest_point(obj1, d)
        p2 = GJK._get_furthest_point(obj2, -d)
        return p1 - p2

    @staticmethod
    def _get_furthest_point(obj: PhysicsObject, d: np.ndarray) -> np.ndarray:
        """Get furthest point in direction d for different geometry types"""
        if obj.geometry_type == 'sphere':
            return obj.position + d * obj.geometry_params['radius']
        elif obj.geometry_type == 'box':
            half_extents = np.array(obj.geometry_params['size'])
            return obj.position + np.sign(d) * half_extents
        elif obj.geometry_type == 'capsule':
            axis = np.array([0, 1, 0])  # Default capsule axis
            radius = obj.geometry_params['radius']
            height = obj.geometry_params['height']
            # ... capsule support function implementation ...
        else:
            raise ValueError(f"Unsupported geometry type: {obj.geometry_type}")

    @staticmethod
    def _handle_simplex(simplex: List[np.ndarray], direction: np.ndarray) -> bool:
        """Evolve the simplex towards the origin"""
        if len(simplex) == 2:
            return GJK._line_case(simplex, direction)
        elif len(simplex) == 3:
            return GJK._triangle_case(simplex, direction)
        elif len(simplex) == 4:
            return GJK._tetrahedron_case(simplex, direction)
        return False

    @staticmethod
    def _line_case(simplex: List[np.ndarray], direction: np.ndarray) -> bool:
        a, b = simplex
        ab = b - a
        ao = -a
        
        if np.dot(ab, ao) > 0:
            direction = np.cross(np.cross(ab, ao), ab)
        else:
            simplex.pop(0)
            direction = ao
            
        return False

    @staticmethod
    def _triangle_case(simplex: List[np.ndarray], direction: np.ndarray) -> bool:
        a, b, c = simplex
        ab = b - a
        ac = c - a
        ao = -a
        abc = np.cross(ab, ac)
        
        if np.dot(np.cross(abc, ac), ao) > 0:
            if np.dot(ac, ao) > 0:
                simplex = [a, c]
                direction = np.cross(np.cross(ac, ao), ac)
            else:
                return GJK._line_case([a, b], direction)
        else:
            if np.dot(np.cross(ab, abc), ao) > 0:
                return GJK._line_case([a, b], direction)
            else:
                if np.dot(abc, ao) > 0:
                    simplex = [a, b, c]
                    direction = abc
                else:
                    simplex = [a, c, b]
                    direction = -abc
                    
        return False

    @staticmethod
    def _tetrahedron_case(simplex: List[np.ndarray], direction: np.ndarray) -> bool:
        a, b, c, d = simplex
        ab = b - a
        ac = c - a
        ad = d - a
        ao = -a
        
        abc = np.cross(ab, ac)
        acd = np.cross(ac, ad)
        adb = np.cross(ad, ab)
        
        if np.dot(abc, ao) > 0:
            simplex = [a, b, c]
            return GJK._triangle_case(simplex, direction)
        elif np.dot(acd, ao) > 0:
            simplex = [a, c, d]
            return GJK._triangle_case(simplex, direction)
        elif np.dot(adb, ao) > 0:
            simplex = [a, d, b]
            return GJK._triangle_case(simplex, direction)
        else:
            return True 