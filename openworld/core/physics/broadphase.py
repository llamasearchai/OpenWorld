import numpy as np
from typing import List, Dict, Tuple, Set
from ..physics.world import PhysicsObject

class Broadphase:
    def update(self, objects: List[PhysicsObject]) -> Set[Tuple[PhysicsObject, PhysicsObject]]:
        raise NotImplementedError
        
    def update_object(self, obj: PhysicsObject):
        raise NotImplementedError

class BruteForce(Broadphase):
    """Simple O(n^2) broadphase checking all pairs"""
    def update(self, objects):
        potential_pairs = set()
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                potential_pairs.add((objects[i], objects[j]))
        return potential_pairs
        
    def update_object(self, obj):
        pass  # No state to update

class Bound:
    """Represents a min or max bound of an object's AABB on an axis"""
    __slots__ = ('value', 'object_id', 'is_min')
    
    def __init__(self, value: float, object_id: str, is_min: bool):
        self.value = value
        self.object_id = object_id
        self.is_min = is_min
        
    def __lt__(self, other):
        return self.value < other.value or (
            self.value == other.value and self.is_min and not other.is_min
        )

class SweepAndPrune(Broadphase):
    """Optimized Sweep and Prune implementation with incremental updates"""
    def __init__(self):
        self.axes = {
            'x': [],
            'y': [],
            'z': []
        }
        self.object_bounds = {}  # object_id -> (min_x, max_x, min_y, max_y, min_z, max_z)
        self.object_map = {}     # object_id -> PhysicsObject
        
    def update(self, objects: List[PhysicsObject]) -> Set[Tuple[PhysicsObject, PhysicsObject]]:
        # Update object map
        self.object_map = {obj.object_id: obj for obj in objects}
        
        # Full sort if needed (first run or large changes)
        if not self.object_bounds or len(objects) != len(self.object_bounds):
            return self._full_update()
        else:
            return self._incremental_update()
            
    def _full_update(self):
        """Perform full sweep and prune on all axes"""
        self.object_bounds.clear()
        
        # Collect bounds for all objects
        for obj in self.object_map.values():
            self._update_object_bounds(obj)
            
        # Sort all axes
        for axis in self.axes.values():
            axis.sort()
            
        return self._find_overlaps()
        
    def _incremental_update(self):
        """Update bounds incrementally and maintain sorted order"""
        updated_objects = set()
        
        # Update changed objects
        for obj in self.object_map.values():
            old_bounds = self.object_bounds.get(obj.object_id)
            new_bounds = self._calculate_bounds(obj)
            
            if old_bounds != new_bounds:
                self._update_object_bounds(obj)
                updated_objects.add(obj.object_id)
                
        # Re-sort affected axes
        for axis_name in self.axes:
            axis = self.axes[axis_name]
            needs_sort = any(
                bound.object_id in updated_objects
                for bound in axis
            )
            if needs_sort:
                axis.sort()
                
        return self._find_overlaps()
        
    def _find_overlaps(self) -> Set[Tuple[PhysicsObject, PhysicsObject]]:
        """Find overlapping pairs across all axes"""
        # Initialize with all possible pairs from first axis
        active = set()
        potential_pairs = set()
        
        for bound in self.axes['x']:
            if bound.is_min:
                for obj_id in active:
                    pair = tuple(sorted((obj_id, bound.object_id)))
                    potential_pairs.add(pair)
                active.add(bound.object_id)
            else:
                active.remove(bound.object_id)
                
        # Check remaining axes
        for axis_name in ['y', 'z']:
            active = set()
            current_pairs = set()
            
            for bound in self.axes[axis_name]:
                if bound.is_min:
                    for obj_id in active:
                        pair = tuple(sorted((obj_id, bound.object_id)))
                        if pair in potential_pairs:
                            current_pairs.add(pair)
                    active.add(bound.object_id)
                else:
                    active.remove(bound.object_id)
                    
            potential_pairs = current_pairs
            
        # Convert to object pairs
        return {
            (self.object_map[a], self.object_map[b])
            for a, b in potential_pairs
        }
        
    def _calculate_bounds(self, obj: PhysicsObject) -> Tuple[float, ...]:
        """Calculate AABB bounds for an object"""
        if obj.geometry_type == 'sphere':
            r = obj.geometry_params['radius']
            min_x = obj.position[0] - r
            max_x = obj.position[0] + r
            min_y = obj.position[1] - r
            max_y = obj.position[1] + r
            min_z = obj.position[2] - r
            max_z = obj.position[2] + r
        elif obj.geometry_type == 'box':
            half_extents = np.array(obj.geometry_params['size']) / 2
            min_x = obj.position[0] - half_extents[0]
            max_x = obj.position[0] + half_extents[0]
            min_y = obj.position[1] - half_extents[1]
            max_y = obj.position[1] + half_extents[1]
            min_z = obj.position[2] - half_extents[2]
            max_z = obj.position[2] + half_extents[2]
        else:
            raise ValueError(f"Unsupported geometry type: {obj.geometry_type}")
            
        return (min_x, max_x, min_y, max_y, min_z, max_z)
        
    def _update_object_bounds(self, obj: PhysicsObject):
        """Update stored bounds for an object"""
        bounds = self._calculate_bounds(obj)
        obj_id = obj.object_id
        
        # Remove old bounds if they exist
        if obj_id in self.object_bounds:
            for axis_name, axis in self.axes.items():
                axis[:] = [b for b in axis if b.object_id != obj_id]
                
        # Store new bounds
        self.object_bounds[obj_id] = bounds
        
        # Add new bounds to axes
        self.axes['x'].extend([
            Bound(bounds[0], obj_id, True),
            Bound(bounds[1], obj_id, False)
        ])
        self.axes['y'].extend([
            Bound(bounds[2], obj_id, True),
            Bound(bounds[3], obj_id, False)
        ])
        self.axes['z'].extend([
            Bound(bounds[4], obj_id, True),
            Bound(bounds[5], obj_id, False)
        ])
        
    def update_object(self, obj: PhysicsObject):
        """Update a single object's bounds in the broadphase"""
        if obj.object_id in self.object_map:
            self._update_object_bounds(obj)

class BVHTree(Broadphase):
    """Bounding Volume Hierarchy tree implementation"""
    def __init__(self):
        self.root = None
        
    def update(self, objects):
        # Rebuild or update BVH tree
        self._build_tree(objects)
        return self._query_pairs()
        
    def _build_tree(self, objects):
        # Build BVH tree from objects
        pass
        
    def _query_pairs(self):
        # Query tree for overlapping pairs
        pass 

    def _calculate_inheritance_cost(self, node: BVHNode, new_node: BVHNode) -> float:
        """Calculate the cost of inserting under this node"""
        if node.is_leaf():
            return 0
            
        combined_aabb = self._merge_aabb(node.aabb, new_node.aabb)
        new_sa = self._aabb_surface_area(combined_aabb)
        old_sa = self._aabb_surface_area(node.aabb)
        return new_sa - old_sa
        
    def _update_parent_aabbs(self, node: BVHNode):
        """Update AABBs up the tree after modification"""
        while node:
            old_aabb = node.aabb
            new_aabb = self._merge_aabb(node.left.aabb, 
                                      node.right.aabb if node.right else node.left.aabb)
            
            if self._aabb_equals(old_aabb, new_aabb):
                break
                
            node.aabb = new_aabb
            node = getattr(node, 'parent', None)
            
    def _should_rebuild(self) -> bool:
        """Determine if tree should be rebuilt based on quality metrics"""
        return self._insertion_counter > len(self.object_nodes) * self.threshold
        
    def _query_pairs(self) -> Set[Tuple[PhysicsObject, PhysicsObject]]:
        """Query the BVH for all potential collision pairs"""
        pairs = set()
        if not self.root:
            return pairs
            
        # Compare all leaf nodes
        leaves = self._get_all_leaves()
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                if self._aabb_overlap(leaves[i].aabb, leaves[j].aabb):
                    pair = (leaves[i].object, leaves[j].object)
                    pairs.add(pair)
                    
        return pairs
        
    def _get_all_leaves(self) -> List[BVHNode]:
        """Get all leaf nodes in the BVH"""
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                leaves.append(node)
            else:
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return leaves
        
    def _calculate_aabb(self, obj: PhysicsObject) -> Tuple[float, ...]:
        """Calculate AABB for an object"""
        if obj.geometry_type == 'sphere':
            r = obj.geometry_params['radius']
            return (
                obj.position[0] - r, obj.position[0] + r,
                obj.position[1] - r, obj.position[1] + r,
                obj.position[2] - r, obj.position[2] + r
            )
        elif obj.geometry_type == 'box':
            half = np.array(obj.geometry_params['size']) / 2
            return (
                obj.position[0] - half[0], obj.position[0] + half[0],
                obj.position[1] - half[1], obj.position[1] + half[1],
                obj.position[2] - half[2], obj.position[2] + half[2]
            )
        else:
            raise ValueError(f"Unsupported geometry type: {obj.geometry_type}")
            
    def _merge_aabb(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
        """Merge two AABBs"""
        return (
            min(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), max(a[3], b[3]),
            min(a[4], b[4]), max(a[5], b[5])
        )
        
    def _merge_aabbs(self, aabbs: List[Tuple[float, ...]]) -> Tuple[float, ...]:
        """Merge multiple AABBs"""
        if not aabbs:
            return (0,0,0,0,0,0)
        merged = list(aabbs[0])
        for aabb in aabbs[1:]:
            merged[0] = min(merged[0], aabb[0])
            merged[1] = max(merged[1], aabb[1])
            merged[2] = min(merged[2], aabb[2])
            merged[3] = max(merged[3], aabb[3])
            merged[4] = min(merged[4], aabb[4])
            merged[5] = max(merged[5], aabb[5])
        return tuple(merged)
        
    def _aabb_surface_area(self, aabb: Tuple[float, ...]) -> float:
        """Calculate surface area of AABB"""
        dx = aabb[1] - aabb[0]
        dy = aabb[3] - aabb[2]
        dz = aabb[5] - aabb[4]
        return 2 * (dx*dy + dy*dz + dz*dx)
        
    def _aabb_overlap(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        """Check if two AABBs overlap"""
        return (a[0] <= b[1] and a[1] >= b[0] and
                a[2] <= b[3] and a[3] >= b[2] and
                a[4] <= b[5] and a[5] >= b[4])
                
    def _aabb_equals(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        """Check if two AABBs are equal"""
        return (a[0] == b[0] and a[1] == b[1] and
                a[2] == b[2] and a[3] == b[3] and
                a[4] == b[4] and a[5] == b[5])
                
    def update_object(self, obj: PhysicsObject):
        """Update an object's position in the BVH"""
        if obj.object_id in self.object_nodes:
            new_aabb = self._calculate_aabb(obj)
            if not self._aabb_equals(self.object_nodes[obj.object_id].aabb, new_aabb):
                self._remove_object(obj.object_id)
                self._insert_object(obj, new_aabb) 