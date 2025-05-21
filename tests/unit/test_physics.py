import pytest
from openworld.core.physics import PhysicsWorld, PhysicsObject

def test_world_creation():
    world = PhysicsWorld(dimensions=3)
    assert world.dimensions == 3
    assert len(world.gravity) == 3
    assert world.time_step == 0.01

def test_object_management():
    world = PhysicsWorld()
    obj_id = world.add_object({
        'object_id': 'test',
        'position': [0, 0, 0],
        'orientation': [0, 0, 0, 1],
        'mass': 1.0
    })
    assert obj_id == 'test'
    assert 'test' in world.objects

def test_simulation_step():
    world = PhysicsWorld()
    world.add_object({...})
    initial_y = world.objects['test'].position[1]
    world._step()
    assert world.objects['test'].position[1] < initial_y  # Should fall due to gravity 