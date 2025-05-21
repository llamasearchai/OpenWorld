#!/usr/bin/env python
"""
OpenWorld Basic Physics Example
============================

This example demonstrates how to use the OpenWorld physics simulation
to create a simple projectile motion simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from openworld.core.physics import PhysicsWorld, Particle
from openworld.utils.logging import configure_logging, get_logger
from openworld.utils.units import u

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

def run_example():
    """Run a basic physics simulation example"""
    print("OpenWorld Basic Physics Example")
    print("===========================")
    
    # 1. Create a physics world
    print("\nInitializing physics world...")
    
    # Create a 2D world with standard gravity
    world = PhysicsWorld(dimensions=2, gravity=9.8)
    logger.info(f"Created physics world with ID: {world.world_id}")
    
    # 2. Add particles
    print("\nAdding particles to the world...")
    
    # Create a projectile
    projectile = Particle(
        position=[0, 0],  # Start at origin
        mass=0.1,         # 100g
        velocity=[10, 15], # Initial velocity (10 m/s horizontal, 15 m/s vertical)
        radius=0.05,      # 5cm radius
        obj_id="projectile1"
    )
    world.add_object(projectile)
    
    # Add another projectile with different initial conditions
    projectile2 = Particle(
        position=[0, 0],   # Start at origin
        mass=0.2,          # 200g
        velocity=[15, 10], # Different initial velocity
        radius=0.07,       # 7cm radius
        obj_id="projectile2"
    )
    world.add_object(projectile2)
    
    # 3. Run simulation
    print("\nRunning physics simulation...")
    
    # Simulation parameters
    dt = 0.01  # Time step (seconds)
    duration = 3.0  # Simulation duration (seconds)
    steps = int(duration / dt)
    
    # Record initial state
    for obj in world.objects.values():
        obj.record_history(world.time)
    
    # Simulate!
    start_time = time.time()
    for step in range(steps):
        # For each object, compute acceleration from forces
        for obj_id, obj in world.objects.items():
            # Get total force (both object-specific and global)
            total_force = obj.get_total_force(world.time.magnitude)
            
            # Add contributions from global forces
            for force_func in world.global_forces:
                force_contrib = force_func(obj, world.time.magnitude)
                total_force += force_contrib
            
            # Compute acceleration (F = ma, so a = F/m)
            acceleration = total_force / obj.mass
            
            # Update object state
            obj.update_state(acceleration, dt * u.second)
            
            # Simplified ground collision detection - just bounce with damping
            if obj.position[1].magnitude < 0:
                # Set position to ground
                obj.position = [obj.position[0].magnitude, 0] * u.meter
                # Reverse vertical velocity with damping
                obj.velocity = [obj.velocity[0].magnitude, -0.8 * obj.velocity[1].magnitude] * u.meter / u.second
        
        # Advance world time
        world.advance_time(dt * u.second)
        
        # Record state
        for obj in world.objects.values():
            obj.record_history(world.time)
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.3f} seconds for {steps} time steps")
    
    # 4. Plot results
    print("\nGenerating plots...")
    
    # Get trajectories for visualization
    histories = world.get_object_histories()
    
    # Plot trajectories
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'c', 'm']
    for i, (obj_id, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        
        # Convert to numpy array for easier handling
        positions = np.array(history["position"])
        
        # Plot trajectory
        plt.plot(positions[:, 0], positions[:, 1], 
                 color + '-', linewidth=2, label=f"Trajectory {obj_id}")
        
        # Mark start and end points
        plt.plot(positions[0, 0], positions[0, 1], color + 'o', markersize=8)
        plt.plot(positions[-1, 0], positions[-1, 1], color + 'x', markersize=8)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Projectile Trajectories')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("projectile_trajectories.png", dpi=300)
    
    # 5. Plot time series of position and velocity
    print("\nGenerating time series plots...")
    
    # Get first object for detailed time series
    obj_id = list(histories.keys())[0]
    history = histories[obj_id]
    
    # Convert to numpy array
    times = np.array(history["time"])
    positions = np.array(history["position"])
    velocities = np.array(history["velocity"])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Position plot
    ax1.plot(times, positions[:, 0], 'b-', linewidth=2, label='X Position')
    ax1.plot(times, positions[:, 1], 'r-', linewidth=2, label='Y Position')
    ax1.set_ylabel('Position (m)')
    ax1.set_title(f'Position vs Time for {obj_id}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity plot
    ax2.plot(times, velocities[:, 0], 'b-', linewidth=2, label='X Velocity')
    ax2.plot(times, velocities[:, 1], 'r-', linewidth=2, label='Y Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title(f'Velocity vs Time for {obj_id}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("projectile_time_series.png", dpi=300)
    
    print("\nExample complete! Output files:")
    print("- projectile_trajectories.png")
    print("- projectile_time_series.png")

if __name__ == "__main__":
    run_example() 