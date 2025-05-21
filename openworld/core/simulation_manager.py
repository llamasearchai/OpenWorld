import threading
from typing import Dict, Any, List
from ..core.physics import PhysicsWorld
from ..core.solar import SolarSimulation
from ..core.battery import BatterySimulation

class SimulationManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._physics_worlds: Dict[str, PhysicsWorld] = {}
        self._solar_sims: Dict[str, SolarSimulation] = {}
        self._battery_sims: Dict[str, BatterySimulation] = {}
        
    def create_physics_world(self, **kwargs) -> str:
        with self._lock:
            world = PhysicsWorld(**kwargs)
            self._physics_worlds[world.id] = world
            return world.id
            
    def get_physics_world(self, world_id: str) -> PhysicsWorld:
        with self._lock:
            return self._physics_worlds[world_id]
            
    def create_solar_simulation(self, device=None) -> str:
        with self._lock:
            sim = SolarSimulation(device)
            self._solar_sims[sim.sim_id] = sim
            return sim.sim_id
            
    def run_solar_simulation(self, sim_id: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            sim = self._solar_sims[sim_id]
            return sim.run_iv_curve(conditions)
            
    def create_battery_simulation(self, **kwargs) -> str:
        with self._lock:
            sim = BatterySimulation(**kwargs)
            self._battery_sims[sim.sim_id] = sim
            return sim.sim_id
            
    def run_battery_protocol(self, sim_id: str, protocol: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            sim = self._battery_sims[sim_id]
            return sim.run_protocol(protocol)
            
    def cleanup(self, sim_type: str, sim_id: str):
        with self._lock:
            if sim_type == "physics":
                del self._physics_worlds[sim_id]
            elif sim_type == "solar":
                del self._solar_sims[sim_id]
            elif sim_type == "battery":
                del self._battery_sims[sim_id] 