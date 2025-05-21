from fastapi import FastAPI, APIRouter, HTTPException, Body, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
import concurrent.futures

# Assuming settings and logger are available via openworld.utils
# from openworld.utils.config import settings # If needed for server config
from openworld.utils.logging import get_logger # Fallback if not available, real one preferred.
from openworld.core.physics import PhysicsWorld
from openworld.core.solar import SolarSimulation
from openworld.core.battery import BatterySimulation

# Placeholder for Pydantic models for request/response validation if not defined elsewhere
# from .schemas import (
#     PhysicsSimParams, PhysicsSimResult,
#     BatterySimParams, BatterySimResult,
#     SolarSimParams, SolarSimResult,
#     AIQuery, AIResult,
#     SimulationInstance, SimulationStatus
# )

logger = get_logger(__name__)

app = FastAPI(
    title="OpenWorld API",
    version="2.0",
    description="API for OpenWorld simulation engine and AI reasoning.",
)

# --- Simulation Management (Conceptual - needs backing store and proper instantiation) ---
# In-memory store for active simulations (replace with persistent store and proper class structure in production)
active_simulations: Dict[str, Dict[str, Any]] = {
    "physics": {}, # Stores physics world instances: world_id -> world_data
    "battery": {}, # Stores battery simulation instances: sim_id -> sim_data
    "solar": {}    # Stores solar simulation instances: sim_id -> sim_data
}
_sim_id_counter = 0

def _generate_sim_id(sim_type_prefix: str) -> str:
    global _sim_id_counter
    _sim_id_counter += 1
    return f"{sim_type_prefix}_{_sim_id_counter:04d}"

# --- API Routers ---
simulations_router = APIRouter(prefix="/simulations", tags=["Simulations"])
ai_router = APIRouter(prefix="/ai", tags=["AI Reasoning"])

# === Simulation Lifecycle & Execution Endpoints ===

# Physics Simulation
@simulations_router.post("/physics/worlds", summary="Create a new physics world instance")
async def create_physics_world_endpoint(create_params: Dict[str, Any] = Body(...)):
    logger.info(f"Creating physics world with params: {create_params}")
    # 실제 PhysicsEngine 인스턴스 생성 및 관리 로직 필요
    world_id = _generate_sim_id("physworld")
    active_simulations["physics"][world_id] = {"status": "created", "params": create_params, "type": "physics_world", "objects": {}}
    # result = actual_physics_world_manager.create_world(**create_params)
    # world_id = result.id
    return {"world_id": world_id, "status": "created", "details": active_simulations["physics"][world_id]}

@simulations_router.post("/physics/worlds/{world_id}/objects", summary="Add an object to a physics world")
async def add_physics_object_endpoint(world_id: str, object_config: Dict[str, Any] = Body(...)):
    if world_id not in active_simulations["physics"]:
        raise HTTPException(status_code=404, detail=f"Physics world '{world_id}' not found")
    # 실제 PhysicsWorld 객체에 object_config 전달 로직 필요
    logger.info(f"Adding object to physics world '{world_id}': {object_config}")
    obj_id = object_config.get("id", f"obj_{len(active_simulations['physics'][world_id]['objects']) + 1}")
    active_simulations["physics"][world_id]["objects"][obj_id] = object_config
    # actual_physics_world_manager.get_world(world_id).add_object(**object_config)
    return {"world_id": world_id, "object_id": obj_id, "status": "object_added"}

@simulations_router.post("/physics/run", summary="Run/advance a physics simulation")
async def run_physics_simulation_endpoint(params: Dict[str, Any] = Body(...)):
    world_id = params.get("world_id")
    if not world_id or world_id not in active_simulations["physics"]:
        raise HTTPException(status_code=404, detail=f"Physics world '{world_id}' not found or not specified for run.")
    logger.info(f"Received request to run physics simulation for world '{world_id}' with params: {params.get('run_params')}")
    # result = actual_physics_world_manager.get_world(world_id).run_simulation(**params.get('run_params'))
    active_simulations["physics"][world_id]["status"] = "ran_placeholder"
    active_simulations["physics"][world_id]["last_run_params"] = params.get('run_params')
    return {"status": "placeholder_success", "world_id": world_id, "message": "Physics simulation run (placeholder)", "results": "placeholder_physics_results"}

# Battery Simulation
@simulations_router.post("/battery", summary="Create a new battery simulation instance")
async def create_battery_simulation_endpoint(create_params: Dict[str, Any] = Body(...)):
    logger.info(f"Creating battery simulation with params: {create_params}")
    sim_id = _generate_sim_id("battsim")
    active_simulations["battery"][sim_id] = {"status": "created", "params": create_params, "type": "battery_simulation"}
    # result = actual_battery_manager.create_simulation(**create_params)
    # sim_id = result.id
    return {"sim_id": sim_id, "status": "created", "details": active_simulations["battery"][sim_id]}

@simulations_router.post("/battery/simulations/{sim_id}/run", response_model=BatterySimResponse)
async def run_battery_simulation(
    sim_id: str,
    params: BatteryRunProtocolParams,
    background_tasks: BackgroundTasks
):
    if sim_id not in active_simulations["battery"]:
        raise HTTPException(status_code=404, detail=f"Battery simulation '{sim_id}' not found")
    
    sim = active_simulations["battery"][sim_id]
    
    def run_simulation_task():
        try:
            results = sim.run_protocol(params.protocol)
            active_simulations["battery"][sim_id].results = results
            return results
        except Exception as e:
            logger.error(f"Battery simulation failed: {str(e)}")
            raise e
    
    # Run in background thread
    future = executor.submit(run_simulation_task)
    background_tasks.add_task(lambda: future.result())  # Ensures task completes
    
    return BatterySimResponse(
        sim_id=sim_id,
        status="running",
        message="Simulation started in background"
    )

@simulations_router.get("/battery/simulations/{sim_id}/results", response_model=BatterySimResponse)
async def get_battery_results(sim_id: str):
    if sim_id not in active_simulations["battery"]:
        raise HTTPException(status_code=404, detail=f"Battery simulation '{sim_id}' not found")
    
    sim = active_simulations["battery"][sim_id]
    if not sim.results:
        raise HTTPException(status_code=400, detail="No results available - simulation not run yet")
    
    return BatterySimResponse(
        sim_id=sim_id,
        status="completed",
        results=sim.results
    )

# Solar Simulation
@simulations_router.post("/solar/simulations", response_model=SolarSimResponse)
async def create_solar_simulation(params: SolarSimCreateParams):
    try:
        sim = SolarSimulation(params.device_name)
        return SolarSimResponse(
            sim_id=sim.sim_id,
            status="created",
            details={"device": sim.device.to_dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@simulations_router.post("/solar/simulations/{sim_id}/run", response_model=SolarSimResponse)
async def run_solar_simulation(sim_id: str, params: SolarRunConditionsParams):
    try:
        sim = get_simulation(sim_id)  # Would be from a simulation manager
        results = sim.run_iv_curve(params.dict())
        return SolarSimResponse(
            sim_id=sim_id,
            status="completed",
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# General Simulation Management (as used by CLI)
@simulations_router.get("", summary="List all active simulation instances")
async def list_simulations_endpoint(sim_type: Optional[str] = None):
    logger.info(f"Request to list simulations. Filter type: {sim_type}")
    if sim_type:
        if sim_type in active_simulations: # e.g. sim_type = "physics", "battery", "solar"
             # Return IDs of main simulation instances, not sub-components like physics_world objects
            return {sim_type: list(s_id for s_id, s_data in active_simulations[sim_type].items() if "world" not in s_data.get("type",""))}
        elif f"{sim_type}_world" in active_simulations["physics"]: # Temp workaround for phys_world
             return {sim_type: list(active_simulations["physics"].keys())}
        else:
            raise HTTPException(status_code=404, detail=f"Unknown simulation type '{sim_type}' or no active simulations of that type.")
    
    all_sim_ids = {}
    for s_type, sims_of_type in active_simulations.items():
        all_sim_ids[s_type] = list(s_id for s_id, s_data in sims_of_type.items()) # List all IDs including worlds for now
    return all_sim_ids
    

@simulations_router.get("/{sim_id_full}/status", summary="Get the status of a specific simulation instance")
async def get_simulation_status_endpoint(sim_id_full: str):
    logger.info(f"Request for status of simulation: {sim_id_full}")
    for sim_type_group_name, sim_type_group_content in active_simulations.items():
        if sim_id_full in sim_type_group_content:
            return {"sim_id": sim_id_full, "status": sim_type_group_content[sim_id_full].get("status", "unknown"), "details": sim_type_group_content[sim_id_full]}
    raise HTTPException(status_code=404, detail=f"Simulation '{sim_id_full}' not found.")

@simulations_router.delete("/{sim_id_full}", summary="Delete a simulation instance")
async def delete_simulation_endpoint(sim_id_full: str):
    logger.info(f"Request to delete simulation: {sim_id_full}")
    for sim_type_group_name, sim_type_group_content in active_simulations.items():
        if sim_id_full in sim_type_group_content:
            del sim_type_group_content[sim_id_full]
            logger.info(f"Deleted simulation '{sim_id_full}' from type group '{sim_type_group_name}'.")
            return {"message": f"Simulation '{sim_id_full}' deleted successfully."}
    raise HTTPException(status_code=404, detail=f"Simulation '{sim_id_full}' not found to delete.")


# === AI Reasoning Endpoint ===

@ai_router.post("/reason", summary="Perform AI reasoning based on a query and strategy")
async def ai_reason_endpoint(payload: Dict[str, Any] = Body(...)): # Assuming payload like {"problem": ..., "domain": ...}
    problem = payload.get("problem")
    domain_strategy = payload.get("domain") # CLI sends 'strategy' as 'domain'
    logger.info(f"Received AI reasoning request. Problem: '{problem[:50]}...', Strategy: '{domain_strategy}'")
    if not problem or not domain_strategy:
        raise HTTPException(status_code=400, detail="Missing 'problem' or 'domain' in request body.")
    
    # Placeholder for calling the actual WorldModelAgent's reasoning logic
    # from openworld.agent.core import WorldModelAgent, WorldModelStrategy # Dynamic import
    # from openworld.config import AgentConfig, PhysicsConfig, LongContextConfig # And configs
    # agent = WorldModelAgent(AgentConfig(), PhysicsConfig(), LongContextConfig()) # Needs proper init
    # result = agent.reason(query=problem, strategy=WorldModelStrategy(domain_strategy))
    # return result # Should match ReasoningResult schema
    
    return {"status": "placeholder_success", "message": "AI reasoning performed (placeholder)", "query": problem, "strategy": domain_strategy, "result": {"inferences": ["Placeholder AI inference 1", "Placeholder AI inference 2"], "confidence": 0.75}}


app.include_router(simulations_router)
app.include_router(ai_router)

# Health check endpoint
@app.get("/health", summary="Health check for the API server")
async def health_check():
    return {"status": "healthy", "message": "OpenWorld API is running."}

# Example of how to run this server for development (though CLI `server` command is preferred):
# if __name__ == "__main__":
#     import uvicorn
#     # Ensure openworld.utils.logging is configured if running directly
#     # from openworld.utils.logging import setup_logging
#     # setup_logging() 
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 