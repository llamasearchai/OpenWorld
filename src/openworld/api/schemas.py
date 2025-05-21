from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# === General Utility Schemas ===

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

# === Physics Simulation Schemas ===

# Configuration for creating a physics world
class PhysicsWorldCreateParams(BaseModel):
    world_id: Optional[str] = Field(None, description="Optional user-defined ID for the world. If None, one will be generated.")
    dimensions: int = Field(3, ge=2, le=3, description="Number of dimensions for the physics simulation (2 or 3).")
    gravity: List[float] = Field([0.0, -9.81, 0.0], description="Gravitational acceleration vector.")
    time_step: float = Field(0.01, gt=0, description="Simulation time step (delta t) in seconds.")
    solver_iterations: int = Field(10, gt=0, description="Number of iterations for the physics solver.")
    # Add other relevant world parameters: e.g., restitution, friction defaults

class PhysicsWorldResponse(BaseModel):
    world_id: str
    status: str
    details: Dict[str, Any] # Could be more specific later

# Configuration for a physics object
class PhysicsObjectConfigSchema(BaseModel):
    object_id: str = Field(description="Unique ID for the object within its world.")
    object_type: str = Field(description="Type of object (e.g., 'sphere', 'cube', 'capsule', 'mesh').")
    position: List[float] = Field([0.0, 0.0, 0.0], description="Initial position [x, y, z] or [x, y].")
    orientation: List[float] = Field([0.0, 0.0, 0.0, 1.0], description="Initial orientation as quaternion [x, y, z, w].")
    mass: float = Field(1.0, gt=0, description="Mass of the object in kg.")
    friction: float = Field(0.5, ge=0, description="Coefficient of friction.")
    restitution: float = Field(0.5, ge=0, le=1, description="Coefficient of restitution (bounciness).")
    linear_damping: float = Field(0.05, ge=0, description="Linear damping factor.")
    angular_damping: float = Field(0.05, ge=0, description="Angular damping factor.")
    kinematic: bool = Field(False, description="Is the object kinematic (not affected by forces, but can move)?")
    # Geometry-specific parameters
    geometry_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the object_type's geometry (e.g., {'radius': 0.5} for sphere).")
    # Example geometry_params:
    # For 'sphere': {'radius': float}
    # For 'cube': {'size': [float, float, float]} (half-extents)
    # For 'capsule': {'radius': float, 'height': float}

class AddObjectRequest(BaseModel):
    # object_id is optional here, if not provided, server can generate one or use one from config.
    config: PhysicsObjectConfigSchema

class ObjectResponse(BaseModel):
    world_id: str
    object_id: str
    status: str
    message: Optional[str] = None

# Parameters for running/stepping a physics simulation
class PhysicsRunParamsSchema(BaseModel):
    duration: Optional[float] = Field(None, gt=0, description="Total duration to simulate in seconds.")
    num_steps: Optional[int] = Field(None, gt=0, description="Number of simulation steps to perform. Provide duration or num_steps.")
    # Potentially other run-time controls

class PhysicsRunResultSchema(BaseModel):
    world_id: str
    status: str
    steps_taken: int
    simulation_time_elapsed: float
    object_states: Optional[Dict[str, Any]] = Field(None, description="Current states of objects (positions, orientations, velocities).")
    # Could include events, collisions, etc.

# === AI Reasoning Schemas ===
class AIQueryRequest(BaseModel):
    problem: str
    domain: str # Maps to strategy in CLI
    # context: Optional[Dict[str, Any]] = None # Optional additional context

class AIReasoningResult(BaseModel):
    query: str
    strategy: str # Renamed from 'domain' to be consistent with agent's terminology
    inferences: List[str]
    supporting_evidence: Optional[List[Any]] = None
    confidence: Optional[float] = None
    # Potentially other fields from ReasoningResult TypedDict in engine.py


# === Battery Simulation Schemas ===
class BatterySimCreateParams(BaseModel):
    sim_id: Optional[str] = None
    model_type: str = Field("SPM", description="e.g., SPM, DFN")
    parameter_set_name: str = Field("default_lithium_ion", description="Name of the parameter set to use.")
    initial_soc: float = Field(0.5, ge=0, le=1, description="Initial State of Charge.")
    temperature_K: float = Field(298.15, gt=0, description="Operating temperature in Kelvin.")

class BatteryProtocolStep(BaseModel):
    step_type: str = Field(description="e.g., 'charge', 'discharge', 'rest', 'cv_charge', 'cc_charge'")
    value: float = Field(description="Value for the step (e.g., current in A, voltage in V, duration in s).")
    unit: str = Field(description="Unit for the value (e.g., 'A', 'V', 's', 'C-rate').")
    duration: Optional[float] = Field(None, description="Duration of this step in seconds (if applicable).")
    # Add termination conditions, etc.

class BatteryRunProtocolParams(BaseModel):
    protocol: List[BatteryProtocolStep]

class BatterySimResponse(BaseModel):
    sim_id: str
    status: str
    details: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None # e.g., time-series data

# === Solar Simulation Schemas ===
class SolarSimCreateParams(BaseModel):
    sim_id: Optional[str] = None
    device_name: str = Field("default_silicon_pn", description="Name or type of the solar cell device to simulate.")
    absorber_thickness_nm: Optional[float] = Field(300.0, description="Absorber thickness in nm, if applicable for standard device.")
    # Or detailed device_config_dict: Optional[Dict[str, Any]] = None for custom devices

class SolarRunConditionsParams(BaseModel):
    light_intensity_W_m2: float = Field(1000.0, description="Incident light intensity in W/m^2.")
    temperature_K: float = Field(300.0, description="Operating temperature in Kelvin.")
    wavelengths_nm: Optional[List[float]] = Field(None, description="Specific wavelengths for spectral response, or None for broadband.")
    # IV curve parameters
    voltage_min_V: float = Field(-0.5, description="Minimum voltage for IV sweep.")
    voltage_max_V: float = Field(1.0, description="Maximum voltage for IV sweep.")
    voltage_steps: int = Field(100, description="Number of steps for IV sweep.")


class SolarSimResponse(BaseModel):
    sim_id: str
    status: str
    details: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None # e.g., IV curve data, efficiency 