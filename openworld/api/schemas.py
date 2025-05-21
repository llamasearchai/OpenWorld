"""
API schema definitions for the OpenWorld platform.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator

# Request schemas

class SimulationType(str, Enum):
    """Types of simulations supported."""
    PHYSICS = "physics"
    BATTERY = "battery"
    SOLAR = "solar"

class PhysicsSimRequest(BaseModel):
    """Request schema for physics simulations."""
    simulation_type: SimulationType = SimulationType.PHYSICS
    duration: float = Field(..., description="Simulation duration in seconds")
    time_step: float = Field(0.01, description="Time step in seconds")
    gravity: Union[float, List[float]] = Field(9.8, description="Gravity value or vector")
    dimensions: int = Field(3, description="Number of dimensions (2 or 3)")
    objects: List[Dict[str, Any]] = Field(..., description="Objects to simulate")
    constraints: Optional[List[Dict[str, Any]]] = None
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v not in [2, 3]:
            raise ValueError('Dimensions must be 2 or 3')
        return v

class BatterySimRequest(BaseModel):
    """Request schema for battery simulations."""
    simulation_type: SimulationType = SimulationType.BATTERY
    model_type: str = Field(
        "DFN",  # Newman's Doyle-Fuller-Newman model
        description="Battery model type (DFN, SPM, etc.)"
    )
    duration: float = Field(..., description="Simulation duration in seconds")
    time_step: float = Field(0.1, description="Time step in seconds")
    parameters: Dict[str, Any] = Field(
        ...,
        description="Battery parameters (capacity, resistance, etc.)"
    )
    current_profile: List[Dict[str, float]] = Field(
        ...,
        description="Current profile for charge/discharge"
    )
    temperature: Optional[float] = Field(
        25.0,
        description="Operating temperature in Celsius"
    )
    include_stress: bool = Field(
        False,
        description="Whether to include mechanical stress calculations"
    )

class SolarSimRequest(BaseModel):
    """Request schema for solar cell simulations."""
    simulation_type: SimulationType = SimulationType.SOLAR
    model_type: str = Field(
        "DriftDiffusion",
        description="Solar cell model type"
    )
    duration: Optional[float] = None
    spectrum: str = Field(
        "AM1.5G",
        description="Solar spectrum to use (AM1.5G, etc.)"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Solar cell parameters"
    )
    voltage_sweep: Optional[Dict[str, float]] = Field(
        None,
        description="Voltage sweep parameters for I-V curves"
    )
    illumination: Optional[float] = Field(
        1.0,
        description="Light intensity in suns (1.0 = standard illumination)"
    )

class AIRequest(BaseModel):
    """Request schema for AI-assisted analysis and optimization."""
    query: str = Field(..., description="Natural language query or instruction")
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Simulation context or results to analyze"
    )
    simulation_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameters for any simulations requested"
    )
    reasoning_strategy: Optional[str] = Field(
        "causal",
        description="Type of reasoning to perform (causal, counterfactual, etc.)"
    )

# Result schemas

class PhysicsResult(BaseModel):
    """Result schema for physics simulations."""
    simulation_type: SimulationType = SimulationType.PHYSICS
    duration: float = Field(..., description="Actual simulation duration in seconds")
    time_steps: int = Field(..., description="Number of time steps performed")
    objects: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Final state of all objects"
    )
    trajectories: Optional[Dict[str, Dict[str, List[Any]]]] = Field(
        None,
        description="Object trajectories over time"
    )
    energy: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Energy metrics over time"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the simulation"
    )

class BatteryResult(BaseModel):
    """Result schema for battery simulations."""
    simulation_type: SimulationType = SimulationType.BATTERY
    duration: float = Field(..., description="Actual simulation duration in seconds")
    time_steps: int = Field(..., description="Number of time steps performed")
    voltage: List[float] = Field(..., description="Voltage over time")
    current: List[float] = Field(..., description="Current over time")
    soc: List[float] = Field(..., description="State of charge over time")
    temperature: Optional[List[float]] = Field(None, description="Temperature over time")
    degradation: Optional[float] = Field(None, description="Estimated degradation")
    stress: Optional[Dict[str, Any]] = Field(
        None,
        description="Mechanical stress data (if requested)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the simulation"
    )

class SolarResult(BaseModel):
    """Result schema for solar cell simulations."""
    simulation_type: SimulationType = SimulationType.SOLAR
    # I-V curve data
    voltage: List[float] = Field(..., description="Voltage points")
    current: List[float] = Field(..., description="Current at each voltage point")
    power: List[float] = Field(..., description="Power at each voltage point")
    # Performance metrics
    voc: float = Field(..., description="Open circuit voltage (V)")
    jsc: float = Field(..., description="Short circuit current density (mA/cm²)")
    ff: float = Field(..., description="Fill factor")
    efficiency: float = Field(..., description="Power conversion efficiency (%)")
    # Additional data
    carrier_density: Optional[Dict[str, Any]] = Field(
        None,
        description="Carrier density profiles"
    )
    band_diagram: Optional[Dict[str, Any]] = Field(
        None,
        description="Band diagram data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the simulation"
    )

class AISolution(BaseModel):
    """Result schema for AI-assisted analysis and optimization."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Natural language response")
    reasoning: List[str] = Field(
        ...,
        description="Reasoning steps used to arrive at the response"
    )
    simulation_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Results of any simulations performed"
    )
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Specific recommendations or optimizations"
    )
    graphs: Optional[Dict[str, Any]] = Field(
        None,
        description="Analytical graphs or data structures"
    ) 