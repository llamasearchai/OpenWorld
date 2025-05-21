"""
OpenWorld Agent - Unified Agent Subsystem for World Modeling
"""

# Core Agent Components (adjust paths based on where these will actually live in the new structure)
# Assuming they will be in src/openworld/agent/core.py or similar initially
# from .core import WorldModelAgent, MetaWorldModelAgent # Example: if core.py contains these

# Reasoning Engine (if specific to agent, otherwise might be in openworld.core.ai)
# from .reasoning import WorldModelEngine, NeuroReasoningEngine

# Memory Systems
from .memory.state_memory import WorldStateMemory # Assuming state_memory.py will be in src/openworld/agent/memory/
# from .memory.quantum_memory import QuantumStateMemory # If this is a separate file/concept

# Tools
from .tools.simulation_tool import SimulationTool # Assuming simulation_tool.py will be in src/openworld/agent/tools/
# from .tools.hyper_simulation_tool import HyperSimulationTool # If this is a separate file/concept

# Potentially agent-specific configurations or modules
# from .config import AgentSpecificConfig # Example
# from . DANTDOBE.FOO import BAR # physics_module -> this was PhysicsModule from the original __init__
# from ..core.ai.learning import LongContextTransformer, LongContextConfig # Assuming long_context.py moves to core.ai or similar

# from .inference_engine import RealTimeInferenceEngine, InferenceEngineConfig # If inference is agent-specific

# __all__ controls what `from openworld.agent import *` imports.
# It's good practice to define it explicitly.
__all__ = [
    # 'WorldModelAgent', 'MetaWorldModelAgent', # Add these back once their files are placed
    # 'WorldModelEngine', 'NeuroReasoningEngine',
    'WorldStateMemory', # 'QuantumStateMemory',
    'SimulationTool', # 'HyperSimulationTool',
    # 'PhysicsModule', 'PhysicsConfig', # PhysicsModule is likely part of openworld.core.physics now
    # 'LongContextTransformer', 'LongContextConfig', # Likely part of openworld.core.ai
    # 'RealTimeInferenceEngine', 'InferenceEngineConfig'
]

# Note: The original __init__ had PhysicsModule, LongContextTransformer, and RealTimeInferenceEngine.
# In the new structure:
# - PhysicsModule and PhysicsConfig would likely move to src/openworld/core/physics/
# - LongContextTransformer and LongContextConfig would likely move to src/openworld/core/ai/ (or stay in agent if very specific)
# - RealTimeInferenceEngine could be in agent/inference or core/ai/inference.
# The __all__ list and imports here are speculative until those modules are placed.
# I've kept some relevant ones and commented out those that are likely to move outside the agent directory. 