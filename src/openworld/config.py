from dataclasses import dataclass, field
from typing import Optional

# Forward-referencing type hints for classes that will be defined later
# This is important as we consolidate configurations.

@dataclass
class PhysicsConfig:
    """Configuration for the core physics engine."""
    d_model: int = 512 # Example from worldmodel_backend/physics_module.py
    gravity_enabled: bool = True
    time_step: float = 0.01
    # Add other physics-specific settings here

@dataclass
class LongContextConfig:
    """Configuration for the Long Context Transformer."""
    d_model: int = 4096 # Example from worldmodel_backend/long_context.py
    n_heads: int = 32
    n_layers: int = 32
    max_seq_len: int = 131072
    # Add other transformer-specific settings here

@dataclass
class AgentConfig:
    """Configuration for the AI Agent."""
    reasoning_strategies: list = field(default_factory=lambda: ['causal', 'counterfactual', 'temporal', 'spatial'])
    learning_rate: float = 0.01
    # Add other agent-specific settings here

@dataclass
class BatteryConfig:
    """Configuration for the Battery Module."""
    default_parameter_set: str = "graphite_nmc"
    # Add other battery-specific settings here

@dataclass
class SolarConfig:
    """Configuration for the Solar Module."""
    default_cell_type: str = "perovskite"
    # Add other solar-specific settings here

@dataclass
class APIConfig:
    """Configuration for the API Server."""
    host: str = "127.0.0.1"
    port: int = 8000
    # Add other API-specific settings here

@dataclass
class OpenWorldConfig:
    """Main configuration class for the OpenWorld application."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    transformer: LongContextConfig = field(default_factory=LongContextConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Global settings
    log_level: str = "INFO"
    data_directory: str = "./data"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OpenWorldConfig':
        """Creates OpenWorldConfig from a dictionary, allowing nested configs."""
        # This method needs to be more robust to handle partial configs and defaults.
        # The scaffold in physics-engine-APIENDPOINIT.txt uses pydantic-settings for this,
        # which is a better approach for loading from .env and dicts.
        # For now, this is a simplified version.
        
        return cls(
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            transformer=LongContextConfig(**config_dict.get('transformer', {})),
            agent=AgentConfig(**config_dict.get('agent', {})),
            battery=BatteryConfig(**config_dict.get('battery', {})),
            solar=SolarConfig(**config_dict.get('solar', {})),
            api=APIConfig(**config_dict.get('api', {})),
            log_level=config_dict.get('log_level', "INFO"),
            data_directory=config_dict.get('data_directory', "./data")
        )

# Example of how Pydantic-settings might be used (conceptual, not directly from scaffold yet)
# from pydantic_settings import BaseSettings
# class Settings(BaseSettings):
#     physics: PhysicsConfig = PhysicsConfig()
#     transformer: LongContextConfig = LongContextConfig()
#     # ... and so on for all configs
#     OPENAI_API_KEY: Optional[str] = None
#     LOG_LEVEL: str = "INFO"

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         extra = "ignore" # or "allow" if you want to load arbitrary env vars

# settings = Settings() # This would be the global settings object 