"""
Configuration classes for the OpenWorld platform.

This module provides configuration classes for different components
of the OpenWorld platform.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, ClassVar

from .utils.exceptions import ConfigurationError

# Path to the config directory, defaults to ~/.openworld
CONFIG_DIR = os.environ.get("OPENWORLD_CONFIG_DIR", 
                          str(Path.home() / ".openworld"))

@dataclass
class PhysicsConfig:
    """Configuration for physics module."""
    # Model parameters
    d_model: int = 1024
    n_layers: int = 8
    n_heads: int = 16
    
    # Physics parameters
    gravity_enabled: bool = True
    collision_enabled: bool = True
    fluid_enabled: bool = False
    conservation_loss_weight: float = 0.1
    gravity: float = 9.8
    time_step: float = 0.01
    
    # Modeling approaches
    use_graph_net: bool = True
    use_particle_based: bool = True
    constrain_physics: bool = True
    
    # Model activation function
    activation: str = "silu"

@dataclass
class LongContextConfig:
    """Configuration for LongContextTransformer."""
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    max_seq_len: int = 131072
    attention_type: str = "sliding_window"
    window_size: int = 4096
    use_alibi: bool = True
    use_flash_attention: bool = True
    rope_base: int = 10000
    vocab_size: int = 32000
    
@dataclass
class APIConfig:
    """Configuration for the API server."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    auth_required: bool = False
    log_level: str = "INFO"

@dataclass
class OpenWorldConfig:
    """Main configuration for the OpenWorld platform."""
    # Component configurations
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    transformer: LongContextConfig = field(default_factory=LongContextConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # General settings
    learning_rate: float = 0.01
    reasoning_strategies: List[str] = field(
        default_factory=lambda: ["causal", "counterfactual", "temporal", "spatial"]
    )
    data_dir: str = field(default_factory=lambda: os.path.join(CONFIG_DIR, "data"))
    
    # Class methods for loading/saving
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OpenWorldConfig':
        """Create a config instance from a dictionary."""
        return cls(
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            transformer=LongContextConfig(**config_dict.get('transformer', {})),
            api=APIConfig(**config_dict.get('api', {})),
            learning_rate=config_dict.get('learning_rate', 0.01),
            reasoning_strategies=config_dict.get('reasoning_strategies', 
                                               ["causal", "counterfactual", "temporal", "spatial"]),
            data_dir=config_dict.get('data_dir', os.path.join(CONFIG_DIR, "data"))
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'OpenWorldConfig':
        """Load configuration from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'physics': {k: v for k, v in self.physics.__dict__.items()},
            'transformer': {k: v for k, v in self.transformer.__dict__.items()},
            'api': {k: v for k, v in self.api.__dict__.items()},
            'learning_rate': self.learning_rate,
            'reasoning_strategies': self.reasoning_strategies,
            'data_dir': self.data_dir
        }
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except IOError as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}") 