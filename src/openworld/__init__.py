import importlib.metadata
try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0" # Default if not installed

__author__ = "The OpenWorld Contributors" # Adapted from PhysicsGPT

# Placeholder for key module imports to make them available at the package level
# For example:
# from .core.physics.engine import PhysicsEngine
# from .agent.core import WorldModelAgent
# from .api.server import app as api_app

# It's generally good practice to keep this minimal and let users import submodules directly. 