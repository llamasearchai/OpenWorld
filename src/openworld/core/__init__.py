"""Core components of the OpenWorld simulation and modeling platform."""

# Expose key modules and classes at the openworld.core level if desired.
# This makes them easier to import for users of the library.

# from .config import OpenWorldConfig # Config might be loaded globally or passed around

# From subdirectories of core:
from .ai import * # Expose AI components like LongContextTransformer
from .physics import * # Expose PhysicsEngine, etc.
# from .battery import * # Placeholder for when battery module is integrated
# from .solar import * # Placeholder for when solar module is integrated
# from .master_models import * # If master_models.py is created for unified models

# It's often better to be explicit with __all__ if using import *
# but for a core module, sometimes exposing submodules directly is convenient.
# Consider listing specific classes/functions in __all__ for better control.

__all__ = [
    # List key classes/functions you want to be part of the public API of openworld.core
    # For example, from physics:
    "PhysicsEngine",
    # From ai (if LongContextTransformer is in its __all__):
    # "LongContextTransformer", "LongContextConfig", 
    # ... and so on for other modules.
] 

# If LongContextTransformer and LongContextConfig are indeed in core.ai.__all__,
# and PhysicsEngine in core.physics.__all__, then the above won't pick them up directly with `from .ai import *`
# A more robust __all__ for openworld.core might be:

# from .ai import LongContextTransformer, LongContextConfig # Assuming these are in core.ai.__all__
# from .physics import PhysicsEngine # Assuming this is in core.physics.__all__

# __all__.extend([
#     "LongContextTransformer", "LongContextConfig",
#     "PhysicsEngine"
# ])

# For now, keeping it simple. The user can refine __all__ lists in submodules and here
# based on what they want as the direct public API of `openworld.core`. 