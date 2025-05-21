"""OpenWorld API Server and Client."""

# Makes the FastAPI app and API client easily accessible.

# from .server import app as openworld_api_app # If server.py defines 'app'
from .client import OpenWorldApiClient
# from .schemas import ... # Expose key Pydantic schemas if needed

__all__ = [
    # "openworld_api_app",
    "OpenWorldApiClient",
] 