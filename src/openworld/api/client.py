import httpx
import webbrowser
from typing import Dict, List, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

class OpenWorldApiClient:
    """Client for interacting with the OpenWorld API server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)
        logger.info(f"OpenWorldApiClient initialized for base URL: {base_url}")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        try:
            response = self.client.request(method, endpoint, **kwargs)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {e.request.url}: {e.response.text}")
            # Attempt to parse error detail from response json
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except:
                error_detail = e.response.text
            raise Exception(f"API Error ({e.response.status_code}): {error_detail}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {e.request.url}: {e}")
            raise Exception(f"Connection Error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred in API client: {e}")
            raise

    def simulate_battery(self, 
                         chemistry: str, 
                         capacity_ah: float, 
                         initial_soc: float, 
                         protocol: List[Dict[str, Any]],
                         # Add other relevant parameters as per actual API endpoint
                         ) -> Dict[str, Any]:
        """
        Placeholder for submitting a battery simulation request.
        Actual endpoint: /simulations/battery/run (example)
        """
        payload = {
            "chemistry": chemistry,
            "capacity_ah": capacity_ah,
            "initial_soc": initial_soc,
            "protocol": protocol
            # ... other params
        }
        logger.info(f"Submitting battery simulation with payload: {payload}")
        # This endpoint path is a placeholder and should match the actual server endpoint
        # return self._request("POST", "/simulations/battery/run", json=payload)
        logger.warning("simulate_battery client method is a placeholder. API endpoint needs implementation.")
        return {"job_id": "dummy_battery_job_123", "status": "submitted_placeholder", "message": "Endpoint not implemented"}

    def simulate_solar_cell(self, 
                              cell_type: str, 
                              layer_thicknesses: Dict[str, float],
                              # Add other relevant parameters
                              ) -> Dict[str, Any]:
        """
        Placeholder for submitting a solar cell simulation request.
        Actual endpoint: /simulations/solar/run (example)
        """
        payload = {
            "cell_type": cell_type,
            "layer_thicknesses": layer_thicknesses
            # ... other params
        }
        logger.info(f"Submitting solar cell simulation with payload: {payload}")
        # return self._request("POST", "/simulations/solar/run", json=payload)
        logger.warning("simulate_solar_cell client method is a placeholder. API endpoint needs implementation.")
        return {"job_id": "dummy_solar_job_456", "status": "submitted_placeholder", "message": "Endpoint not implemented"}


    def get_simulation_status(self, job_id: str) -> Dict[str, Any]:
        """
        Placeholder for getting the status of a simulation.
        Actual endpoint: /simulations/{job_id}/status (example)
        """
        logger.info(f"Fetching status for job_id: {job_id}")
        # return self._request("GET", f"/simulations/{job_id}/status")
        logger.warning(f"get_simulation_status client method is a placeholder. API endpoint needs implementation.")
        return {"job_id": job_id, "status": "running_placeholder", "progress": 50}

    def get_simulation_results(self, job_id: str) -> Dict[str, Any]:
        """
        Placeholder for fetching the results of a completed simulation.
        Actual endpoint: /simulations/{job_id}/results (example)
        """
        logger.info(f"Fetching results for job_id: {job_id}")
        # return self._request("GET", f"/simulations/{job_id}/results")
        logger.warning("get_simulation_results client method is a placeholder. API endpoint needs implementation.")
        return {"job_id": job_id, "status": "completed_placeholder", "data": {"message": "Dummy results"}}

    def get_dashboard_url(self, job_id: str, relative: bool = False) -> str:
        """
        Constructs the URL for a specific job's dashboard.
        Actual endpoint could be /dashboard/{job_id} or similar.
        """
        # This method would construct the URL. The actual dashboard rendering is server-side
        # or a separate web application.
        # The client just needs to know the URL pattern.
        dashboard_path = f"/dashboard/{job_id}"
        if relative:
            return dashboard_path
        return f"{self.base_url.rstrip('/')}{dashboard_path}"

    def open_dashboard_in_browser(self, job_id: str):
        """Opens the dashboard for a given job_id in the default web browser."""
        url = self.get_dashboard_url(job_id)
        logger.info(f"Opening dashboard for job {job_id} at {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"Could not open dashboard in browser: {e}")

# Example usage (if run directly, for testing client methods)
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # Assume API server is running at http://localhost:8000
    client = OpenWorldApiClient()

    logger.info("--- Testing Battery Simulation (Placeholder) ---")
    dummy_protocol = [
        {"mode": "charge", "c_rate": 1.0, "cutoff_voltage": 4.2},
        {"mode": "rest", "duration_seconds": 600},
        {"mode": "discharge", "c_rate": 1.0, "cutoff_voltage": 3.0}
    ]
    battery_sim_result = client.simulate_battery(
        chemistry="nmc_placeholder", 
        capacity_ah=3.0, 
        initial_soc=0.5,
        protocol=dummy_protocol
    )
    print(f"Battery Sim API Response: {battery_sim_result}")
    battery_job_id = battery_sim_result.get("job_id")

    if battery_job_id:
        status = client.get_simulation_status(battery_job_id)
        print(f"Status for {battery_job_id}: {status}")
        # client.open_dashboard_in_browser(battery_job_id) # Dashboard endpoint also needs implementation

    logger.info("\n--- Testing Solar Simulation (Placeholder) ---")
    solar_sim_result = client.simulate_solar_cell(
        cell_type="perovskite_placeholder",
        layer_thicknesses={"absorber_nm": 500, "etl_nm": 50, "htl_nm": 50}
    )
    print(f"Solar Sim API Response: {solar_sim_result}")
    solar_job_id = solar_sim_result.get("job_id")
    if solar_job_id:
        status = client.get_simulation_status(solar_job_id)
        print(f"Status for {solar_job_id}: {status}")

    logger.info("\nClient tests finished. These are placeholders until actual API endpoints are active.") 