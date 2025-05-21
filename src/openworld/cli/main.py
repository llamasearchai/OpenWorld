import typer
import uvicorn
import json
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
import httpx # For making API calls from CLI
from typing import Optional, List

# Attempt to import settings and logger from the new structure
# This assumes that the openworld package is installed or src is in PYTHONPATH
try:
    from openworld.utils.config import settings # Assuming settings will be loaded here
    from openworld.utils.logging import get_logger # Assuming logger setup
except ImportError:
    # Fallback for initial setup or if running script directly out of context
    # This part might need adjustment based on how settings are globally managed
    print("Warning: Could not import OpenWorld settings/logger. Using fallback or defaults.", file=sys.stderr)
    # Define a simple fallback settings object for API_HOST/PORT if needed for CLI to function minimally
    class FallbackSettings:
        API_HOST = os.getenv("API_HOST", "127.0.0.1")
        API_PORT = int(os.getenv("API_PORT", "8000"))
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        # Add other critical settings if CLI depends on them directly
    settings = FallbackSettings()
    
    # Basic logger if a shared one isn't available
    import logging
    logging.basicConfig(level=settings.LOG_LEVEL)
    _cli_logger = logging.getLogger("OpenWorld_CLI_Fallback")
    def get_logger(name, level=None): return _cli_logger


# --- CLI App Initialization ---
app_cli = typer.Typer(
    name="openworld",
    help="OpenWorld v2.0 CLI: Interact with the simulation engine and AI.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
console = Console()
logger = get_logger("OpenWorld_CLI", level=settings.LOG_LEVEL)

# --- API Client Configuration ---
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

# --- Helper Functions ---
def handle_api_response(response: httpx.Response):
    """Checks API response and prints errors or results."""
    try:
        response.raise_for_status() 
        results = response.json()
        console.print("[bold green]API Request Successful![/bold green]")
        console.print(Syntax(json.dumps(results, indent=2), "json", theme="default", line_numbers=False))
        return results
    except httpx.RequestError as e:
        console.print(f"[bold red]API Connection Error:[/bold red] Failed to connect to {e.request.url}. Is the server running?")
        raise typer.Exit(code=1)
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
             error_json = e.response.json()
             error_detail = error_json.get("detail", error_detail)
        except json.JSONDecodeError:
             pass
        console.print(f"[bold red]API Error {e.response.status_code}:[/bold red] {error_detail}")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print("[bold red]API Error:[/bold red] Received non-JSON response from server.")
        console.print(response.text)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected client-side error occurred: {e}[/bold red]")
        logger.error("Unexpected CLI error", exc_info=True)
        raise typer.Exit(code=1)

def load_json_params(params_input: str) -> dict:
    """Loads JSON from string or file path."""
    try:
        if os.path.exists(params_input):
            with open(params_input, 'r') as f:
                params = json.load(f)
            console.print(f"Loaded parameters from file: [cyan]{params_input}[/cyan]")
        else:
            params = json.loads(params_input)
        return params
    except json.JSONDecodeError:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON string provided for parameters.")
        raise typer.Exit(code=1)
    except OSError as e:
        console.print(f"[bold red]Error:[/bold red] Could not read parameter file '{params_input}': {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error loading parameters: {e}[/bold red]")
        raise typer.Exit(code=1)

# --- CLI Commands ---

@app_cli.command(name="server", help="Starts the OpenWorld API server (requires uvicorn).")
def run_server_cli(
    host: str = typer.Option(settings.API_HOST, help="Host to bind the server to."),
    port: int = typer.Option(settings.API_PORT, help="Port to bind the server to."),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes."),
):
    """Starts the FastAPI API server using Uvicorn."""
    console.print(f"[bold cyan]Attempting to start OpenWorld API server on {host}:{port}...[/bold cyan]")
    try:
        # The path to app:app must be correct for uvicorn to find it.
        # Assuming openworld.api.server:app will be the location after restructuring.
        uvicorn.run(
            "openworld.api.server:app", 
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=settings.LOG_LEVEL.lower(),
        )
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Could not import server application. Ensure 'openworld.api.server:app' is correct and all dependencies are installed: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
         console.print(f"[bold red]Failed to start server: {e}[/bold red]")
         logger.error("Server startup failed", exc_info=True)
         raise typer.Exit(code=1)

# --- Simulation Subcommand Group (adapted from old CLI and scaffold) ---
sim_app = typer.Typer(name="simulate", help="Run and manage simulations.", no_args_is_help=True)
app_cli.add_typer(sim_app, name="simulate")

# TODO: Adapt the physics, battery, solar simulation commands from the scaffold to this structure.
# The old `simulate` command took a scenario. The new structure implies more specific subcommands.
# For example, `openworld simulate physics --params-file <file.json>`

@sim_app.command("physics", help="Run a physics simulation.")
def simulate_physics_cli(
    params_input: str = typer.Option(..., "--params", "-p", help="JSON string or path to JSON file with simulation parameters."),
    output_file: Optional[typer.Path(writable=True)] = typer.Option(None, "-o", "--output", help="Save full results JSON to file."),
):
    """Runs a physics simulation by calling the API."""
    params = load_json_params(params_input)
    url = f"{API_BASE_URL}/simulations/physics/run"
    console.print(f"[cyan]Requesting physics simulation with parameters from: {params_input}[/cyan]")
    try:
        with httpx.Client(timeout=None) as client: # Set timeout to None for potentially long simulations
            response = client.post(url, json=params)
        results = handle_api_response(response)
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"Full simulation results saved to: [cyan]{output_file}[/cyan]")
    except Exception as e:
        logger.error(f"CLI simulate physics error: {e}", exc_info=True)
        if not isinstance(e, typer.Exit):
            console.print(f"[bold red]An unexpected error occurred during physics simulation: {e}[/bold red]")
            raise typer.Exit(code=1)


@sim_app.command("battery", help="Run a battery simulation.")
def simulate_battery_cli(
    params_input: str = typer.Option(..., "--params", "-p", help="JSON string or path to JSON file with simulation parameters."),
    output_file: Optional[typer.Path(writable=True)] = typer.Option(None, "-o", "--output", help="Save full results JSON to file."),
):
    """Runs a battery simulation by calling the API."""
    params = load_json_params(params_input)
    url = f"{API_BASE_URL}/simulations/battery/run"
    console.print(f"[cyan]Requesting battery simulation with parameters from: {params_input}[/cyan]")
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(url, json=params)
        results = handle_api_response(response)
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"Full simulation results saved to: [cyan]{output_file}[/cyan]")
    except Exception as e:
        logger.error(f"CLI simulate battery error: {e}", exc_info=True)
        if not isinstance(e, typer.Exit):
            console.print(f"[bold red]An unexpected error occurred during battery simulation: {e}[/bold red]")
            raise typer.Exit(code=1)


@sim_app.command("solar", help="Run a solar cell simulation.")
def simulate_solar_cli(
    params_input: str = typer.Option(..., "--params", "-p", help="JSON string or path to JSON file with simulation parameters."),
    output_file: Optional[typer.Path(writable=True)] = typer.Option(None, "-o", "--output", help="Save full results JSON to file."),
):
    """Runs a solar cell simulation by calling the API."""
    params = load_json_params(params_input)
    url = f"{API_BASE_URL}/simulations/solar/run"
    console.print(f"[cyan]Requesting solar cell simulation with parameters from: {params_input}[/cyan]")
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(url, json=params)
        results = handle_api_response(response)
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"Full simulation results saved to: [cyan]{output_file}[/cyan]")
    except Exception as e:
        logger.error(f"CLI simulate solar error: {e}", exc_info=True)
        if not isinstance(e, typer.Exit):
            console.print(f"[bold red]An unexpected error occurred during solar simulation: {e}[/bold red]")
            raise typer.Exit(code=1)


# @sim_app.command("scenario", help="Run a simulation based on a scenario JSON (legacy).")
# def simulate_scenario_cli(
#     scenario_json: str = typer.Argument(..., help="JSON string or path to JSON file defining the simulation scenario."),
#     output_file: Optional[typer.Path(writable=True)] = typer.Option(None, "-o", "--output", help="Save full results JSON to file."),
# ):
#     """Runs a generic simulation scenario via a general-purpose API endpoint (if available)."""
#     params = load_json_params(scenario_json)
#     # This command is a bit of a holdover from the old CLI.
#     # The new structure suggests more specific `simulate physics/battery/solar` commands.
#     # We need a generic "/simulations/run_scenario" endpoint or adapt this.
#     # For now, let's assume a placeholder endpoint.
#     
#     # Placeholder: this would ideally map to a specific endpoint like `/simulations/physics/run` or a generic one
#     # For example, if it's a physics simulation:
#     # url = f"{API_BASE_URL}/simulations/physics/run" # Or a more generic scenario runner
#     # For now, this is a conceptual placeholder.
#     
#     # This is a placeholder since the original `worldmodel_agent_backend/cli.py` directly called HyperSimulationTool.
#     # In the new structure, CLI should generally call API endpoints.
#     # This part needs to be reconciled with the actual API capabilities.
#     console.print(f"[bold red]Error:[/bold red] Direct scenario execution via CLI to be refactored to use specific API endpoints (e.g., physics, battery, solar).")
#     console.print("Original command was: HyperSimulationTool().run_simulation(params)")
#     # results = {"status": "error", "message": "Legacy scenario command needs API endpoint."}
#     # if output_file:
#     #     with open(output_file, 'w') as f: json.dump(results, f, indent=2)
#     raise typer.Exit(1)


# --- AI/Reasoning Subcommand Group (adapted from old CLI and scaffold) ---
ai_app = typer.Typer(name="reason", help="Interact with the AI reasoning system.", no_args_is_help=True)
app_cli.add_typer(ai_app, name="reason") # Renamed from `ai` to match old CLI `reason`

@ai_app.command("query", help="Ask the OpenWorld AI to reason about a query based on a strategy.")
def ai_reason_query_cli(
    query: str = typer.Argument(..., help="The reasoning query."),
    strategy: str = typer.Option("causal", "--strategy", "-s", help="Reasoning strategy (e.g., 'causal', 'counterfactual', 'temporal', 'spatial')."),
    # use_dspy: bool = typer.Option(False, "--dspy", help="Use DSPy backend (if available)."), # From scaffold, can add later
    output_file: Optional[typer.Path(writable=True)] = typer.Option(None, "-o", "--output", help="Save full AI response JSON to file.")
):
    """Performs reasoning using the AI via the API."""
    console.print(f"[cyan]Requesting AI reasoning for query '{query[:50]}...' with strategy '{strategy}'...[/cyan]")
    
    # The AIRequest schema in scaffold might have `domain` not `strategy`. Reconciled to use `domain` in payload.
    request_body = {"problem": query, "domain": strategy} 
    url = f"{API_BASE_URL}/ai/reason" # Target a specific /ai/reason endpoint

    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=600.0) as client:
            response = client.post(url, json=request_body)
            results = handle_api_response(response)
            if output_file:
                 with open(output_file, 'w') as f: json.dump(results, f, indent=2)
                 console.print(f"Full AI response saved to: [cyan]{output_file}[/cyan]")
    except typer.Exit: # Re-raise if handle_api_response already exited
        raise
    except Exception as e:
        logger.error(f"CLI AI reason error: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred during AI reasoning: {e}[/bold red]")
        raise typer.Exit(code=1)


# --- Management Subcommand Group (from scaffold) ---
manage_app = typer.Typer(name="manage", help="Manage simulation instances.", no_args_is_help=True)
app_cli.add_typer(manage_app, name="manage")

@manage_app.command("list", help="List active simulation instances.")
def manage_list_cli(
     sim_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type ('physics', 'battery', 'solar').")
 ):
     url = f"{API_BASE_URL}/simulations"
     params = {"sim_type": sim_type} if sim_type else {}
     try:
         with httpx.Client() as client:
             response = client.get(url, params=params)
         results = handle_api_response(response)
         table = Table(title="Active Simulations")
         table.add_column("Type", style="cyan")
         table.add_column("Simulation ID", style="magenta")
         if isinstance(results, dict): # Expecting dict like {"physics": [...], "battery": [...]}
             for sim_type_key, id_list in results.items():
                 for sim_id_val in id_list:
                     table.add_row(sim_type_key, sim_id_val)
         else:
             console.print("[yellow]Unexpected result format for simulation list.[/yellow]")
         console.print(table)
     except Exception as e:
         logger.error("CLI manage list error", exc_info=True)


@manage_app.command("status", help="Get the status of a specific simulation.")
def manage_status_cli(sim_id: str = typer.Argument(..., help="The ID of the simulation.")):
     url = f"{API_BASE_URL}/simulations/{sim_id}/status"
     try:
         with httpx.Client() as client:
             response = client.get(url)
         handle_api_response(response)
     except Exception as e:
         logger.error("CLI manage status error", exc_info=True)


@manage_app.command("delete", help="Delete a simulation instance.")
def manage_delete_cli(sim_id: str = typer.Argument(..., help="The ID of the simulation to delete.")):
     if not typer.confirm(f"Are you sure you want to delete simulation '{sim_id}'?"):
         raise typer.Abort()
     url = f"{API_BASE_URL}/simulations/{sim_id}"
     try:
         with httpx.Client() as client:
             response = client.delete(url)
             response.raise_for_status() 
         console.print(f"[bold green]Simulation '{sim_id}' deleted successfully.[/bold green]")
     except httpx.HTTPStatusError as e:
         if e.response.status_code == 404:
             console.print(f"[bold yellow]Simulation '{sim_id}' not found.[/bold yellow]")
         else:
             handle_api_response(e.response)
     except Exception as e:
         logger.error("CLI manage delete error", exc_info=True)
         console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
         raise typer.Exit(1)

# --- Prometheus server (from old CLI) ---
# This was in the old CLI's main(). It might be better started by the API server
# or as a separate process if needed, not directly in the Typer app's flow unless
# it's a specific command.
# For now, I'll make it a command.
@app_cli.command(name="start-metrics-server", help="Starts a Prometheus metrics server (experimental).")
def start_metrics_cli(
    port: int = typer.Option(8001, help="Port for the Prometheus metrics server.") # Changed port from 8000 to avoid conflict
):
    try:
        from prometheus_client import start_http_server
        start_http_server(port)
        console.print(f"[green]Prometheus metrics server started on port {port}.[/green]")
        console.print("This server will run indefinitely. Press Ctrl+C to stop.")
        # Keep alive or exit? For a CLI command, it should probably exit or run in background.
        # For now, it will block. User can background it.
        import time
        while True: time.sleep(1) # Keep alive until Ctrl+C
    except ImportError:
        console.print("[bold red]Error:[/bold red] `prometheus-client` is not installed. Please install it.")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Failed to start metrics server: {e}[/bold red]")
        raise typer.Exit(1)


# --- Entry point for CLI ---
def app_cli_entry():
     # Global setup for CLI before commands run (if any)
     # The old CLI had:
     # agent = MetaWorldModelAgent(config={
     # "strategies": ["causal", "counterfactual", "temporal", "spatial"],
     # "learning_rate": 0.01
     # })
     # This agent instantiation is no longer needed here if commands call the API.
     app_cli()

if __name__ == "__main__":
     app_cli_entry() 