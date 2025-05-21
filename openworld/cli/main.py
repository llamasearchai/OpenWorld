"""
Command Line Interface for the OpenWorld platform.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..api.python_api import OpenWorldAPI
from ..api.schemas import (
    PhysicsSimRequest, BatterySimRequest, SolarSimRequest, 
    AIRequest, SimulationType
)
from ..config import OpenWorldConfig
from ..utils.logging import configure_logging
from ..agent.world_model_agent import WorldModelStrategy

# Create typer app
app = typer.Typer(
    name="openworld",
    help="OpenWorld: Unified Physics Simulation and AI Reasoning Platform",
    add_completion=False
)

# Create app groups
physics_app = typer.Typer(help="Physics simulation commands")
battery_app = typer.Typer(help="Battery simulation commands")
solar_app = typer.Typer(help="Solar cell simulation commands")
ai_app = typer.Typer(help="AI reasoning commands")
config_app = typer.Typer(help="Configuration commands")

# Add subcommands to main app
app.add_typer(physics_app, name="physics")
app.add_typer(battery_app, name="battery")
app.add_typer(solar_app, name="solar")
app.add_typer(ai_app, name="ai")
app.add_typer(config_app, name="config")

# Create rich console for pretty output
console = Console()

# Global API instance
api = None

def get_api():
    """Get or create the OpenWorld API instance."""
    global api
    if api is None:
        api = OpenWorldAPI()
    return api

@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration file"
    )
):
    """
    OpenWorld: Unified Physics Simulation and AI Reasoning Platform.
    """
    # Configure logging
    configure_logging(level=log_level)
    
    # Initialize API with config if provided
    global api
    if config_path:
        api = OpenWorldAPI(OpenWorldConfig.from_file(config_path))
    else:
        api = OpenWorldAPI()

@physics_app.command("run")
def run_physics_simulation(
    duration: float = typer.Option(
        10.0, "--duration", "-d",
        help="Simulation duration in seconds"
    ),
    time_step: float = typer.Option(
        0.01, "--time-step", "-t",
        help="Time step in seconds"
    ),
    gravity: float = typer.Option(
        9.8, "--gravity", "-g",
        help="Gravity value (m/s²)"
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="JSON input file with simulation parameters"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="File to write results to (JSON format)"
    )
):
    """
    Run a physics simulation.
    """
    api = get_api()
    
    # Load request from file or create new
    if input_file:
        with open(input_file, 'r') as f:
            request_data = json.load(f)
        request = PhysicsSimRequest(**request_data)
    else:
        # Create a simple example with a falling particle
        request = PhysicsSimRequest(
            duration=duration,
            time_step=time_step,
            gravity=gravity,
            dimensions=3,
            objects=[
                {
                    "type": "particle",
                    "position": [0, 10, 0],
                    "velocity": [0, 0, 0],
                    "mass": 1.0,
                    "radius": 0.5
                }
            ]
        )
    
    # Run simulation
    console.print(Panel("Running physics simulation...", title="OpenWorld Physics"))
    result = api.run_physics_simulation(request)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        console.print(f"Results saved to {output_file}")
    else:
        # Print summary to console
        table = Table(title="Physics Simulation Results")
        table.add_column("Property")
        table.add_column("Value")
        
        table.add_row("Duration", f"{result.duration} seconds")
        table.add_row("Time steps", str(result.time_steps))
        table.add_row("Objects", str(len(result.objects)))
        
        console.print(table)
        
        # Print trajectory for first object
        if result.trajectories and len(result.trajectories) > 0:
            obj_id = list(result.trajectories.keys())[0]
            traj = result.trajectories[obj_id]
            
            console.print(f"Trajectory for object {obj_id}:")
            pos_table = Table(title=f"Position data (showing first and last 3 points)")
            pos_table.add_column("Time")
            pos_table.add_column("Position")
            
            # Show first 3 points
            for i in range(min(3, len(traj["time"]))):
                pos_table.add_row(
                    f"{traj['time'][i]:.2f}",
                    f"{traj['position'][i]}"
                )
            
            if len(traj["time"]) > 6:
                pos_table.add_row("...", "...")
            
            # Show last 3 points
            for i in range(max(0, len(traj["time"]) - 3), len(traj["time"])):
                pos_table.add_row(
                    f"{traj['time'][i]:.2f}",
                    f"{traj['position'][i]}"
                )
            
            console.print(pos_table)

@battery_app.command("run")
def run_battery_simulation(
    duration: float = typer.Option(
        3600.0, "--duration", "-d",
        help="Simulation duration in seconds"
    ),
    model_type: str = typer.Option(
        "DFN", "--model", "-m",
        help="Battery model type (DFN, SPM, etc.)"
    ),
    temperature: float = typer.Option(
        25.0, "--temperature", "-t",
        help="Operating temperature in Celsius"
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="JSON input file with simulation parameters"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="File to write results to (JSON format)"
    )
):
    """
    Run a battery simulation.
    """
    api = get_api()
    
    # Load request from file or create new
    if input_file:
        with open(input_file, 'r') as f:
            request_data = json.load(f)
        request = BatterySimRequest(**request_data)
    else:
        # Create a simple example
        request = BatterySimRequest(
            duration=duration,
            model_type=model_type,
            temperature=temperature,
            parameters={
                "capacity": 3.0,  # Ah
                "nominal_voltage": 3.7  # V
            },
            current_profile=[
                {"time": 0, "current": -1.5},  # 0.5C discharge
                {"time": duration, "current": 0.0}
            ]
        )
    
    # Run simulation
    console.print(Panel("Running battery simulation...", title="OpenWorld Battery"))
    result = api.run_battery_simulation(request)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        console.print(f"Results saved to {output_file}")
    else:
        # Print summary to console
        table = Table(title="Battery Simulation Results")
        table.add_column("Property")
        table.add_column("Value")
        
        table.add_row("Duration", f"{result.duration} seconds")
        table.add_row("Model", result.metadata.get("model_type", "Unknown"))
        table.add_row("Initial Voltage", f"{result.voltage[0]:.2f} V")
        table.add_row("Final Voltage", f"{result.voltage[-1]:.2f} V")
        table.add_row("Initial SoC", f"{result.soc[0]:.1%}")
        table.add_row("Final SoC", f"{result.soc[-1]:.1%}")
        
        console.print(table)

@solar_app.command("run")
def run_solar_simulation(
    model_type: str = typer.Option(
        "DriftDiffusion", "--model", "-m",
        help="Solar cell model type"
    ),
    spectrum: str = typer.Option(
        "AM1.5G", "--spectrum", "-s",
        help="Solar spectrum (AM1.5G, etc.)"
    ),
    illumination: float = typer.Option(
        1.0, "--illumination", "-i",
        help="Light intensity in suns"
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input",
        help="JSON input file with simulation parameters"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="File to write results to (JSON format)"
    )
):
    """
    Run a solar cell simulation.
    """
    api = get_api()
    
    # Load request from file or create new
    if input_file:
        with open(input_file, 'r') as f:
            request_data = json.load(f)
        request = SolarSimRequest(**request_data)
    else:
        # Create a simple example
        request = SolarSimRequest(
            model_type=model_type,
            spectrum=spectrum,
            illumination=illumination,
            parameters={
                "material": "Silicon",
                "thickness": 200.0,  # µm
                "bandgap": 1.1  # eV
            },
            voltage_sweep={
                "start": 0.0,
                "end": 0.7,
                "points": 100
            }
        )
    
    # Run simulation
    console.print(Panel("Running solar cell simulation...", title="OpenWorld Solar"))
    result = api.run_solar_simulation(request)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        console.print(f"Results saved to {output_file}")
    else:
        # Print summary to console
        table = Table(title="Solar Cell Simulation Results")
        table.add_column("Property")
        table.add_column("Value")
        
        table.add_row("Voc", f"{result.voc:.4f} V")
        table.add_row("Jsc", f"{result.jsc:.2f} mA/cm²")
        table.add_row("Fill Factor", f"{result.ff:.4f}")
        table.add_row("Efficiency", f"{result.efficiency:.2f}%")
        table.add_row("Model", result.metadata.get("model_type", "Unknown"))
        table.add_row("Spectrum", result.metadata.get("spectrum", "Unknown"))
        
        console.print(table)

@ai_app.command("reason")
def run_ai_reasoning(
    query: str = typer.Argument(
        ..., help="Natural language query or instruction"
    ),
    strategy: str = typer.Option(
        "causal", "--strategy",
        help="Reasoning strategy to use"
    ),
    context_file: Optional[Path] = typer.Option(
        None, "--context", "-c",
        help="JSON file with context data"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="File to write results to (JSON format)"
    )
):
    """
    Run AI-assisted analysis and reasoning.
    """
    api = get_api()
    
    # Load context from file if provided
    context = None
    if context_file:
        with open(context_file, 'r') as f:
            context = json.load(f)
    
    # Create request
    request = AIRequest(
        query=query,
        context=context,
        reasoning_strategy=strategy
    )
    
    # Run AI analysis
    console.print(Panel(f"Running {strategy} reasoning on query: {query}", title="OpenWorld AI"))
    result = api.run_ai_analysis(request)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        console.print(f"Results saved to {output_file}")
    else:
        # Print to console
        console.print(Panel(result.response, title="AI Response"))
        
        console.print("Reasoning steps:")
        for i, step in enumerate(result.reasoning):
            console.print(f"{i+1}. {step}")

@config_app.command("init")
def init_config(
    output_file: Path = typer.Option(
        Path.home() / ".openworld" / "config.json",
        "--output", "-o",
        help="Path to save the configuration"
    )
):
    """
    Initialize a new configuration file.
    """
    # Create default config
    config = OpenWorldConfig()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save config
    config.save(output_file)
    console.print(f"Created default configuration at {output_file}")

@config_app.command("show")
def show_config(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration file (default: ~/.openworld/config.json)"
    )
):
    """
    Show the current configuration.
    """
    # Use default config path if not specified
    if config_path is None:
        config_path = Path.home() / ".openworld" / "config.json"
    
    # Load and show config
    try:
        config = OpenWorldConfig.from_file(config_path)
        config_dict = config.to_dict()
        
        # Convert to JSON for pretty printing
        json_str = json.dumps(config_dict, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        
        console.print(Panel(syntax, title=f"Configuration: {config_path}"))
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/] {e}")
        return 1

def app_cli_entry():
    """Entry point for the CLI."""
    app() 