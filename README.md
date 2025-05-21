<div align="center">
  <img src="openworld.svg" alt="OpenWorld Logo" width="200"/>
</div>

# OpenWorld Simulation Platform

OpenWorld is a comprehensive, multi-physics simulation platform designed for advanced research and development. It provides a modular and extensible framework for simulating complex systems involving physics, solar energy, battery technology, and AI-driven agents.

## Features

*   **Modular Core Components:**
    *   **Physics Engine:** Robust 3D rigid-body dynamics simulation with various collision detection (GJK, Sweep and Prune, BVH) and contact resolution mechanisms.
    *   **Solar Simulation:** Detailed solar cell device modeling and IV curve simulation under various environmental conditions.
    *   **Battery Simulation:** Electrochemical battery modeling (SPM, DFN) with support for custom charge/discharge protocols.
    *   **Agent Framework:** Tools for creating intelligent agents capable of reasoning and interacting within simulated worlds.
*   **Flexible API:** A FastAPI-based server provides endpoints for creating, managing, and running simulations, as well as for AI reasoning tasks.
*   **Command-Line Interface (CLI):** A Typer-based CLI for easy interaction with the simulation platform.
*   **Extensible Architecture:** Designed for easy integration of new models, algorithms, and simulation domains.
*   **Unit System:** Integrated unit handling using the `pint` library for dimensional consistency.

## Project Structure

```
OpenWorld/
├── openworld/                # Core library code
│   ├── agent/                # Agent framework (core, memory, reasoning, tools)
│   ├── api/                  # FastAPI server implementation (endpoints, schemas)
│   ├── cli/                  # Typer-based command-line interface
│   ├── core/                 # Core simulation engines
│   │   ├── base/
│   │   ├── battery/          # Battery simulation engine
│   │   ├── physics/          # Physics engine (world, objects, collision, contact)
│   │   └── solar/            # Solar simulation engine (device, material)
│   ├── solvers/              # Numerical solvers
│   ├── utils/                # Utility functions (logging, units, exceptions)
│   └── visualization/        # Visualization tools (e.g., dashboard integration)
├── src/                      # Alternative source layout (if preferred, sync with openworld/)
├── tests/                    # Unit, integration, and validation tests
│   ├── integration/
│   ├── unit/
│   └── validation/
├── examples/                 # Example scripts and notebooks
│   ├── notebooks/
│   └── scripts/
├── docs/                     # Project documentation
├── .gitignore                # Git ignore file
├── Dockerfile                # Docker configuration
├── LICENSE                   # Project license
├── openworld.svg             # Project Logo
├── pyproject.toml            # Project metadata and build configuration
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/llamasearchai/OpenWorld.git
    cd OpenWorld
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode (for development):**
    ```bash
    pip install -e .
    ```

### Running the API Server

```bash
python -m openworld.api.server  # Or your configured entry point
```
The API will typically be available at `http://127.0.0.1:8000`.

### Using the CLI

```bash
openworld-cli --help
openworld-cli physics-create --help
# ... and other commands
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Nik Jois (nikjois@llamasearch.ai) 