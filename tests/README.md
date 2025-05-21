# Tests for OpenWorld

This directory contains all tests for the OpenWorld project, organized into subdirectories based on test type.

## Subdirectories:

- `unit/`: Unit tests, focusing on individual components in isolation.
- `integration/`: Integration tests, verifying interactions between components.
- `validation/`: Validation tests, ensuring the scientific accuracy of simulation results against benchmarks or known data.

Refer to the README files within each subdirectory for more specific information.

## Running Tests

Tests are typically run using `pytest` from the project root directory:

```bash
pytest
```

Or to run tests for a specific directory or file:

```bash
pytest tests/unit/
pytest tests/integration/test_some_integration.py
```

Make sure you have the development dependencies installed (see `pyproject.toml` under `[project.optional-dependencies].dev`). 