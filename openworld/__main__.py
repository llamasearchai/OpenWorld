"""
Main entry point for the OpenWorld platform.

This module allows running OpenWorld as a module using 'python -m openworld'.
"""

import sys
from .cli.main import app_cli_entry

def main():
    """Main function that starts the OpenWorld CLI."""
    app_cli_entry()

if __name__ == "__main__":
    sys.exit(main()) 