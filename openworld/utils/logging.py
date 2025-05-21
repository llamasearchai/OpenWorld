"""
Logging utilities for OpenWorld.

This module provides a consistent logging setup across the platform.
"""

import logging
import os
import sys
from typing import Optional, Union, Dict

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Map string level names to logging constants
LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Get the default log level from environment or use INFO
DEFAULT_LEVEL = os.environ.get("OPENWORLD_LOG_LEVEL", "INFO")
DEFAULT_LEVEL = LEVEL_MAP.get(DEFAULT_LEVEL, logging.INFO)

def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name, typically module name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              or corresponding logging constants
              
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set level
    if level is None:
        level = DEFAULT_LEVEL
    elif isinstance(level, str):
        level = LEVEL_MAP.get(level.upper(), DEFAULT_LEVEL)
    
    logger.setLevel(level)
    
    # Only add handler if it doesn't have any
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def configure_logging(level: Optional[Union[str, int]] = None, 
                     format: str = DEFAULT_FORMAT,
                     log_file: Optional[str] = None) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Log level for the root logger
        format: Log format string
        log_file: Optional file to log to
    """
    # Set level
    if level is None:
        level = DEFAULT_LEVEL
    elif isinstance(level, str):
        level = LEVEL_MAP.get(level.upper(), DEFAULT_LEVEL)
        
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    # Log the configuration
    root_logger.info(f"Logging configured with level={logging.getLevelName(level)}"
                    f"{f', log_file={log_file}' if log_file else ''}") 