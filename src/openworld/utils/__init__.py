"""Utility functions for the OpenWorld project."""

# Example utility import (if you add utility modules)
# from .file_helpers import load_json, save_json
from .logging_config import get_logger, configure_global_logging
from .units import ureg, Q_, BatteryUnits, SolarUnits, PhysicsUnits, get_value, convert_to

# It's common to set up a logger here or provide a logging configuration function.
# For example, a get_logger function that standardizes logger creation:
import logging

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_configured_logger(name: str, level: int = logging.INFO, log_format: str = DEFAULT_LOG_FORMAT) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers if they already exist (e.g. if called multiple times or in a complex setup)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

__all__ = [
    "get_configured_logger",
    "get_logger",
    "configure_global_logging",
    "ureg",
    "Q_",
    "BatteryUnits",
    "SolarUnits",
    "PhysicsUnits",
    "get_value",
    "convert_to",
] 