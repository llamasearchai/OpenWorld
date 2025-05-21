"""
Logging utility functions for PhysicsGPT.
"""

import logging
import sys
from pathlib import Path

def get_logger(name, level=logging.INFO, log_file=None):
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name, typically __name__
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if no handlers exist
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def configure_global_logging(level=logging.INFO, log_dir=None):
    """
    Configure global logging settings.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Optional directory to store log files
        
    Returns:
        Root logger
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_dir specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(str(log_path / "openworld.log")) # Changed log file name
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger 