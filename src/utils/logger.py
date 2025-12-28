"""
Centralized logger access for all MERIDIAN modules.
"""

import logging
from config.logging_config import setup_logger


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (use __name__ from calling module)
    log_level : str
        Logging level
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    return setup_logger(name, log_level)