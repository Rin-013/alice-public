"""
Logging Utilities
=================

Standardized logging setup for gaming submodules.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger for a gaming submodule.

    Args:
        name: Module name (e.g. 'gaming.cognitive.director')
        level: Optional override. Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    return logger
