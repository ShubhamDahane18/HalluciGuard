"""Structured logging configuration for HalluciGuard."""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
