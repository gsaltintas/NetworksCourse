"""Configure using loggers."""

import logging
import sys

import logging_json
import contextual_logger


def setup_logging(log_level: str, name: str = "dps"):
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fields = {
        "level": "levelname",
        "timestamp": "asctime",
        "module": "module",
        "function": "funcName",
        "logger": "name"
    }
    formatter = logging_json.JSONFormatter(fields=fields)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
