"""Central logging helpers."""

import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", to_file: Optional[str] = None) -> None:
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers = [stream_handler]

    if to_file:
        file_handler = logging.FileHandler(to_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        configure_logging(level)
    logger.setLevel(level.upper())
    logger.propagate = True
    return logger

