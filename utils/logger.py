"""Logging setup utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: str, level: str = "INFO") -> logging.Logger:
    """Configure the root logger and return a named logger for the caller.

    Handlers are attached to the root logger so that every module-level
    logger obtained via `logging.getLogger(__name__)` elsewhere in the
    codebase (e.g. in `clients/client.py`, `server/server.py`) propagates to
    them automatically, without each module needing to know a shared name.

    Safe to call multiple times: existing root handlers are reused instead
    of being duplicated.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    if not root_logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(Path(log_dir) / f"{name.replace('.', '_')}.log")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger(name)
