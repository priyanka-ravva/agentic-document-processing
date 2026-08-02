"""Logging configuration."""

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging with Rich formatting."""

    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    return logging.getLogger(name)
