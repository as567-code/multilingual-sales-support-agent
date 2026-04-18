"""Structured logging (structlog) used by every agent.

JSON renderer in production, colored key-value in dev — picked by LOG_LEVEL's
owner, we just set up a sane default here that plays nicely with pytest -s.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.config import get_settings

_configured = False


def configure_logging() -> None:
    """Idempotent structlog + stdlib setup."""
    global _configured
    if _configured:
        return

    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)
