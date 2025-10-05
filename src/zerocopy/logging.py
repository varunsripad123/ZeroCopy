"""Logging utilities with graceful fallback when structlog is unavailable."""
from __future__ import annotations

import logging
from typing import Any

try:  # pragma: no cover - optional dependency
    from structlog import get_logger as _get_logger
except Exception:  # pragma: no cover - fallback
    def _get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


def get_logger(name: str) -> Any:
    return _get_logger(name)


__all__ = ["get_logger"]
