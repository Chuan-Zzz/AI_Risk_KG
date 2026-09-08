"""Shared runtime helpers for command-line entry points and storage."""

from __future__ import annotations

import re
from pathlib import Path

from src.core.config import Config, get_config

_SAFE_EVENT_ID = re.compile(r"^[\w-]+$")


def event_id_for(event: dict, default: str = "unknown") -> str:
    """Return an event identifier accepted by filesystem-backed output."""
    event_id = str(event.get("event_id") or event.get("incident_id") or default)
    return sanitize_event_id(event_id)


def sanitize_event_id(event_id: str) -> str:
    """Reject IDs that could escape the per-event output directory."""
    if not _SAFE_EVENT_ID.fullmatch(event_id):
        raise ValueError(f"Invalid event_id: {event_id!r}")
    return event_id


def output_root(config: Config | None = None) -> Path:
    """Resolve the configured output directory relative to the project root."""
    cfg = config or get_config()
    value = cfg.get("output.dir", cfg.get("paths.output_dir", "output"))
    path = Path(str(value))
    return path if path.is_absolute() else cfg.project_root / path
