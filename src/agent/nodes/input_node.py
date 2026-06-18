"""Stage 1: News Reports Input - Load and organize event documents."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import PipelineState

logger = logging.getLogger(__name__)


def input_node(state: PipelineState) -> dict[str, Any]:
    documents = state.get("documents", [])
    event_id = state.get("event_id", "unknown")

    report_count = len(documents)

    publish_times = [d.publish_time for d in documents if d.publish_time]
    first_seen = min(publish_times) if publish_times else None
    last_seen = max(publish_times) if publish_times else None

    source_set = sorted({d.source_name for d in documents if d.source_name})
    language_set = sorted({d.language for d in documents if d.language})

    logger.info(
        f"[Stage 1] Event {event_id}: {report_count} reports, "
        f"sources={len(source_set)}, langs={language_set}"
    )

    return {
        "report_count": report_count,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "source_set": source_set,
        "language_set": language_set,
        "current_stage": "input",
        "stages_completed": state.get("stages_completed", []) + ["input"],
        "status": "PROCESSING",
    }
