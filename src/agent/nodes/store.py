"""Store node - Output results to JSON, TTL, and Neo4j."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.config import get_config
from src.core.runtime import output_root, sanitize_event_id

logger = logging.getLogger(__name__)

def store_node(state: PipelineState) -> dict[str, Any]:
    event_id = sanitize_event_id(state.get("event_id", "unknown"))
    subgraph = state.get("event_subgraph")

    config = get_config()

    if subgraph is None:
        logger.warning(f"[Store] No subgraph to store for event {event_id}")
        return {"status": "FAILED", "error_message": "No subgraph generated", "current_stage": "store"}

    validation_passed = state.get("validation_passed", False)
    strict_mode = config.get("validation.strict_mode", True)
    if strict_mode and not validation_passed:
        logger.error("[Store] Event %s failed validation; strict mode skipped output", event_id)
        return {
            "status": "FAILED_VALIDATION",
            "error_message": "; ".join(state.get("validation_errors", [])[:5]),
            "current_stage": "store",
            "stages_completed": state.get("stages_completed", []) + ["store"],
        }

    output_dir = output_root(config) / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "event_subgraph.json"
    json_data = subgraph.model_dump(mode="json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Store] JSON saved: {json_path}")

    # TTL output
    try:
        from src.storage.ttl_store import write_ttl
        ttl_path = output_dir / "event_subgraph.ttl"
        write_ttl(subgraph, ttl_path)
        logger.info(f"[Store] TTL saved: {ttl_path}")
    except Exception as e:
        logger.warning(f"[Store] TTL output failed: {e}")

    # Neo4j is written in Phase 2 (kg_fusion.write_to_neo4j), not per-event

    node_count = len(subgraph.nodes)
    edge_count = len(subgraph.edges)
    ks_count = len(subgraph.knowledge_statements)

    logger.info(f"[Store] Event {event_id}: {node_count} nodes, {edge_count} edges, {ks_count} statements")

    return {
        "status": "COMPLETED" if validation_passed else "COMPLETED_WITH_VALIDATION_ERRORS",
        "current_stage": "store",
        "stages_completed": state.get("stages_completed", []) + ["store"],
    }
