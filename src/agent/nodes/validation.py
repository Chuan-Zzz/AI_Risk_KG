"""Consistency validation for the event knowledge subgraph."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.models import ExtractionMode, RelationType
from src.core.ontology import get_ontology

logger = logging.getLogger(__name__)


def validation_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    subgraph = state.get("event_subgraph")
    errors: list[str] = []

    if subgraph is None:
        return {
            "validation_errors": ["No subgraph to validate"],
            "validation_passed": False,
            "current_stage": "validation",
            "retry_count": state.get("retry_count", 0) + 1,
            "stages_completed": state.get("stages_completed", []) + ["validation"],
        }

    ontology = get_ontology()

    # 1. Type constraints
    for edge in subgraph.edges:
        subject = _find_node(subgraph.nodes, edge.subject_id)
        obj = _find_node(subgraph.nodes, edge.object_id)
        if subject and obj:
            valid, err = ontology.validate_relation(subject.entity_type, edge.predicate, obj.entity_type)
            if not valid and err:
                errors.append(err)

    # 2. Evidence constraints
    for node in subgraph.nodes:
        if node.entity_type.value == "AIRiskIncident":
            continue
        if node.extraction_mode in (ExtractionMode.EXPLICIT, ExtractionMode.ABSTRACTED):
            if not node.evidence:
                errors.append(f"Node {node.name} ({node.entity_type.value}) has no evidence")

    # 3. Evidence source_doc_id validity
    for node in subgraph.nodes:
        if node.entity_type.value == "AIRiskIncident":
            continue
        if node.extraction_mode in (ExtractionMode.EXPLICIT, ExtractionMode.ABSTRACTED):
            if not node.source_doc_ids:
                errors.append(
                    f"Node {node.name} ({node.entity_type.value}) has empty source_doc_ids"
                )
        for ev in node.evidence:
            if not ev.source_doc_id or ev.source_doc_id == "event_level":
                errors.append(
                    f"Node {node.name}: evidence has invalid source_doc_id "
                    f"('{ev.source_doc_id}')"
                )

    for edge in subgraph.edges:
        for ev in edge.evidence:
            if not ev.source_doc_id or ev.source_doc_id == "event_level":
                errors.append(
                    f"Edge {edge.predicate.value}: evidence has invalid source_doc_id "
                    f"('{ev.source_doc_id}')"
                )

    # 4. Inference constraints
    for node in subgraph.nodes:
        if node.extraction_mode == ExtractionMode.INFERRED:
            if not node.evidence:
                errors.append(f"Inferred node {node.name} has no evidence")
            if not node.reasoning:
                errors.append(f"Inferred node {node.name} has no reasoning")

    for edge in subgraph.edges:
        if edge.extraction_mode == ExtractionMode.INFERRED:
            if not edge.reasoning:
                errors.append(f"Edge {edge.predicate.value}: inferred without reasoning")

    # 5. Risk chain completeness
    edge_dicts = [
        {"predicate": e.predicate.value, "object_type": _find_node_type(subgraph.nodes, e.object_id)}
        for e in subgraph.edges
    ]
    chain_errors = ontology.validate_risk_chain_completeness(edge_dicts)
    errors.extend(chain_errors)

    passed = len(errors) == 0
    retry_count = state.get("retry_count", 0)

    if not passed:
        retry_count += 1

    logger.info(
        f"[Validation] Event {event_id}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} errors, retry={retry_count})"
    )

    return {
        "validation_errors": errors,
        "validation_passed": passed,
        "retry_count": retry_count,
        "current_stage": "validation",
        "stages_completed": state.get("stages_completed", []) + ["validation"],
    }


def _find_node(nodes, node_id: str):
    for n in nodes:
        if n.id == node_id:
            return n
    return None


def _find_node_type(nodes, node_id: str) -> str:
    node = _find_node(nodes, node_id)
    return node.entity_type.value if node else ""
