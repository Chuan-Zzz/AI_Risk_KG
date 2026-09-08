"""Ablation experiment runners for the gold-standard event set."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agent.nodes.aggregation import aggregation_node
from src.agent.nodes.entity_reuse import entity_reuse_node
from src.agent.nodes.explicit_extract import explicit_extract_node
from src.agent.nodes.graph_build import _build_incident_node, _build_knowledge_statements, graph_build_node
from src.agent.nodes.inference import inference_node
from src.agent.nodes.input_node import input_node
from src.agent.nodes.risk_chain import risk_chain_node
from src.agent.nodes.validation import validation_node
from src.agent.state import PipelineState, create_initial_state
from src.core.config import get_config
from src.core.models import EntityNode, EventKnowledgeSubgraph, EvidenceItem, NewsReport, RelationEdge
from src.experiments.flags import build_experiment_config
from src.storage.ttl_store import write_ttl
from src.utils.text import normalize_entity_name

logger = logging.getLogger(__name__)

def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _merge_evidence(existing: list[EvidenceItem], incoming: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in existing + incoming:
        key = (item.source_doc_id, item.evidence_sentence.strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _node_key(node: EntityNode) -> tuple[str, str]:
    return (normalize_entity_name(node.name).lower(), node.entity_type.value)


def _edge_key(edge: RelationEdge) -> tuple[str, str, str]:
    return (
        normalize_entity_name(edge.subject_name).lower(),
        edge.predicate.value,
        normalize_entity_name(edge.object_name).lower(),
    )


def _merge_node(existing: EntityNode, incoming: EntityNode) -> EntityNode:
    existing.evidence = _merge_evidence(existing.evidence, incoming.evidence)
    existing.source_doc_ids = _unique_preserve_order(existing.source_doc_ids + incoming.source_doc_ids)
    existing.support_count = max(existing.support_count, len(existing.source_doc_ids), incoming.support_count)
    existing.confidence = round(max(existing.confidence, incoming.confidence), 2)
    if incoming.description and not existing.description:
        existing.description = incoming.description
    if len(incoming.name) > len(existing.name):
        existing.name = incoming.name
    if incoming.attributes:
        existing.attributes.update(incoming.attributes)
    if incoming.reasoning and not existing.reasoning:
        existing.reasoning = incoming.reasoning
    return existing


def _merge_edge(existing: RelationEdge, incoming: RelationEdge) -> RelationEdge:
    existing.evidence = _merge_evidence(existing.evidence, incoming.evidence)
    existing.source_doc_ids = _unique_preserve_order(existing.source_doc_ids + incoming.source_doc_ids)
    existing.support_count = max(existing.support_count, len(existing.source_doc_ids), incoming.support_count)
    existing.confidence = round(max(existing.confidence, incoming.confidence), 2)
    if incoming.reasoning and not existing.reasoning:
        existing.reasoning = incoming.reasoning
    return existing


def _run_linear_pipeline(state: PipelineState) -> PipelineState:
    for node in (
        input_node,
        explicit_extract_node,
        aggregation_node,
        entity_reuse_node,
        risk_chain_node,
        inference_node,
        graph_build_node,
    ):
        state.update(node(state))

    state.update(validation_node(state))
    if not state.get("validation_passed", False):
        logger.warning("[Ablation] Validation failed for %s", state.get("event_id", "unknown"))

    if state.get("event_subgraph") is not None and (
        state.get("validation_passed", False)
        or not get_config().get("validation.strict_mode", True)
    ):
        state["status"] = "COMPLETED"
    elif state.get("event_subgraph") is not None:
        state["status"] = "FAILED_VALIDATION"
        state["error_message"] = "; ".join(state.get("validation_errors", [])[:5])
    else:
        state["status"] = "FAILED"
        state["error_message"] = "No subgraph generated"
    return state


def _merge_subgraphs_simple_union(
    event_id: str,
    documents: list[NewsReport],
    subgraphs: list[EventKnowledgeSubgraph],
    metadata_state: PipelineState,
) -> EventKnowledgeSubgraph:
    incident_node = _build_incident_node(event_id, documents)
    canonical_nodes: dict[tuple[str, str], EntityNode] = {}
    canonical_edges: dict[tuple[str, str, str], RelationEdge] = {}

    node_key = _node_key(incident_node)
    canonical_nodes[node_key] = incident_node

    for subgraph in subgraphs:
        id_map: dict[str, str] = {subgraph.incident_node.id: incident_node.id}

        for node in subgraph.nodes:
            if node.entity_type.value == "AIRiskIncident":
                id_map[node.id] = incident_node.id
                continue

            key = _node_key(node)
            if key in canonical_nodes:
                canonical = _merge_node(canonical_nodes[key], node.model_copy(deep=True))
            else:
                canonical = node.model_copy(deep=True)
                canonical_nodes[key] = canonical
            id_map[node.id] = canonical.id

        id_to_name: dict[str, str] = {node.id: node.name for node in canonical_nodes.values()}

        for edge in subgraph.edges:
            copied = edge.model_copy(deep=True)
            copied.subject_id = id_map.get(copied.subject_id, copied.subject_id)
            copied.object_id = id_map.get(copied.object_id, copied.object_id)
            copied.subject_name = id_to_name.get(copied.subject_id, copied.subject_name)
            copied.object_name = id_to_name.get(copied.object_id, copied.object_name)
            copied.id = f"{copied.subject_id}_{copied.predicate.value}_{copied.object_id}"

            key = _edge_key(copied)
            if key in canonical_edges:
                canonical_edges[key] = _merge_edge(canonical_edges[key], copied)
            else:
                canonical_edges[key] = copied

    nodes = list(canonical_nodes.values())
    edges = list(canonical_edges.values())
    support_statistics: dict[str, int] = {}

    return EventKnowledgeSubgraph(
        event_id=event_id,
        incident_node=incident_node,
        nodes=nodes,
        edges=edges,
        knowledge_statements=_build_knowledge_statements(edges),
        support_statistics=support_statistics,
        report_count=metadata_state.get("report_count", len(documents)),
        first_seen=metadata_state.get("first_seen"),
        last_seen=metadata_state.get("last_seen"),
    )


def run_ablation_variant(event_id: str, documents: list[NewsReport], variant: str) -> PipelineState:
    experiment = build_experiment_config(variant)

    if variant != "no_agg":
        state = create_initial_state(event_id, documents, experiment=experiment)
        return _run_linear_pipeline(state)

    metadata_state = create_initial_state(event_id, documents, experiment=experiment)
    metadata_state.update(input_node(metadata_state))

    doc_subgraphs: list[EventKnowledgeSubgraph] = []
    failed_docs: list[str] = []

    for doc in documents:
        doc_state = create_initial_state(event_id, [doc], experiment=experiment)
        doc_state = _run_linear_pipeline(doc_state)
        subgraph = doc_state.get("event_subgraph")
        if subgraph is None:
            failed_docs.append(doc.doc_id)
            continue
        doc_subgraphs.append(subgraph)

    if not doc_subgraphs:
        metadata_state["status"] = "FAILED"
        metadata_state["error_message"] = "No document-level subgraphs generated"
        return metadata_state

    merged = _merge_subgraphs_simple_union(event_id, documents, doc_subgraphs, metadata_state)
    metadata_state["event_subgraph"] = merged
    metadata_state["current_stage"] = "graph_build"
    metadata_state["stages_completed"] = metadata_state.get("stages_completed", []) + ["doc_union"]
    metadata_state["validation_passed"] = False
    metadata_state["validation_errors"] = []
    metadata_state["retry_count"] = 0
    if failed_docs:
        metadata_state["error_message"] = f"Document-level failures: {', '.join(failed_docs[:10])}"

    metadata_state.update(validation_node(metadata_state))
    if metadata_state.get("validation_passed", False) or not get_config().get("validation.strict_mode", True):
        metadata_state["status"] = "COMPLETED"
    else:
        metadata_state["status"] = "FAILED_VALIDATION"
        metadata_state["error_message"] = "; ".join(metadata_state.get("validation_errors", [])[:5])
    return metadata_state


def save_subgraph_outputs(subgraph: EventKnowledgeSubgraph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "event_subgraph.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subgraph.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    ttl_path = output_dir / "event_subgraph.ttl"
    write_ttl(subgraph, ttl_path)
