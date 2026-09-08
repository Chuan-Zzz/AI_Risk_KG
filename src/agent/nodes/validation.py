"""Consistency validation for the event knowledge subgraph.

When validation detects fixable issues (e.g. ontology domain/range violations,
self-loop edges, orphan nodes from broken chains), the node performs **soft
repair** by removing the offending edges from ``subgraph.edges`` in-place.
This prevents a single bad edge from causing the entire event to be rejected
under ``strict_mode``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.models import ExtractionMode, OntologyClass, RelationType
from src.core.ontology import get_ontology
from src.experiments.flags import is_experiment_enabled

logger = logging.getLogger(__name__)


def validation_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    subgraph = state.get("event_subgraph")
    errors: list[str] = []
    repaired: list[str] = []

    if subgraph is None:
        return {
            "validation_errors": ["No subgraph to validate"],
            "validation_passed": False,
            "current_stage": "validation",
            "retry_count": state.get("retry_count", 0) + 1,
            "stages_completed": state.get("stages_completed", []) + ["validation"],
        }

    ontology = get_ontology()
    disable_ontology = is_experiment_enabled(state, "disable_ontology_constraint")
    disable_evidence = is_experiment_enabled(state, "disable_evidence_constraint")

    # Build lookup for O(1) access
    nodes_by_id = {node.id: node for node in subgraph.nodes}
    if len(nodes_by_id) != len(subgraph.nodes):
        errors.append("Subgraph contains duplicate node IDs")

    document_ids = {document.doc_id for document in state.get("documents", [])}

    # 1. Structural integrity: remove edges with unknown endpoints or self-loops
    edges_to_keep = []
    for edge in subgraph.edges:
        if edge.subject_id == edge.object_id:
            repaired.append(f"Removed self-loop edge {edge.predicate.value}")
            continue
        if edge.subject_id not in nodes_by_id or edge.object_id not in nodes_by_id:
            repaired.append(
                f"Removed dangling edge {edge.predicate.value} "
                f"(subject={'OK' if edge.subject_id in nodes_by_id else 'MISSING'}, "
                f"object={'OK' if edge.object_id in nodes_by_id else 'MISSING'})"
            )
            continue
        edges_to_keep.append(edge)
    if len(edges_to_keep) < len(subgraph.edges):
        subgraph.edges = edges_to_keep

    # 2. Ontology constraints: soft-repair by removing violating edges
    if not disable_ontology:
        violating_edges = []
        for edge in subgraph.edges:
            subject = nodes_by_id.get(edge.subject_id)
            obj = nodes_by_id.get(edge.object_id)
            if subject and obj:
                valid, err = ontology.validate_relation(
                    subject.entity_type, edge.predicate, obj.entity_type
                )
                if not valid and err:
                    violating_edges.append((edge, err))

        if violating_edges:
            violating_ids = {e.id for e, _ in violating_edges}
            for edge, err in violating_edges:
                repaired.append(f"Removed ontology-violating edge: {err}")
            subgraph.edges = [
                e for e in subgraph.edges if e.id not in violating_ids
            ]

    # 3. Evidence constraints
    if not disable_evidence:
        for node in subgraph.nodes:
            if node.entity_type == OntologyClass.AI_RISK_INCIDENT:
                continue
            if node.extraction_mode in (ExtractionMode.EXPLICIT, ExtractionMode.ABSTRACTED):
                if not node.evidence:
                    errors.append(f"Node {node.name} ({node.entity_type.value}) has no evidence")

    # 4. Evidence source_doc_id validity
    if not disable_evidence:
        for node in subgraph.nodes:
            if node.entity_type == OntologyClass.AI_RISK_INCIDENT:
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
                elif document_ids and ev.source_doc_id not in document_ids:
                    errors.append(f"Node {node.name}: unknown source_doc_id '{ev.source_doc_id}'")

        for edge in subgraph.edges:
            for ev in edge.evidence:
                if not ev.source_doc_id or ev.source_doc_id == "event_level":
                    errors.append(
                        f"Edge {edge.predicate.value}: evidence has invalid source_doc_id "
                        f"('{ev.source_doc_id}')"
                    )
                elif document_ids and ev.source_doc_id not in document_ids:
                    errors.append(f"Edge {edge.predicate.value}: unknown source_doc_id '{ev.source_doc_id}'")

    # 5. Inference constraints
    for node in subgraph.nodes:
        if node.extraction_mode == ExtractionMode.INFERRED:
            if not disable_evidence and not node.evidence:
                errors.append(f"Inferred node {node.name} has no evidence")
            if not disable_evidence and not node.reasoning:
                errors.append(f"Inferred node {node.name} has no reasoning")

    for edge in subgraph.edges:
        if edge.extraction_mode == ExtractionMode.INFERRED and not disable_evidence:
            if not edge.reasoning:
                errors.append(f"Edge {edge.predicate.value}: inferred without reasoning")

    # 6. Risk chain completeness via graph traversal
    chain_errors = _validate_risk_chain_connectivity(subgraph, nodes_by_id)
    errors.extend(chain_errors)

    passed = len(errors) == 0
    retry_count = state.get("retry_count", 0)

    if not passed:
        retry_count += 1

    if repaired:
        logger.info(
            f"[Validation] Event {event_id}: soft-repaired {len(repaired)} issues: "
            + "; ".join(repaired[:3])
        )

    logger.info(
        f"[Validation] Event {event_id}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} errors, retry={retry_count}, repaired={len(repaired)})"
    )

    return {
        "validation_errors": errors,
        "validation_passed": passed,
        "retry_count": retry_count,
        "current_stage": "validation",
        "stages_completed": state.get("stages_completed", []) + ["validation"],
    }


def _validate_risk_chain_connectivity(subgraph, nodes_by_id: dict) -> list[str]:
    """Validate risk chain by traversing the actual graph edges.

    Checks that there is a connected path:
        incident → Risk → Consequence → Impact
    using the correct predicates at each hop. Also warns (not errors) if
    RiskSource or AffectedActor are missing from the chain.
    """
    errors: list[str] = []

    # Find the incident node
    incident = None
    for node in subgraph.nodes:
        if node.entity_type == OntologyClass.AI_RISK_INCIDENT:
            incident = node
            break
    if not incident:
        return errors  # No incident — other checks handle this

    # Build adjacency: subject_id -> [(predicate, object_id)]
    adj: dict[str, list[tuple[str, str]]] = {}
    for edge in subgraph.edges:
        adj.setdefault(edge.subject_id, []).append(
            (edge.predicate.value, edge.object_id)
        )

    # Helper: BFS from a node, following only edges with given predicates,
    # to find any target node of the given type.
    def _reach_type(
        start_id: str, predicates: set[str], target_type: OntologyClass
    ) -> str | None:
        visited = {start_id}
        queue = [start_id]
        while queue:
            current = queue.pop(0)
            for pred, obj_id in adj.get(current, []):
                if pred not in predicates:
                    continue
                if obj_id in visited:
                    continue
                obj_node = nodes_by_id.get(obj_id)
                if obj_node and obj_node.entity_type == target_type:
                    return obj_id
                visited.add(obj_id)
                queue.append(obj_id)
        return None

    # Check: incident → (hasRisk) → Risk
    risk_id = _reach_type(
        incident.id,
        {RelationType.HAS_RISK.value},
        OntologyClass.RISK,
    )
    if not risk_id:
        errors.append("AIRiskIncident must connect to at least one Risk (via hasRisk)")
        return errors

    # Check: Risk → (leadsTo | hasConsequence) → Consequence
    consequence_id = _reach_type(
        risk_id,
        {RelationType.LEADS_TO.value, RelationType.HAS_CONSEQUENCE.value},
        OntologyClass.CONSEQUENCE,
    )
    if not consequence_id:
        errors.append(
            "Risk must connect to at least one Consequence (via leadsTo or hasConsequence)"
        )
        return errors

    # Check: Consequence → (impacts | hasImpact) → Impact
    impact_id = _reach_type(
        consequence_id,
        {RelationType.IMPACTS.value, RelationType.HAS_IMPACT.value},
        OntologyClass.IMPACT,
    )
    if not impact_id:
        errors.append(
            "Consequence must connect to at least one Impact (via impacts or hasImpact)"
        )

    # Check: RiskSource → (causes) → Risk  (warning only)
    has_risk_source = any(
        n.entity_type == OntologyClass.RISK_SOURCE for n in subgraph.nodes
    )
    if not has_risk_source:
        logger.info(
            f"[Validation] RiskSource missing from risk chain "
            f"(may affect RCC metric)"
        )

    # Check: Impact → (affects) → AffectedActor  (warning only)
    has_affected_actor = any(
        n.entity_type == OntologyClass.AFFECTED_ACTOR for n in subgraph.nodes
    )
    if not has_affected_actor:
        logger.info(
            f"[Validation] AffectedActor missing from risk chain "
            f"(may affect RCC metric)"
        )

    return errors
