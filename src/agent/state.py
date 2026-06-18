"""Pipeline state definition for LangGraph StateGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from src.core.models import (
    EntityNode,
    EventKnowledgeSubgraph,
    EvidenceItem,
    NewsReport,
    RelationEdge,
)


class PipelineState(TypedDict, total=False):
    # === Input ===
    event_id: str
    documents: list[NewsReport]

    # === Stage 1: News Reports Input ===
    report_count: int
    first_seen: str | None
    last_seen: str | None
    source_set: list[str]
    language_set: list[str]

    # === Stage 2: Document-level Explicit Extraction ===
    doc_level_entities: dict[str, list[EntityNode]]  # doc_id -> entities

    # === Stage 3: Event Evidence Aggregation ===
    event_evidence_package: dict[str, Any]
    representative_reports: list[str]
    core_entities: dict[str, list[str]]
    support_statistics: dict[str, int]
    conflicts: list[dict[str, Any]]

    # === Stage 4 Layer 1: Explicit Entity Reuse ===
    reused_entities: list[EntityNode]

    # === Stage 4 Layer 2: Risk Chain Slot Filling ===
    risk_chain: dict[str, Any]

    # === Stage 4 Layer 3: Controlled Inference ===
    inferences: dict[str, Any]
    role_assignments: list[dict[str, Any]]
    governance: dict[str, Any]
    system_attributes: dict[str, Any]

    # === Stage 5: Evidence-backed KG Construction ===
    event_subgraph: EventKnowledgeSubgraph | None
    validation_errors: list[str]
    validation_passed: bool
    retry_count: int

    # === Flow Control ===
    current_stage: str
    stages_completed: list[str]
    status: str  # PENDING | PROCESSING | COMPLETED | FAILED
    error_message: str | None


def create_initial_state(
    event_id: str,
    documents: list[NewsReport],
) -> PipelineState:
    return PipelineState(
        event_id=event_id,
        documents=documents,
        report_count=0,
        first_seen=None,
        last_seen=None,
        source_set=[],
        language_set=[],
        doc_level_entities={},
        event_evidence_package={},
        representative_reports=[],
        core_entities={},
        support_statistics={},
        conflicts=[],
        reused_entities=[],
        risk_chain={},
        inferences={},
        role_assignments=[],
        governance={},
        system_attributes={},
        event_subgraph=None,
        validation_errors=[],
        validation_passed=False,
        retry_count=0,
        current_stage="input",
        stages_completed=[],
        status="PENDING",
        error_message=None,
    )
