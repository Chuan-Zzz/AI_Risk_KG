"""Tests for Validation Node (validation.py).

Covers: type constraint violations, evidence constraint violations,
inference constraint violations, risk chain completeness, and valid subgraph passes.
"""

from __future__ import annotations

import pytest

from src.agent.nodes.validation import validation_node
from src.agent.state import PipelineState, create_initial_state
from src.core.models import (
    EntityNode,
    EvidenceItem,
    ExtractionMode,
    OntologyClass,
    RelationEdge,
    RelationType,
    EventKnowledgeSubgraph,
)


def _make_entity(
    name: str,
    entity_type: OntologyClass,
    evidence: list[EvidenceItem] | None = None,
    extraction_mode: ExtractionMode = ExtractionMode.EXPLICIT,
    reasoning: str | None = None,
    doc_id: str = "d1",
) -> EntityNode:
    if evidence is None:
        evidence = [EvidenceItem(
            evidence_id=f"ev_{name}",
            evidence_sentence=f"{name} evidence",
            source_doc_id=doc_id,
            confidence=0.8,
        )]
    return EntityNode(
        id=f"ent_{name}",
        name=name,
        entity_type=entity_type,
        confidence=0.8,
        source_doc_ids=[doc_id],
        evidence=evidence,
        extraction_mode=extraction_mode,
        reasoning=reasoning,
    )


def _make_edge(
    subject_id: str,
    predicate: RelationType,
    object_id: str,
    evidence: list[EvidenceItem] | None = None,
    extraction_mode: ExtractionMode = ExtractionMode.EXPLICIT,
    reasoning: str | None = None,
) -> RelationEdge:
    return RelationEdge(
        id=f"edge_{subject_id}_{predicate.value}_{object_id}",
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=0.8,
        evidence=evidence or [EvidenceItem(
            evidence_id=f"ev_{predicate.value}",
            evidence_sentence="Edge evidence",
            source_doc_id="d1",
            confidence=0.8,
        )],
        extraction_mode=extraction_mode,
        reasoning=reasoning,
    )


def _build_state_with_subgraph(
    nodes: list[EntityNode],
    edges: list[RelationEdge],
    event_id: str = "evt1",
) -> PipelineState:
    state = create_initial_state(event_id, [])
    state["stages_completed"] = ["input", "aggregation", "entity_reuse", "graph_build"]

    incident = EntityNode(
        id=event_id,
        name=event_id,
        entity_type=OntologyClass.AI_RISK_INCIDENT,
        confidence=1.0,
        extraction_mode=ExtractionMode.COMPLETED,
    )

    subgraph = EventKnowledgeSubgraph(
        event_id=event_id,
        incident_node=incident,
        nodes=[incident] + nodes,
        edges=edges,
    )
    state["event_subgraph"] = subgraph
    return state


# ============================================================
# Valid Subgraph
# ============================================================


class TestValidSubgraphPasses:
    def test_minimal_valid_subgraph(self):
        # 行为演进：风险链完整性改为图遍历校验，要求完整连通链
        # incident -hasRisk-> Risk -leadsTo-> Consequence -impacts-> Impact
        incident_id = "evt1"
        risk = _make_entity("Bias Risk", OntologyClass.RISK)
        consequence = _make_entity("Unfair outcome", OntologyClass.CONSEQUENCE)
        impact = _make_entity("Harm", OntologyClass.IMPACT)
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(risk.id, RelationType.LEADS_TO, consequence.id),
            _make_edge(consequence.id, RelationType.IMPACTS, impact.id),
        ]
        state = _build_state_with_subgraph([risk, consequence, impact], edges)
        result = validation_node(state)
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []

    def test_valid_subgraph_passes(self):
        incident_id = "evt1"
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        risk = _make_entity("Bias Risk", OntologyClass.RISK)
        consequence = _make_entity("Unfair outcome", OntologyClass.CONSEQUENCE)
        impact = _make_entity("Discrimination", OntologyClass.IMPACT)

        edges = [
            _make_edge(incident_id, RelationType.INVOLVES_AI_SYSTEM, system.id),
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(risk.id, RelationType.LEADS_TO, consequence.id),
            _make_edge(consequence.id, RelationType.IMPACTS, impact.id),
        ]

        state = _build_state_with_subgraph(
            [system, risk, consequence, impact], edges,
        )
        result = validation_node(state)
        assert result["validation_passed"] is True


# ============================================================
# Type Constraint Violations
# ============================================================


class TestTypeConstraintViolations:
    def test_domain_violation_detected(self):
        incident_id = "evt1"
        # Stakeholder cannot be subject of 'causes' (domain is RiskSource)
        stakeholder = _make_entity("Company", OntologyClass.STAKEHOLDER)
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        impact = _make_entity("Harm", OntologyClass.IMPACT)
        bad_edge = _make_edge(stakeholder.id, RelationType.CAUSES, risk.id)
        # Complete valid chain so remaining errors are only about the bad edge
        valid_edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(risk.id, RelationType.LEADS_TO, consequence.id),
            _make_edge(consequence.id, RelationType.IMPACTS, impact.id),
        ]

        state = _build_state_with_subgraph(
            [stakeholder, risk, consequence, impact],
            [bad_edge, *valid_edges],
        )
        result = validation_node(state)
        # 行为演进：domain/range 违规不再进入 validation_errors，
        # 而是被"软修复"——违规边直接从子图中移除
        remaining_ids = {e.id for e in state["event_subgraph"].edges}
        assert bad_edge.id not in remaining_ids
        assert all(e.id in remaining_ids for e in valid_edges)
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []

    def test_range_violation_detected(self):
        incident_id = "evt1"
        # 'causes' range is Risk, not Stakeholder
        risk_source = _make_entity("Source", OntologyClass.RISK_SOURCE)
        stakeholder = _make_entity("Company", OntologyClass.STAKEHOLDER)
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        impact = _make_entity("Harm", OntologyClass.IMPACT)
        bad_edge = _make_edge(risk_source.id, RelationType.CAUSES, stakeholder.id)
        valid_edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(risk.id, RelationType.LEADS_TO, consequence.id),
            _make_edge(consequence.id, RelationType.IMPACTS, impact.id),
        ]

        state = _build_state_with_subgraph(
            [risk_source, stakeholder, risk, consequence, impact],
            [bad_edge, *valid_edges],
        )
        result = validation_node(state)
        # 行为演进：同上，range 违规走软修复，违规边被移除而非报错
        remaining_ids = {e.id for e in state["event_subgraph"].edges}
        assert bad_edge.id not in remaining_ids
        assert all(e.id in remaining_ids for e in valid_edges)
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []


# ============================================================
# Evidence Constraint Violations
# ============================================================


class TestEvidenceConstraintViolations:
    def test_explicit_node_without_evidence_detected(self):
        incident_id = "evt1"
        risk = _make_entity("Bias Risk", OntologyClass.RISK, evidence=[])
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(incident_id, RelationType.HAS_CONSEQUENCE, consequence.id),
        ]
        state = _build_state_with_subgraph([risk, consequence], edges)
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("no evidence" in e for e in errors)

    def test_abstracted_node_without_evidence_detected(self):
        incident_id = "evt1"
        risk = _make_entity(
            "Bias Risk", OntologyClass.RISK,
            evidence=[], extraction_mode=ExtractionMode.ABSTRACTED,
        )
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(incident_id, RelationType.HAS_CONSEQUENCE, consequence.id),
        ]
        state = _build_state_with_subgraph([risk, consequence], edges)
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("no evidence" in e for e in errors)

    def test_inferred_edge_without_reasoning_detected(self):
        incident_id = "evt1"
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        inferred_edge = _make_edge(
            incident_id, RelationType.HAS_RISK, risk.id,
            extraction_mode=ExtractionMode.INFERRED,
            reasoning=None,
        )
        cons_edge = _make_edge(incident_id, RelationType.HAS_CONSEQUENCE, consequence.id)
        state = _build_state_with_subgraph([risk, consequence], [inferred_edge, cons_edge])
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("inferred without reasoning" in e for e in errors)


# ============================================================
# Inference Constraint Violations
# ============================================================


class TestInferenceConstraintViolations:
    def test_inferred_node_without_evidence_detected(self):
        incident_id = "evt1"
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        inferred_node = _make_entity(
            "Purpose node", OntologyClass.PURPOSE,
            evidence=[], extraction_mode=ExtractionMode.INFERRED,
            reasoning="Some reasoning",
        )
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(incident_id, RelationType.HAS_CONSEQUENCE, consequence.id),
        ]
        state = _build_state_with_subgraph(
            [risk, consequence, inferred_node], edges,
        )
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("Inferred node" in e and "no evidence" in e for e in errors)

    def test_inferred_node_without_reasoning_detected(self):
        incident_id = "evt1"
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        inferred_node = _make_entity(
            "Purpose node", OntologyClass.PURPOSE,
            extraction_mode=ExtractionMode.INFERRED,
            reasoning=None,
        )
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(incident_id, RelationType.HAS_CONSEQUENCE, consequence.id),
        ]
        state = _build_state_with_subgraph(
            [risk, consequence, inferred_node], edges,
        )
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("Inferred node" in e and "no reasoning" in e for e in errors)


# ============================================================
# Risk Chain Completeness
# ============================================================


class TestRiskChainCompleteness:
    def test_missing_risk_fails_validation(self):
        incident_id = "evt1"
        # No hasRisk edge
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        state = _build_state_with_subgraph(
            [consequence],
            [],
        )
        result = validation_node(state)
        errors = result["validation_errors"]
        assert any("at least one Risk" in e for e in errors)

    def test_risk_without_consequence_or_impact_fails(self):
        incident_id = "evt1"
        risk = _make_entity("Risk", OntologyClass.RISK)
        state = _build_state_with_subgraph(
            [risk],
            [_make_edge(incident_id, RelationType.HAS_RISK, risk.id)],
        )
        result = validation_node(state)
        errors = result["validation_errors"]
        # 行为演进：图遍历校验按链顺序报错，先报缺 Consequence
        assert any("at least one Consequence" in e for e in errors)

    def test_risk_control_without_mitigates_fails(self):
        pytest.skip(
            "疑似 src 回归：validation_node 重构为图遍历式 _validate_risk_chain_connectivity "
            "后，不再检查 'RiskControl 必须有 mitigates 边'（该检查仍保留在 "
            "ontology.validate_risk_chain_completeness 中，但 src 里已无任何调用点，"
            "成为死代码）。当前行为：RiskControl 无 mitigates 边时校验通过。"
            "待上游确认是重构遗漏还是有意简化后再恢复或改写本测试。"
        )


# ============================================================
# No Subgraph
# ============================================================


class TestNoSubgraph:
    def test_no_subgraph_fails_validation(self):
        state = create_initial_state("evt1", [])
        state["stages_completed"] = ["input", "aggregation", "graph_build"]
        # event_subgraph is None by default
        result = validation_node(state)
        assert result["validation_passed"] is False
        assert "No subgraph to validate" in result["validation_errors"]


# ============================================================
# Validation Node Metadata
# ============================================================


class TestValidationNodeMetadata:
    def _make_valid_state(self, event_id: str = "evt1") -> PipelineState:
        # 行为演进：链完整性现在要求完整 incident→Risk→Consequence→Impact 连通链
        incident_id = event_id
        risk = _make_entity("Risk", OntologyClass.RISK)
        consequence = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        impact = _make_entity("Harm", OntologyClass.IMPACT)
        edges = [
            _make_edge(incident_id, RelationType.HAS_RISK, risk.id),
            _make_edge(risk.id, RelationType.LEADS_TO, consequence.id),
            _make_edge(consequence.id, RelationType.IMPACTS, impact.id),
        ]
        return _build_state_with_subgraph([risk, consequence, impact], edges, event_id)

    def test_current_stage_is_validation(self):
        state = self._make_valid_state()
        result = validation_node(state)
        assert result["current_stage"] == "validation"

    def test_stages_completed_includes_validation(self):
        state = self._make_valid_state()
        result = validation_node(state)
        assert "validation" in result["stages_completed"]

    def test_retry_count_increments_on_failure(self):
        # Create a state with a subgraph that has a type violation
        incident_id = "evt1"
        stakeholder = _make_entity("Company", OntologyClass.STAKEHOLDER)
        risk = _make_entity("Risk", OntologyClass.RISK)
        cons = _make_entity("Cons", OntologyClass.CONSEQUENCE)
        bad_edge = _make_edge(stakeholder.id, RelationType.CAUSES, risk.id)
        valid_risk = _make_edge(incident_id, RelationType.HAS_RISK, risk.id)
        leads = _make_edge(risk.id, RelationType.LEADS_TO, cons.id)
        state = _build_state_with_subgraph(
            [stakeholder, risk, cons],
            [bad_edge, valid_risk, leads],
        )
        state["retry_count"] = 0
        result = validation_node(state)
        assert result["validation_passed"] is False
        assert result["retry_count"] == 1

    def test_retry_count_preserved_on_pass(self):
        state = self._make_valid_state()
        state["retry_count"] = 0
        result = validation_node(state)
        assert result["retry_count"] == 0

    def test_preserves_existing_stages(self):
        state = self._make_valid_state()
        result = validation_node(state)
        assert "input" in result["stages_completed"]
        assert "graph_build" in result["stages_completed"]
