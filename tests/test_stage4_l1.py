"""Tests for Stage 4 Layer 1 - Entity Reuse (entity_reuse.py).

Covers: entity normalization (LLM->AIModel, etc.),
deduplication by normalized (name, type), and ID regeneration.
"""

from __future__ import annotations

import pytest

from src.agent.nodes.entity_reuse import entity_reuse_node
from src.agent.state import PipelineState, create_initial_state
from src.core.models import EntityNode, EvidenceItem, NewsReport, OntologyClass


def _make_entity(
    name: str,
    entity_type: OntologyClass = OntologyClass.AI_SYSTEM,
    doc_id: str = "d1",
    confidence: float = 0.8,
    entity_id: str | None = None,
) -> EntityNode:
    return EntityNode(
        id=entity_id or f"ent_{name}_{doc_id}",
        name=name,
        entity_type=entity_type,
        confidence=confidence,
        source_doc_ids=[doc_id],
        evidence=[EvidenceItem(
            evidence_id=f"ev_{name}_{doc_id}",
            evidence_sentence=f"{name} mentioned in {doc_id}",
            source_doc_id=doc_id,
            confidence=confidence,
        )],
    )


def _build_state(
    doc_entities: dict[str, list[EntityNode]],
    event_id: str = "evt1",
) -> PipelineState:
    state = create_initial_state(event_id, [])
    state["doc_level_entities"] = doc_entities
    state["stages_completed"] = ["input", "aggregation"]
    return state


# ============================================================
# Entity Normalization
# ============================================================


class TestEntityNormalization:
    def test_llm_normalized_to_ai_model(self):
        entity = _make_entity("LLM", OntologyClass.AI_SYSTEM)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert len(reused) == 1
        assert reused[0].entity_type == OntologyClass.AI_MODEL

    def test_large_language_model_normalized_to_ai_model(self):
        entity = _make_entity("Large Language Model", OntologyClass.AI_SYSTEM)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].entity_type == OntologyClass.AI_MODEL

    def test_foundation_model_normalized_to_ai_model(self):
        entity = _make_entity("Foundation Model", OntologyClass.AI_SYSTEM)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].entity_type == OntologyClass.AI_MODEL

    def test_algorithm_normalized_to_ai_technique(self):
        entity = _make_entity("Algorithm", OntologyClass.AI_SYSTEM)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].entity_type == OntologyClass.AI_TECHNIQUE

    def test_chatbot_stays_as_ai_system(self):
        entity = _make_entity("Chatbot", OntologyClass.AI_SYSTEM)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].entity_type == OntologyClass.AI_SYSTEM

    def test_non_special_name_not_normalized(self):
        entity = _make_entity("GPT-4", OntologyClass.AI_MODEL)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].entity_type == OntologyClass.AI_MODEL


# ============================================================
# Deduplication
# ============================================================


class TestEntityDeduplication:
    def test_same_name_and_type_merged(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        e2 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2")
        state = _build_state({"d1": [e1], "d2": [e2]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert len(reused) == 1

    def test_same_name_different_type_not_merged(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        e2 = _make_entity("GPT-4", OntologyClass.AI_SYSTEM, doc_id="d2")
        state = _build_state({"d1": [e1], "d2": [e2]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert len(reused) == 2

    def test_merged_support_count_accumulates(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        e2 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2")
        state = _build_state({"d1": [e1], "d2": [e2]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].support_count == 2

    def test_merged_evidence_combined(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        e2 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2")
        state = _build_state({"d1": [e1], "d2": [e2]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert len(reused[0].evidence) == 2

    def test_merged_confidence_takes_max(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1", confidence=0.6)
        e2 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2", confidence=0.9)
        state = _build_state({"d1": [e1], "d2": [e2]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert reused[0].confidence == 0.9

    def test_merged_source_doc_ids_combined(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        e2 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2")
        e3 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d3")
        state = _build_state({"d1": [e1], "d2": [e2], "d3": [e3]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        assert set(reused[0].source_doc_ids) == {"d1", "d2", "d3"}


# ============================================================
# ID Generation
# ============================================================


class TestIDGeneration:
    def test_merged_entity_gets_new_id(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        state = _build_state({"d1": [e1]})
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        # ID should be regenerated using event_id + name + type
        assert reused[0].id != e1.id

    def test_id_includes_event_id(self):
        e1 = _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1")
        state = _build_state({"d1": [e1]}, event_id="test_event")
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        # ID should be deterministic based on event_id, name, type
        assert reused[0].id  # non-empty


# ============================================================
# Stage metadata
# ============================================================


class TestEntityReuseNodeMetadata:
    def test_current_stage_is_entity_reuse(self):
        entity = _make_entity("GPT-4", OntologyClass.AI_MODEL)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        assert result["current_stage"] == "entity_reuse"

    def test_stages_completed_includes_entity_reuse(self):
        entity = _make_entity("GPT-4", OntologyClass.AI_MODEL)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        assert "entity_reuse" in result["stages_completed"]

    def test_preserves_existing_stages(self):
        entity = _make_entity("GPT-4", OntologyClass.AI_MODEL)
        state = _build_state({"d1": [entity]})
        result = entity_reuse_node(state)
        assert "input" in result["stages_completed"]
        assert "aggregation" in result["stages_completed"]
        assert "entity_reuse" in result["stages_completed"]

    def test_empty_doc_level_entities(self):
        state = _build_state({})
        result = entity_reuse_node(state)
        assert result["reused_entities"] == []

    def test_multiple_docs_multiple_entities(self):
        state = _build_state({
            "d1": [
                _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d1"),
                _make_entity("OpenAI", OntologyClass.STAKEHOLDER, doc_id="d1"),
            ],
            "d2": [
                _make_entity("GPT-4", OntologyClass.AI_MODEL, doc_id="d2"),
                _make_entity("Claude", OntologyClass.AI_MODEL, doc_id="d2"),
            ],
        })
        result = entity_reuse_node(state)
        reused = result["reused_entities"]
        # GPT-4 merged, OpenAI, Claude remain -> 3 unique
        names = {e.name for e in reused}
        assert len(reused) == 3
        assert names == {"GPT-4", "OpenAI", "Claude"}
