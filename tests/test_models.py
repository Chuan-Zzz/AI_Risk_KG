"""Tests for core data models (models.py).

Covers: enums, Pydantic model creation/validation, field constraints,
CLASS_ALIASES, RELATION_ALIASES, and serialization/deserialization.
"""

from __future__ import annotations

import pytest

from src.core.models import (
    CLASS_ALIASES,
    RELATION_ALIASES,
    EntityNode,
    EventKnowledgeSubgraph,
    EvidenceItem,
    ExtractionMode,
    KnowledgeStatement,
    NewsReport,
    OntologyClass,
    RelationEdge,
    RelationType,
)


# ============================================================
# ExtractionMode Enum
# ============================================================


class TestExtractionMode:
    def test_all_four_modes_exist(self):
        assert len(ExtractionMode) == 4

    def test_explicit_value(self):
        assert ExtractionMode.EXPLICIT.value == "explicit"

    def test_abstracted_value(self):
        assert ExtractionMode.ABSTRACTED.value == "abstracted"

    def test_inferred_value(self):
        assert ExtractionMode.INFERRED.value == "inferred"

    def test_completed_value(self):
        assert ExtractionMode.COMPLETED.value == "completed"

    def test_string_comparison(self):
        assert ExtractionMode.EXPLICIT == "explicit"


# ============================================================
# OntologyClass Enum
# ============================================================


class TestOntologyClass:
    def test_total_class_count(self):
        assert len(OntologyClass) == 41

    def test_technical_classes(self):
        tech = {"AISystem", "AIModel", "AITechnique", "AICapability"}
        assert tech.issubset({m.value for m in OntologyClass})

    def test_stakeholder_classes(self):
        stakeholders = {
            "Stakeholder", "AIDeveloper", "AIProvider", "AIDeployer",
            "AIUser", "Regulator", "AffectedActor",
        }
        assert stakeholders.issubset({m.value for m in OntologyClass})

    def test_risk_classes(self):
        risk = {
            "RiskSource", "Risk", "Hazard", "Threat", "Vulnerability",
            "Consequence", "Impact", "RiskControl",
        }
        assert risk.issubset({m.value for m in OntologyClass})

    def test_governance_classes(self):
        gov = {
            "Regulation", "Standard", "Obligation", "ComplianceRequirement",
            "EnforcementAction", "CodeOfConduct",
        }
        assert gov.issubset({m.value for m in OntologyClass})

    def test_evidence_classes(self):
        evidence = {"NewsReport", "InformationSource", "Evidence", "KnowledgeStatement"}
        assert evidence.issubset({m.value for m in OntologyClass})

    def test_event_classes(self):
        events = {"AIRiskIncident", "RoleAssignment"}
        assert events.issubset({m.value for m in OntologyClass})

    def test_auxiliary_classes(self):
        aux = {
            "ExtractionMode", "Trend", "Purpose", "AILifecyclePhase",
            "Domain", "Documentation", "TechnicalDocumentation",
            "VerificationTest",
        }
        assert aux.issubset({m.value for m in OntologyClass})

    def test_lookup_by_value(self):
        assert OntologyClass("AISystem") == OntologyClass.AI_SYSTEM

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OntologyClass("NonExistentClass")


# ============================================================
# RelationType Enum
# ============================================================


class TestRelationType:
    def test_total_relation_count(self):
        assert len(RelationType) == 37

    def test_technical_relations(self):
        tech = {
            "develops", "provides", "deploys", "uses", "hasModel",
            "usesTechnique", "hasCapability",
        }
        assert tech.issubset({m.value for m in RelationType})

    def test_event_core_relations(self):
        event = {
            "involvesAISystem", "involvesStakeholder", "hasRisk",
            "hasConsequence", "hasImpact", "hasReport",
        }
        assert event.issubset({m.value for m in RelationType})

    def test_risk_chain_relations(self):
        chain = {"causes", "leadsTo", "impacts", "affects", "mitigates"}
        assert chain.issubset({m.value for m in RelationType})

    def test_role_relations(self):
        roles = {"roleHeldBy", "hasRoleInIncident", "roleInvolvesSystem"}
        assert roles.issubset({m.value for m in RelationType})

    def test_lookup_by_value(self):
        assert RelationType("causes") == RelationType.CAUSES

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RelationType("nonexistentRelation")


# ============================================================
# CLASS_ALIASES
# ============================================================


class TestClassAliases:
    def test_llm_maps_to_ai_model(self):
        assert CLASS_ALIASES["llm"] == "AIModel"

    def test_large_language_model_maps_to_ai_model(self):
        assert CLASS_ALIASES["large language model"] == "AIModel"

    def test_foundation_model_maps_to_ai_model(self):
        assert CLASS_ALIASES["foundation model"] == "AIModel"

    def test_system_maps_to_ai_system(self):
        assert CLASS_ALIASES["system"] == "AISystem"

    def test_model_maps_to_ai_model(self):
        assert CLASS_ALIASES["model"] == "AIModel"

    def test_chinese_aliases(self):
        assert CLASS_ALIASES["基础模型"] == "AIModel"
        assert CLASS_ALIASES["大模型"] == "AIModel"

    def test_law_maps_to_regulation(self):
        assert CLASS_ALIASES["law"] == "Regulation"

    def test_act_maps_to_regulation(self):
        assert CLASS_ALIASES["act"] == "Regulation"

    def test_mitigation_maps_to_risk_control(self):
        assert CLASS_ALIASES["mitigation"] == "RiskControl"

    def test_fine_maps_to_enforcement_action(self):
        assert CLASS_ALIASES["fine"] == "EnforcementAction"

    def test_victim_maps_to_affected_actor(self):
        assert CLASS_ALIASES["victim"] == "AffectedActor"

    def test_incident_maps_to_ai_risk_incident(self):
        assert CLASS_ALIASES["incident"] == "AIRiskIncident"

    def test_event_maps_to_ai_risk_incident(self):
        assert CLASS_ALIASES["event"] == "AIRiskIncident"

    def test_all_alias_values_are_valid_ontology_classes(self):
        valid = {m.value for m in OntologyClass}
        for alias, mapped in CLASS_ALIASES.items():
            assert mapped in valid, f"Alias '{alias}' maps to invalid class '{mapped}'"

    def test_no_duplicate_values_in_aliases(self):
        # Duplicates are actually fine for different alias names,
        # but each alias key should be unique
        keys = list(CLASS_ALIASES.keys())
        assert len(keys) == len(set(keys))


# ============================================================
# RELATION_ALIASES
# ============================================================


class TestRelationAliases:
    def test_developed_by_maps_to_develops(self):
        assert RELATION_ALIASES["developed_by"] == "develops"

    def test_provided_by_maps_to_provides(self):
        assert RELATION_ALIASES["provided by"] == "provides"

    def test_deployed_by_maps_to_deploys(self):
        assert RELATION_ALIASES["deployed_by"] == "deploys"

    def test_results_in_maps_to_leads_to(self):
        assert RELATION_ALIASES["results_in"] == "leadsTo"

    def test_regulates_maps_to_governs(self):
        assert RELATION_ALIASES["regulates"] == "governs"

    def test_governed_by_maps_to_governs(self):
        assert RELATION_ALIASES["governed_by"] == "governs"

    def test_all_alias_values_are_valid_relation_types(self):
        valid = {m.value for m in RelationType}
        for alias, mapped in RELATION_ALIASES.items():
            assert mapped in valid, f"Alias '{alias}' maps to invalid relation '{mapped}'"


# ============================================================
# Pydantic Models
# ============================================================


class TestNewsReport:
    def test_create_minimal(self):
        report = NewsReport(doc_id="d1", event_id="e1", title="Test", content="Body")
        assert report.doc_id == "d1"
        assert report.event_id == "e1"
        assert report.title == "Test"
        assert report.content == "Body"
        assert report.language == "en"
        assert report.summary is None
        assert report.url is None
        assert report.publish_time is None
        assert report.source_name is None

    def test_create_full(self):
        report = NewsReport(
            doc_id="d1",
            event_id="e1",
            title="Test Title",
            content="Full content here",
            summary="Short summary",
            source_name="BBC",
            url="https://example.com/article",
            publish_time="2025-01-15T10:30:00Z",
            language="zh",
        )
        assert report.source_name == "BBC"
        assert report.language == "zh"

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            NewsReport(doc_id="d1")

    def test_serialization_roundtrip(self):
        report = NewsReport(
            doc_id="d1", event_id="e1", title="T", content="C",
            publish_time="2025-01-01",
        )
        data = report.model_dump()
        restored = NewsReport.model_validate(data)
        assert restored == report


class TestEvidenceItem:
    def test_create_with_defaults(self):
        item = EvidenceItem(
            evidence_id="ev1",
            evidence_sentence="Tesla Autopilot crashed",
            source_doc_id="d1",
        )
        assert item.confidence == 0.0
        assert item.mention_span is None

    def test_create_full(self):
        item = EvidenceItem(
            evidence_id="ev1",
            evidence_sentence="OpenAI released GPT-4",
            source_doc_id="d1",
            mention_span="OpenAI",
            confidence=0.9,
        )
        assert item.confidence == 0.9
        assert item.mention_span == "OpenAI"

    def test_confidence_below_zero_raises(self):
        with pytest.raises(Exception):
            EvidenceItem(
                evidence_id="ev1",
                evidence_sentence="test",
                source_doc_id="d1",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(Exception):
            EvidenceItem(
                evidence_id="ev1",
                evidence_sentence="test",
                source_doc_id="d1",
                confidence=1.5,
            )

    def test_confidence_boundary_zero(self):
        item = EvidenceItem(
            evidence_id="ev1", evidence_sentence="t", source_doc_id="d1",
            confidence=0.0,
        )
        assert item.confidence == 0.0

    def test_confidence_boundary_one(self):
        item = EvidenceItem(
            evidence_id="ev1", evidence_sentence="t", source_doc_id="d1",
            confidence=1.0,
        )
        assert item.confidence == 1.0

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            EvidenceItem(evidence_id="ev1")


class TestEntityNode:
    def _make_entity(self, **overrides):
        defaults = dict(
            id="ent1",
            name="GPT-4",
            entity_type=OntologyClass.AI_MODEL,
        )
        defaults.update(overrides)
        return EntityNode(**defaults)

    def test_create_minimal(self):
        entity = self._make_entity()
        assert entity.id == "ent1"
        assert entity.name == "GPT-4"
        assert entity.entity_type == OntologyClass.AI_MODEL
        assert entity.extraction_mode == ExtractionMode.EXPLICIT
        assert entity.confidence == 0.0
        assert entity.support_count == 1
        assert entity.evidence == []
        assert entity.source_doc_ids == []
        assert entity.attributes == {}
        assert entity.description is None
        assert entity.reasoning is None

    def test_create_full(self):
        evidence = EvidenceItem(
            evidence_id="ev1", evidence_sentence="test", source_doc_id="d1",
        )
        entity = self._make_entity(
            description="A large language model",
            evidence=[evidence],
            extraction_mode=ExtractionMode.EXPLICIT,
            confidence=0.95,
            support_count=3,
            source_doc_ids=["d1", "d2", "d3"],
            attributes={"version": "4"},
            reasoning="Matched by name",
        )
        assert entity.confidence == 0.95
        assert entity.support_count == 3
        assert len(entity.evidence) == 1

    def test_support_count_zero_raises(self):
        with pytest.raises(Exception):
            self._make_entity(support_count=0)

    def test_support_count_negative_raises(self):
        with pytest.raises(Exception):
            self._make_entity(support_count=-1)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            self._make_entity(confidence=1.1)

    def test_invalid_entity_type_raises(self):
        with pytest.raises(Exception):
            self._make_entity(entity_type="InvalidType")

    def test_serialization_roundtrip(self):
        evidence = EvidenceItem(
            evidence_id="ev1", evidence_sentence="s", source_doc_id="d1", confidence=0.8,
        )
        entity = self._make_entity(
            evidence=[evidence],
            confidence=0.8,
            source_doc_ids=["d1"],
        )
        data = entity.model_dump(mode="json")
        restored = EntityNode.model_validate(data)
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type

    def test_different_entity_types(self):
        for cls in [OntologyClass.AI_SYSTEM, OntologyClass.STAKEHOLDER,
                    OntologyClass.RISK, OntologyClass.REGULATION]:
            entity = self._make_entity(entity_type=cls, name=cls.value)
            assert entity.entity_type == cls


class TestRelationEdge:
    def _make_edge(self, **overrides):
        defaults = dict(
            id="edge1",
            subject_id="ent1",
            predicate=RelationType.INVOLVES_AI_SYSTEM,
            object_id="ent2",
        )
        defaults.update(overrides)
        return RelationEdge(**defaults)

    def test_create_minimal(self):
        edge = self._make_edge()
        assert edge.extraction_mode == ExtractionMode.EXPLICIT
        assert edge.confidence == 0.0
        assert edge.support_count == 1
        assert edge.evidence == []
        assert edge.reasoning is None

    def test_create_with_reasoning(self):
        edge = self._make_edge(
            extraction_mode=ExtractionMode.INFERRED,
            reasoning="Deduced from context",
        )
        assert edge.reasoning == "Deduced from context"

    def test_confidence_validation(self):
        with pytest.raises(Exception):
            self._make_edge(confidence=2.0)

    def test_support_count_validation(self):
        with pytest.raises(Exception):
            self._make_edge(support_count=0)

    def test_serialization_roundtrip(self):
        edge = self._make_edge(confidence=0.7, source_doc_ids=["d1"])
        data = edge.model_dump(mode="json")
        restored = RelationEdge.model_validate(data)
        assert restored.predicate == edge.predicate


class TestKnowledgeStatement:
    def _make_statement(self, **overrides):
        evidence = EvidenceItem(
            evidence_id="ev1",
            evidence_sentence="OpenAI developed GPT-4",
            source_doc_id="d1",
            confidence=0.9,
        )
        defaults = dict(
            id="ks1",
            subject_id="ent1",
            predicate=RelationType.DEVELOPS,
            object_id="ent2",
            evidence=evidence,
            extraction_mode=ExtractionMode.EXPLICIT,
            confidence=0.9,
        )
        defaults.update(overrides)
        return KnowledgeStatement(**defaults)

    def test_create(self):
        ks = self._make_statement()
        assert ks.confidence == 0.9
        assert ks.support_count == 1

    def test_confidence_validation(self):
        with pytest.raises(Exception):
            self._make_statement(confidence=-0.1)

    def test_support_count_validation(self):
        with pytest.raises(Exception):
            self._make_statement(support_count=0)

    def test_serialization_roundtrip(self):
        ks = self._make_statement()
        data = ks.model_dump(mode="json")
        restored = KnowledgeStatement.model_validate(data)
        assert restored.predicate == ks.predicate


class TestEventKnowledgeSubgraph:
    def _make_subgraph(self, **overrides):
        incident = EntityNode(
            id="evt1",
            name="evt1",
            entity_type=OntologyClass.AI_RISK_INCIDENT,
            confidence=1.0,
        )
        defaults = dict(
            event_id="evt1",
            incident_node=incident,
        )
        defaults.update(overrides)
        return EventKnowledgeSubgraph(**defaults)

    def test_create_minimal(self):
        sg = self._make_subgraph()
        assert sg.event_id == "evt1"
        assert sg.nodes == []
        assert sg.edges == []
        assert sg.knowledge_statements == []
        assert sg.support_statistics == {}
        assert sg.report_count == 0
        assert sg.first_seen is None
        assert sg.last_seen is None

    def test_create_with_nodes_and_edges(self):
        incident = EntityNode(
            id="evt1", name="evt1",
            entity_type=OntologyClass.AI_RISK_INCIDENT,
            confidence=1.0,
        )
        system = EntityNode(
            id="sys1", name="GPT-4",
            entity_type=OntologyClass.AI_MODEL,
            confidence=0.9,
        )
        edge = RelationEdge(
            id="e1", subject_id="evt1",
            predicate=RelationType.INVOLVES_AI_SYSTEM,
            object_id="sys1",
            confidence=0.9,
        )
        sg = self._make_subgraph(
            nodes=[system],
            edges=[edge],
            report_count=3,
            first_seen="2025-01-01",
            last_seen="2025-01-15",
        )
        assert len(sg.nodes) == 1
        assert len(sg.edges) == 1
        assert sg.report_count == 3
        assert sg.first_seen == "2025-01-01"
        assert sg.last_seen == "2025-01-15"

    def test_serialization_roundtrip(self):
        sg = self._make_subgraph(report_count=5)
        data = sg.model_dump(mode="json")
        restored = EventKnowledgeSubgraph.model_validate(data)
        assert restored.event_id == sg.event_id
        assert restored.report_count == 5

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            EventKnowledgeSubgraph()
