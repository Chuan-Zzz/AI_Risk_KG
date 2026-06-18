"""Tests for Stage 5 - Graph Build (graph_build.py).

Covers: AIRiskIncident node generation, involvesAISystem/involvesStakeholder edges,
risk chain node generation, RoleAssignment generation, KnowledgeStatement generation,
and inference node generation.
"""

from __future__ import annotations

import pytest

from src.agent.nodes.graph_build import graph_build_node
from src.agent.state import PipelineState, create_initial_state
from src.core.models import (
    EntityNode,
    EvidenceItem,
    ExtractionMode,
    NewsReport,
    OntologyClass,
    RelationType,
)


def _make_entity(
    name: str,
    entity_type: OntologyClass = OntologyClass.AI_SYSTEM,
    doc_id: str = "d1",
    evidence_text: str | None = None,
) -> EntityNode:
    evidence = []
    if evidence_text:
        evidence.append(EvidenceItem(
            evidence_id=f"ev_{name}",
            evidence_sentence=evidence_text,
            source_doc_id=doc_id,
            confidence=0.9,
        ))
    return EntityNode(
        id=f"ent_{name}",
        name=name,
        entity_type=entity_type,
        confidence=0.9,
        source_doc_ids=[doc_id],
        evidence=evidence,
    )


def _build_base_state(event_id: str = "evt1") -> PipelineState:
    state = create_initial_state(event_id, [])
    state["report_count"] = 3
    state["first_seen"] = "2025-01-01"
    state["last_seen"] = "2025-01-15"
    state["support_statistics"] = {}
    state["stages_completed"] = ["input", "aggregation", "entity_reuse", "risk_chain", "inference"]
    state["reused_entities"] = []
    state["risk_chain"] = {}
    state["inferences"] = {}
    return state


# ============================================================
# AIRiskIncident Node
# ============================================================


class TestIncidentNode:
    def test_incident_node_created(self):
        state = _build_base_state("test_event")
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        incident = subgraph.incident_node
        assert incident.entity_type == OntologyClass.AI_RISK_INCIDENT
        assert incident.id == "test_event"
        assert incident.name == "test_event"
        assert incident.confidence == 1.0

    def test_incident_in_nodes_list(self):
        state = _build_base_state()
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        incident_types = [n.entity_type for n in subgraph.nodes]
        assert OntologyClass.AI_RISK_INCIDENT in incident_types


# ============================================================
# involvesAISystem / involvesStakeholder Edges
# ============================================================


class TestSystemAndStakeholderEdges:
    def test_ai_system_creates_involves_edge(self):
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [system]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        involves_edges = [
            e for e in subgraph.edges
            if e.predicate == RelationType.INVOLVES_AI_SYSTEM
        ]
        assert len(involves_edges) == 1
        assert involves_edges[0].object_id == "ent_GPT-4"

    def test_stakeholder_creates_involves_edge(self):
        stakeholder = _make_entity("OpenAI", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        involves_edges = [
            e for e in subgraph.edges
            if e.predicate == RelationType.INVOLVES_STAKEHOLDER
        ]
        assert len(involves_edges) == 1

    def test_ai_model_does_not_create_involves_edge(self):
        model = _make_entity("GPT-4", OntologyClass.AI_MODEL)
        state = _build_base_state()
        state["reused_entities"] = [model]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        involves_edges = [
            e for e in subgraph.edges
            if e.predicate in (RelationType.INVOLVES_AI_SYSTEM, RelationType.INVOLVES_STAKEHOLDER)
        ]
        assert len(involves_edges) == 0

    def test_reused_entities_added_to_nodes(self):
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        stakeholder = _make_entity("OpenAI", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [system, stakeholder]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        names = {n.name for n in subgraph.nodes}
        assert "GPT-4" in names
        assert "OpenAI" in names


# ============================================================
# Risk Chain Nodes
# ============================================================


class TestRiskChainNodes:
    def test_risk_chain_creates_risk_node(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Bias in hiring", "evidence": "Evidence text", "confidence": 0.8},
            "consequence": {"value": "Unfair hiring", "evidence": "Ev", "confidence": 0.7},
            "impact": {"value": "Discrimination", "evidence": "Ev", "confidence": 0.75},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        risk_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.RISK]
        assert len(risk_nodes) == 1
        assert risk_nodes[0].name == "Bias in hiring"

    def test_risk_chain_creates_consequence_node(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "consequence": {"value": "Unfair outcome", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        cons_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.CONSEQUENCE]
        assert len(cons_nodes) == 1
        assert cons_nodes[0].name == "Unfair outcome"

    def test_risk_chain_creates_impact_node(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "impact": {"value": "Harm to society", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        impact_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.IMPACT]
        assert len(impact_nodes) == 1

    def test_risk_chain_creates_risk_source_node(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk_source": {"value": "Training data bias", "evidence": "Ev", "confidence": 0.7},
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        rs_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.RISK_SOURCE]
        assert len(rs_nodes) == 1
        assert rs_nodes[0].name == "Training data bias"

    def test_risk_chain_creates_affected_actor_node(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "impact": {"value": "Impact", "evidence": "Ev", "confidence": 0.7},
            "affected_actor": {"value": "Job applicants", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        aa_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.AFFECTED_ACTOR]
        assert len(aa_nodes) == 1
        assert aa_nodes[0].name == "Job applicants"

    def test_has_risk_edge_created(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Bias", "evidence": "Ev", "confidence": 0.8},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        has_risk_edges = [e for e in subgraph.edges if e.predicate == RelationType.HAS_RISK]
        assert len(has_risk_edges) == 1
        assert has_risk_edges[0].subject_id == "evt1"

    def test_chain_causes_edge(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk_source": {"value": "Data bias", "evidence": "Ev", "confidence": 0.7},
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        causes_edges = [e for e in subgraph.edges if e.predicate == RelationType.CAUSES]
        assert len(causes_edges) == 1

    def test_chain_leads_to_edge(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "consequence": {"value": "Consequence", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        leads_edges = [e for e in subgraph.edges if e.predicate == RelationType.LEADS_TO]
        assert len(leads_edges) == 1

    def test_chain_impacts_edge(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "consequence": {"value": "Consequence", "evidence": "Ev", "confidence": 0.7},
            "impact": {"value": "Impact", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        impacts_edges = [e for e in subgraph.edges if e.predicate == RelationType.IMPACTS]
        assert len(impacts_edges) == 1

    def test_chain_affects_edge(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Risk", "evidence": "Ev", "confidence": 0.8},
            "impact": {"value": "Impact", "evidence": "Ev", "confidence": 0.7},
            "affected_actor": {"value": "Users", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        affects_edges = [e for e in subgraph.edges if e.predicate == RelationType.AFFECTS]
        assert len(affects_edges) == 1

    def test_risk_control_mitigates_edge(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Bias", "evidence": "Ev", "confidence": 0.8},
            "risk_control": {"value": "Audit system", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        mitigates_edges = [e for e in subgraph.edges if e.predicate == RelationType.MITIGATES]
        assert len(mitigates_edges) == 1

    def test_hazard_created_and_connected_to_risk(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Unmonitored AI", "evidence": "Ev", "confidence": 0.8},
            "hazard": {"value": "No human oversight", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        hazard_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.HAZARD]
        assert len(hazard_nodes) == 1
        hazard_edges = [e for e in subgraph.edges if e.predicate == RelationType.HAS_HAZARD]
        assert len(hazard_edges) == 1
        assert hazard_edges[0].object_id == hazard_nodes[0].id

    def test_threat_created_and_connected_to_risk(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "Model manipulation", "evidence": "Ev", "confidence": 0.8},
            "threat": {"value": "Adversarial attacks", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        threat_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.THREAT]
        assert len(threat_nodes) == 1
        threat_edges = [e for e in subgraph.edges if e.predicate == RelationType.HAS_THREAT]
        assert len(threat_edges) == 1

    def test_vulnerability_created_and_connected_to_ai_system(self):
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [system]
        state["risk_chain"] = {
            "risk": {"value": "Data leakage", "evidence": "Ev", "confidence": 0.8},
            "vulnerability": {"value": "Insufficient input validation", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        vuln_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.VULNERABILITY]
        assert len(vuln_nodes) == 1
        vuln_edges = [e for e in subgraph.edges if e.predicate == RelationType.HAS_VULNERABILITY]
        assert len(vuln_edges) == 1
        assert vuln_edges[0].subject_id == "ent_GPT-4"

    def test_vulnerability_falls_back_to_risk_source(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk_source": {"value": "Weak training data", "evidence": "Ev", "confidence": 0.7},
            "risk": {"value": "Bias", "evidence": "Ev", "confidence": 0.8},
            "vulnerability": {"value": "No bias checking", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        vuln_edges = [e for e in subgraph.edges if e.predicate == RelationType.HAS_VULNERABILITY]
        assert len(vuln_edges) == 1
        rs_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.RISK_SOURCE]
        assert vuln_edges[0].subject_id == rs_nodes[0].id

    def test_empty_risk_chain_no_chain_nodes(self):
        state = _build_base_state()
        state["risk_chain"] = {}
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        chain_types = {
            OntologyClass.RISK_SOURCE, OntologyClass.RISK,
            OntologyClass.CONSEQUENCE, OntologyClass.IMPACT,
            OntologyClass.AFFECTED_ACTOR, OntologyClass.RISK_CONTROL,
        }
        chain_nodes = [n for n in subgraph.nodes if n.entity_type in chain_types]
        assert len(chain_nodes) == 0

    def test_slot_with_empty_value_skipped(self):
        state = _build_base_state()
        state["risk_chain"] = {
            "risk": {"value": "", "evidence": "Ev", "confidence": 0.8},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        risk_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.RISK]
        assert len(risk_nodes) == 0


# ============================================================
# RoleAssignment Generation
# ============================================================


class TestRoleAssignment:
    def test_developer_role_detected(self):
        stakeholder = _make_entity("OpenAI", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder]
        state["role_assignments"] = [
            {"stakeholder_name": "OpenAI", "role": "AIDeveloper",
             "evidence": "OpenAI developed GPT-4", "source_doc_id": "d1",
             "reasoning": "developed", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        role_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.ROLE_ASSIGNMENT]
        assert len(role_nodes) == 1
        assert "AIDeveloper" in role_nodes[0].name

    def test_provider_role_detected(self):
        stakeholder = _make_entity("Google", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder]
        state["role_assignments"] = [
            {"stakeholder_name": "Google", "role": "AIProvider",
             "evidence": "Google provides AI services", "source_doc_id": "d1",
             "reasoning": "provides", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        role_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.ROLE_ASSIGNMENT]
        assert len(role_nodes) == 1
        assert "AIProvider" in role_nodes[0].name

    def test_regulator_role_detected(self):
        regulator = _make_entity("FTC", OntologyClass.REGULATOR)
        state = _build_base_state()
        state["reused_entities"] = [regulator]
        state["role_assignments"] = [
            {"stakeholder_name": "FTC", "role": "Regulator",
             "evidence": "FTC investigated the company", "source_doc_id": "d1",
             "reasoning": "investigated", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        role_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.ROLE_ASSIGNMENT]
        assert len(role_nodes) == 1
        assert "Regulator" in role_nodes[0].name

    def test_no_role_when_no_assignment(self):
        stakeholder = _make_entity("John Doe", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder]
        state["role_assignments"] = []
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        role_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.ROLE_ASSIGNMENT]
        assert len(role_nodes) == 0

    def test_role_edges_created(self):
        stakeholder = _make_entity("OpenAI", OntologyClass.STAKEHOLDER)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder]
        state["role_assignments"] = [
            {"stakeholder_name": "OpenAI", "role": "AIDeveloper",
             "evidence": "OpenAI developed GPT-4", "source_doc_id": "d1",
             "reasoning": "developed", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        role_predicates = {e.predicate for e in subgraph.edges}
        assert RelationType.ROLE_HELD_BY in role_predicates
        assert RelationType.HAS_ROLE_IN_INCIDENT in role_predicates

    def test_developer_creates_develops_edge_to_system(self):
        stakeholder = _make_entity("OpenAI", OntologyClass.STAKEHOLDER)
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder, system]
        state["role_assignments"] = [
            {"stakeholder_name": "OpenAI", "role": "AIDeveloper",
             "evidence": "OpenAI developed GPT-4", "source_doc_id": "d1",
             "reasoning": "developed", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        develops_edges = [e for e in subgraph.edges if e.predicate == RelationType.DEVELOPS]
        assert len(develops_edges) == 1
        assert develops_edges[0].subject_id == "ent_OpenAI"
        assert develops_edges[0].object_id == "ent_GPT-4"

    def test_provider_creates_provides_edge(self):
        stakeholder = _make_entity("Google", OntologyClass.STAKEHOLDER)
        system = _make_entity("Bard", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder, system]
        state["role_assignments"] = [
            {"stakeholder_name": "Google", "role": "AIProvider",
             "evidence": "Google provides Bard", "source_doc_id": "d1",
             "reasoning": "provides", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        provides_edges = [e for e in subgraph.edges if e.predicate == RelationType.PROVIDES]
        assert len(provides_edges) == 1
        assert provides_edges[0].subject_id == "ent_Google"
        assert provides_edges[0].object_id == "ent_Bard"

    def test_deployer_creates_deploys_edge(self):
        stakeholder = _make_entity("Acme Corp", OntologyClass.STAKEHOLDER)
        system = _make_entity("HiringBot", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder, system]
        state["role_assignments"] = [
            {"stakeholder_name": "Acme Corp", "role": "AIDeployer",
             "evidence": "Acme deployed HiringBot", "source_doc_id": "d1",
             "reasoning": "deployed", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        deploys_edges = [e for e in subgraph.edges if e.predicate == RelationType.DEPLOYS]
        assert len(deploys_edges) == 1
        assert deploys_edges[0].subject_id == "ent_Acme Corp"

    def test_user_creates_uses_edge(self):
        stakeholder = _make_entity("Users", OntologyClass.AI_USER)
        system = _make_entity("ChatGPT", OntologyClass.AI_SYSTEM)
        state = _build_base_state()
        state["reused_entities"] = [stakeholder, system]
        state["role_assignments"] = [
            {"stakeholder_name": "Users", "role": "AIUser",
             "evidence": "Users used ChatGPT", "source_doc_id": "d1",
             "reasoning": "used", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        uses_edges = [e for e in subgraph.edges if e.predicate == RelationType.USES]
        assert len(uses_edges) == 1
        assert uses_edges[0].subject_id == "ent_Users"

    def test_regulator_investigates_incident(self):
        regulator = _make_entity("FTC", OntologyClass.REGULATOR)
        state = _build_base_state()
        state["reused_entities"] = [regulator]
        state["role_assignments"] = [
            {"stakeholder_name": "FTC", "role": "Regulator",
             "evidence": "FTC investigated", "source_doc_id": "d1",
             "reasoning": "investigated", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        investigates = [e for e in subgraph.edges if e.predicate == RelationType.INVESTIGATES]
        assert len(investigates) == 1
        assert investigates[0].object_id == "evt1"

    def test_regulator_enforces_regulation(self):
        regulator = _make_entity("FTC", OntologyClass.REGULATOR)
        regulation = _make_entity("EU AI Act", OntologyClass.REGULATION)
        state = _build_base_state()
        state["reused_entities"] = [regulator, regulation]
        state["role_assignments"] = [
            {"stakeholder_name": "FTC", "role": "Regulator",
             "evidence": "FTC enforces", "source_doc_id": "d1",
             "reasoning": "enforces", "confidence": 0.9},
        ]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        enforces = [e for e in subgraph.edges if e.predicate == RelationType.ENFORCES]
        assert len(enforces) == 1
        assert enforces[0].object_id == "ent_EU AI Act"

    def test_ai_system_complies_with_regulation(self):
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        regulation = _make_entity("GDPR", OntologyClass.REGULATION)
        state = _build_base_state()
        state["reused_entities"] = [system, regulation]
        state["role_assignments"] = []
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        complies = [e for e in subgraph.edges if e.predicate == RelationType.COMPLIES_WITH_REGULATION]
        assert len(complies) == 1
        assert complies[0].subject_id == "ent_GPT-4"
        assert complies[0].object_id == "ent_GDPR"

    def test_ai_system_conforms_to_standard(self):
        system = _make_entity("GPT-4", OntologyClass.AI_SYSTEM)
        standard = _make_entity("ISO 42001", OntologyClass.STANDARD)
        state = _build_base_state()
        state["reused_entities"] = [system, standard]
        state["role_assignments"] = []
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        conforms = [e for e in subgraph.edges if e.predicate == RelationType.CONFORMS_TO_STANDARD]
        assert len(conforms) == 1
        assert conforms[0].subject_id == "ent_GPT-4"
        assert conforms[0].object_id == "ent_ISO 42001"


# ============================================================
# KnowledgeStatement Generation
# ============================================================


class TestKnowledgeStatements:
    def test_statements_generated_for_edges_with_evidence(self):
        system = _make_entity(
            "GPT-4", OntologyClass.AI_SYSTEM,
            evidence_text="GPT-4 was involved",
        )
        state = _build_base_state()
        state["reused_entities"] = [system]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        # Each edge's evidence items generate a KnowledgeStatement
        assert len(subgraph.knowledge_statements) > 0

    def test_no_statements_for_edges_without_evidence(self):
        state = _build_base_state()
        # No reused entities, no risk chain -- only incident node, no edges
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        assert len(subgraph.knowledge_statements) == 0

    def test_statement_preserves_edge_predicate(self):
        system = _make_entity(
            "GPT-4", OntologyClass.AI_SYSTEM,
            evidence_text="Evidence text",
        )
        state = _build_base_state()
        state["reused_entities"] = [system]
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        predicates = {ks.predicate for ks in subgraph.knowledge_statements}
        assert RelationType.INVOLVES_AI_SYSTEM in predicates


# ============================================================
# Inference Nodes
# ============================================================


class TestInferenceNodes:
    def test_purpose_inference_creates_node(self):
        state = _build_base_state()
        state["inferences"] = {
            "purpose": {"value": "Content generation", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        purpose_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.PURPOSE]
        assert len(purpose_nodes) == 1
        assert purpose_nodes[0].name == "Content generation"
        assert purpose_nodes[0].extraction_mode == ExtractionMode.INFERRED

    def test_lifecycle_phase_inference_creates_node(self):
        state = _build_base_state()
        state["inferences"] = {
            "lifecycle_phase": {"value": "Deployment", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        lp_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.AI_LIFECYCLE_PHASE]
        assert len(lp_nodes) == 1
        assert lp_nodes[0].name == "Deployment"

    def test_domain_inference_creates_node(self):
        state = _build_base_state()
        state["inferences"] = {
            "impact_domain": {"value": "Employment", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        domain_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.DOMAIN]
        assert len(domain_nodes) == 1

    def test_inference_with_reasoning(self):
        state = _build_base_state()
        state["inferences"] = {
            "purpose": {
                "value": "Hiring",
                "evidence": "Ev",
                "confidence": 0.7,
                "reasoning": "The system was used for resume screening",
            },
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        purpose_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.PURPOSE]
        assert purpose_nodes[0].reasoning == "The system was used for resume screening"

    def test_empty_inference_value_skipped(self):
        state = _build_base_state()
        state["inferences"] = {
            "purpose": {"value": "", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        purpose_nodes = [n for n in subgraph.nodes if n.entity_type == OntologyClass.PURPOSE]
        assert len(purpose_nodes) == 0

    def test_unknown_inference_field_skipped(self):
        state = _build_base_state()
        state["inferences"] = {
            "unknown_field": {"value": "Something", "evidence": "Ev", "confidence": 0.7},
        }
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        assert len(subgraph.nodes) == 1  # Only the incident node


# ============================================================
# Subgraph metadata
# ============================================================


class TestSubgraphMetadata:
    def test_event_id_matches(self):
        state = _build_base_state("custom_event")
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        assert subgraph.event_id == "custom_event"

    def test_report_count_propagated(self):
        state = _build_base_state()
        state["report_count"] = 5
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        assert subgraph.report_count == 5

    def test_first_last_seen_propagated(self):
        state = _build_base_state()
        state["first_seen"] = "2025-01-01"
        state["last_seen"] = "2025-01-15"
        result = graph_build_node(state)
        subgraph = result["event_subgraph"]
        assert subgraph.first_seen == "2025-01-01"
        assert subgraph.last_seen == "2025-01-15"

    def test_current_stage_is_graph_build(self):
        state = _build_base_state()
        result = graph_build_node(state)
        assert result["current_stage"] == "graph_build"

    def test_stages_completed_includes_graph_build(self):
        state = _build_base_state()
        result = graph_build_node(state)
        assert "graph_build" in result["stages_completed"]

    def test_retry_count_preserved(self):
        state = _build_base_state()
        state["retry_count"] = 2
        result = graph_build_node(state)
        assert result["retry_count"] == 2
