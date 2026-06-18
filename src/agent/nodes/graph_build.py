"""Stage 5: Evidence-backed Knowledge Graph Construction."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.models import (
    EntityNode,
    EventKnowledgeSubgraph,
    EvidenceItem,
    ExtractionMode,
    KnowledgeStatement,
    OntologyClass,
    RelationEdge,
    RelationType,
)
from src.utils.text import generate_id, generate_uuid

logger = logging.getLogger(__name__)

# Stakeholder subtypes — these should all normalize to Stakeholder for cross-event merging
_STAKEHOLDER_TYPES = frozenset({
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
    OntologyClass.AFFECTED_ACTOR,
})

_SLOT_TYPE_MAP = {
    "risk_source": OntologyClass.RISK_SOURCE,
    "risk": OntologyClass.RISK,
    "misuse": OntologyClass.MISUSE,
    "hazard": OntologyClass.HAZARD,
    "threat": OntologyClass.THREAT,
    "vulnerability": OntologyClass.VULNERABILITY,
    "consequence": OntologyClass.CONSEQUENCE,
    "impact": OntologyClass.IMPACT,
    "affected_actor": OntologyClass.AFFECTED_ACTOR,
    "risk_control": OntologyClass.RISK_CONTROL,
}

# The main chain order (Misuse connects separately, like Hazard)
_MAIN_CHAIN_ORDER = ["risk_source", "risk", "consequence", "impact", "affected_actor"]

# Maps each slot to its outgoing predicate on the main chain
_CHAIN_PREDICATE_MAP = {
    "risk_source": RelationType.CAUSES,        # RiskSource -> Risk
    "risk": RelationType.LEADS_TO,             # Risk -> Consequence (or Misuse)
    "misuse": RelationType.LEADS_TO,           # Misuse -> Consequence
    "consequence": RelationType.IMPACTS,       # Consequence -> Impact
    "impact": RelationType.AFFECTS,            # Impact -> AffectedActor
}

# Slots that get an edge from the incident node
_INCIDENT_EDGE_MAP = {
    "risk_source": RelationType.IS_RISK_SOURCE_FOR,
    "risk": RelationType.HAS_RISK,
    "misuse": RelationType.HAS_RISK,           # Misuse is a subclass of Risk
    "consequence": RelationType.HAS_CONSEQUENCE,
    "impact": RelationType.HAS_IMPACT,
}


def _get_fallback_doc_id(state: PipelineState, event_id: str) -> str:
    rep_reports = state.get("event_evidence_package", {}).get("representative_reports", [])
    return rep_reports[0] if rep_reports else event_id


def _resolve_source_doc_id(slot_data: dict, fallback_doc_id: str) -> str:
    doc_id = slot_data.get("source_doc_id", "").strip()
    if doc_id and doc_id != "event_level":
        return doc_id
    evidence_text = slot_data.get("evidence", "")
    if evidence_text.startswith("["):
        end = evidence_text.index("]") if "]" in evidence_text else -1
        if end > 0:
            extracted = evidence_text[1:end].strip()
            if extracted:
                return extracted
    return fallback_doc_id


def graph_build_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    logger.info(f"[Stage 5] Building event knowledge subgraph for {event_id}")

    nodes: list[EntityNode] = []
    edges: list[RelationEdge] = []

    # 1. AIRiskIncident node
    documents = state.get("documents", [])
    incident_node = _build_incident_node(event_id, documents)
    nodes.append(incident_node)

    # 2. NewsReport nodes + InformationSource nodes
    _add_report_nodes(state, event_id, incident_node.name, nodes, edges)

    # 3. Reuse technical/stakeholder entities + technical edges
    incident_name = incident_node.name
    _add_reused_entities(state, event_id, incident_name, nodes, edges)

    # 3. Technical detail edges (AISystem/AIModel -> Technique/Capability)
    _add_technical_edges(nodes, edges, event_id, incident_name)

    # 4. Stakeholder role assignment (from LLM inference in Stage 4 L3)
    _add_role_assignments(state, event_id, incident_name, nodes, edges)

    # 4b. Technical relations from roles (develops/provides/deploys/uses)
    _add_technical_role_edges(state, event_id, incident_name, nodes, edges)

    # 5. Risk chain nodes
    _add_risk_chain(state, event_id, incident_name, nodes, edges)

    # 6. Inference nodes + edges
    _add_inference_nodes(state, event_id, incident_name, nodes, edges)

    # 6b. Governance + Documentation + Trend nodes from Stage 4 L3
    _add_governance_nodes(state, event_id, incident_name, nodes, edges)

    # 6c. Apply inferred system attributes to AISystem nodes
    _apply_system_attributes(state, nodes)

    # 7. KnowledgeStatements
    statements = _build_knowledge_statements(edges)

    subgraph = EventKnowledgeSubgraph(
        event_id=event_id,
        incident_node=incident_node,
        nodes=nodes,
        edges=edges,
        knowledge_statements=statements,
        support_statistics=state.get("support_statistics", {}),
        report_count=state.get("report_count", 0),
        first_seen=state.get("first_seen"),
        last_seen=state.get("last_seen"),
    )

    logger.info(
        f"[Stage 5] Built subgraph: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(statements)} knowledge statements"
    )

    return {
        "event_subgraph": subgraph,
        "current_stage": "graph_build",
        "stages_completed": state.get("stages_completed", []) + ["graph_build"],
        "retry_count": state.get("retry_count", 0),
    }


def _build_incident_node(event_id: str, documents: list) -> EntityNode:
    # Pick the most representative title (shortest factual one)
    titles = [d.title for d in documents if d.title]
    name = event_id
    if titles:
        scored = [(t, len(t.split())) for t in titles if len(t.split()) >= 4]
        if scored:
            name = min(scored, key=lambda x: x[1])[0]

    doc_ids = [d.doc_id for d in documents]
    evidence = [
        EvidenceItem(
            evidence_id=generate_uuid(),
            evidence_sentence=t,
            source_doc_id=d.doc_id,
            confidence=1.0,
        )
        for d in documents if (t := d.title)
    ][:5]

    return EntityNode(
        id=event_id,
        name=name,
        entity_type=OntologyClass.AI_RISK_INCIDENT,
        description=event_id,
        evidence=evidence,
        extraction_mode=ExtractionMode.COMPLETED,
        confidence=1.0,
        source_doc_ids=doc_ids,
    )


def _add_reused_entities(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    reused = state.get("reused_entities", [])

    for entity in reused:
        nodes.append(entity)
        if entity.entity_type in (OntologyClass.AI_SYSTEM, OntologyClass.GPAI_MODEL):
            edges.append(_make_edge(
                event_id, RelationType.INVOLVES_AI_SYSTEM, entity.id,
                entity.evidence, ExtractionMode.EXPLICIT, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))
        elif entity.entity_type in _STAKEHOLDER_TYPES:
            edges.append(_make_edge(
                event_id, RelationType.INVOLVES_STAKEHOLDER, entity.id,
                entity.evidence, ExtractionMode.EXPLICIT, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))
        elif entity.entity_type == OntologyClass.REGULATION:
            edges.append(_make_edge(
                event_id, RelationType.GOVERNS, entity.id,
                entity.evidence, ExtractionMode.EXPLICIT, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))
        elif entity.entity_type == OntologyClass.STANDARD:
            edges.append(_make_edge(
                event_id, RelationType.SPECIFIES, entity.id,
                entity.evidence, ExtractionMode.EXPLICIT, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))
        elif entity.entity_type == OntologyClass.DATA:
            edges.append(_make_edge(
                event_id, RelationType.HAS_EVIDENCE, entity.id,
                entity.evidence, ExtractionMode.ABSTRACTED, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))


def _add_technical_edges(
    nodes: list[EntityNode],
    edges: list[RelationEdge],
    event_id: str,
    incident_name: str,
) -> None:
    """Connect AISystem/AIModel/GPAIModel -> usesTechnique -> AITechnique and -> hasCapability -> AICapability.

    If no AISystem/AIModel/GPAIModel exists, connect Technique/Capability directly to the incident.
    """
    ai_tech_nodes = [n for n in nodes if n.entity_type in (
        OntologyClass.AI_SYSTEM, OntologyClass.AI_MODEL, OntologyClass.GPAI_MODEL,
    )]

    techniques = [n for n in nodes if n.entity_type == OntologyClass.AI_TECHNIQUE]
    capabilities = [n for n in nodes if n.entity_type == OntologyClass.AI_CAPABILITY]
    data_nodes = [n for n in nodes if n.entity_type == OntologyClass.DATA]

    if ai_tech_nodes:
        primary = max(ai_tech_nodes, key=lambda s: s.support_count)
        for tech in techniques:
            edges.append(_make_edge(
                primary.id, RelationType.USES_TECHNIQUE, tech.id,
                tech.evidence, ExtractionMode.ABSTRACTED, tech.confidence, tech.source_doc_ids,
                subject_name=primary.name, object_name=tech.name,
            ))
        for cap in capabilities:
            edges.append(_make_edge(
                primary.id, RelationType.HAS_CAPABILITY, cap.id,
                cap.evidence, ExtractionMode.ABSTRACTED, cap.confidence, cap.source_doc_ids,
                subject_name=primary.name, object_name=cap.name,
            ))
        for d in data_nodes:
            edges.append(_make_edge(
                primary.id, RelationType.USES, d.id,
                d.evidence, ExtractionMode.ABSTRACTED, d.confidence, d.source_doc_ids,
                subject_name=primary.name, object_name=d.name,
            ))
    else:
        # No AISystem/AIModel — connect directly to incident
        for tech in techniques:
            edges.append(_make_edge(
                event_id, RelationType.USES_TECHNIQUE, tech.id,
                tech.evidence, ExtractionMode.ABSTRACTED, tech.confidence, tech.source_doc_ids,
                subject_name=incident_name, object_name=tech.name,
            ))
        for cap in capabilities:
            edges.append(_make_edge(
                event_id, RelationType.HAS_CAPABILITY, cap.id,
                cap.evidence, ExtractionMode.ABSTRACTED, cap.confidence, cap.source_doc_ids,
                subject_name=incident_name, object_name=cap.name,
            ))
        for d in data_nodes:
            edges.append(_make_edge(
                event_id, RelationType.HAS_EVIDENCE, d.id,
                d.evidence, ExtractionMode.ABSTRACTED, d.confidence, d.source_doc_ids,
                subject_name=incident_name, object_name=d.name,
            ))


def _add_role_assignments(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    role_assignments = state.get("role_assignments", [])
    if not role_assignments:
        return

    stakeholders = [e for e in nodes if e.entity_type in _STAKEHOLDER_TYPES]
    ai_systems = [e for e in nodes if e.entity_type == OntologyClass.AI_SYSTEM]

    stakeholder_map = {s.name: s for s in stakeholders}

    for ra in role_assignments:
        name = ra.get("stakeholder_name", "")
        role_str = ra.get("role", "")
        if not name or not role_str:
            continue

        role_cls = OntologyClass(role_str) if role_str in OntologyClass.__members__.values() else None
        if role_cls is None:
            continue

        s = stakeholder_map.get(name)
        if not s:
            # Fuzzy match: case-insensitive
            for sname, snode in stakeholder_map.items():
                if sname.lower() == name.lower():
                    s = snode
                    name = sname
                    break
        if not s:
            continue

        confidence = float(ra.get("confidence", 0.8))
        doc_id = ra.get("source_doc_id", "")
        reasoning = ra.get("reasoning", "")

        role_evidence = [
            EvidenceItem(
                evidence_id=generate_uuid(),
                evidence_sentence=ra.get("evidence", ""),
                source_doc_id=doc_id,
                confidence=confidence,
            )
        ] if ra.get("evidence") else []

        role_id = generate_id(event_id, s.id, role_str)
        role_node = EntityNode(
            id=role_id,
            name=f"{s.name}_{role_str}",
            entity_type=OntologyClass.ROLE_ASSIGNMENT,
            evidence=role_evidence,
            extraction_mode=ExtractionMode.INFERRED,
            confidence=confidence,
            source_doc_ids=[doc_id] if doc_id else s.source_doc_ids,
            reasoning=reasoning,
        )
        nodes.append(role_node)

        edges.append(_make_edge(role_id, RelationType.ROLE_HELD_BY, s.id, s.evidence, ExtractionMode.INFERRED, confidence, s.source_doc_ids, subject_name=role_node.name, object_name=s.name))
        edges.append(_make_edge(role_id, RelationType.HAS_ROLE_IN_INCIDENT, event_id, s.evidence, ExtractionMode.INFERRED, confidence, s.source_doc_ids, subject_name=role_node.name, object_name=incident_name))
        if ai_systems:
            edges.append(_make_edge(role_id, RelationType.ROLE_INVOLVES_SYSTEM, ai_systems[0].id, s.evidence, ExtractionMode.INFERRED, 0.7, s.source_doc_ids, subject_name=role_node.name, object_name=ai_systems[0].name))


def _add_risk_chain(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    risk_chain = state.get("risk_chain", {})
    fallback_doc_id = _get_fallback_doc_id(state, event_id)
    chain_entities: dict[str, EntityNode] = {}

    # --- Build all chain entities ---
    all_slots = ["risk_source", "risk", "hazard", "threat", "vulnerability", "consequence", "impact", "affected_actor", "risk_control"]
    for slot_name in all_slots:
        slot_data = risk_chain.get(slot_name)
        if not slot_data or not slot_data.get("value"):
            continue
        entity_type = _SLOT_TYPE_MAP.get(slot_name)
        if not entity_type:
            continue

        value = slot_data["value"]
        confidence = float(slot_data.get("confidence", 0.75))
        doc_id = _resolve_source_doc_id(slot_data, fallback_doc_id)

        entity = EntityNode(
            id=generate_id(event_id, slot_name, value[:30]),
            name=value,
            entity_type=entity_type,
            evidence=[EvidenceItem(
                evidence_id=generate_uuid(),
                evidence_sentence=slot_data.get("evidence", ""),
                source_doc_id=doc_id,
                confidence=confidence,
            )],
            extraction_mode=ExtractionMode.ABSTRACTED,
            confidence=confidence,
            source_doc_ids=[doc_id],
        )
        nodes.append(entity)
        chain_entities[slot_name] = entity

    # --- Connect main chain: RiskSource -> Risk -> Consequence -> Impact -> AffectedActor ---
    # Skip missing slots but keep the chain connected through the last present entity.
    prev_entity = None
    prev_slot_name = None
    for slot_name in _MAIN_CHAIN_ORDER:
        entity = chain_entities.get(slot_name)
        if not entity:
            continue

        # Edge from incident to this entity
        if slot_name in _INCIDENT_EDGE_MAP:
            edges.append(_make_edge(
                event_id, _INCIDENT_EDGE_MAP[slot_name], entity.id,
                entity.evidence, ExtractionMode.ABSTRACTED, entity.confidence, entity.source_doc_ids,
                subject_name=incident_name, object_name=entity.name,
            ))

        # Edge from previous present entity to this entity
        if prev_entity and prev_slot_name:
            predicate = _CHAIN_PREDICATE_MAP.get(prev_slot_name)
            if predicate:
                chain_doc_ids = list(set(prev_entity.source_doc_ids + entity.source_doc_ids))
                edges.append(_make_edge(
                    prev_entity.id, predicate, entity.id,
                    entity.evidence, ExtractionMode.ABSTRACTED,
                    min(entity.confidence, prev_entity.confidence), chain_doc_ids,
                    subject_name=prev_entity.name, object_name=entity.name,
                ))

        prev_entity = entity
        prev_slot_name = slot_name

    # --- Connect Hazard separately: RiskSource or Risk -> hasHazard -> Hazard ---
    hazard = chain_entities.get("hazard")
    if hazard:
        parent = chain_entities.get("risk_source") or chain_entities.get("risk")
        if parent:
            edges.append(_make_edge(
                parent.id, RelationType.HAS_HAZARD, hazard.id,
                hazard.evidence, ExtractionMode.ABSTRACTED, hazard.confidence,
                list(set(parent.source_doc_ids + hazard.source_doc_ids)),
                subject_name=parent.name, object_name=hazard.name,
            ))

    # --- Connect Threat: RiskSource or Risk -> hasThreat -> Threat ---
    threat = chain_entities.get("threat")
    if threat:
        parent = chain_entities.get("risk_source") or chain_entities.get("risk")
        if parent:
            edges.append(_make_edge(
                parent.id, RelationType.HAS_THREAT, threat.id,
                threat.evidence, ExtractionMode.ABSTRACTED, threat.confidence,
                list(set(parent.source_doc_ids + threat.source_doc_ids)),
                subject_name=parent.name, object_name=threat.name,
            ))

    # --- Connect Vulnerability: AISystem or RiskSource -> hasVulnerability -> Vulnerability ---
    vuln = chain_entities.get("vulnerability")
    if vuln:
        vuln_parent = None
        for n in nodes:
            if n.entity_type == OntologyClass.AI_SYSTEM:
                vuln_parent = n
                break
        if not vuln_parent:
            vuln_parent = chain_entities.get("risk_source")
        if vuln_parent:
            edges.append(_make_edge(
                vuln_parent.id, RelationType.HAS_VULNERABILITY, vuln.id,
                vuln.evidence, ExtractionMode.ABSTRACTED, vuln.confidence,
                list(set(vuln_parent.source_doc_ids + vuln.source_doc_ids)),
                subject_name=vuln_parent.name, object_name=vuln.name,
            ))

    # --- RiskControl -> mitigates -> Risk ---
    rc_entity = chain_entities.get("risk_control")
    risk_entity = chain_entities.get("risk")
    if rc_entity and risk_entity:
        rc_doc_ids = list(set(risk_entity.source_doc_ids + rc_entity.source_doc_ids))
        edges.append(_make_edge(
            rc_entity.id, RelationType.MITIGATES, risk_entity.id,
            rc_entity.evidence, ExtractionMode.ABSTRACTED, rc_entity.confidence, rc_doc_ids,
            subject_name=rc_entity.name, object_name=risk_entity.name,
        ))


def _add_inference_nodes(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    _INFERENCE_TYPE_MAP = {
        "purpose": OntologyClass.PURPOSE,
        "lifecycle_phase": OntologyClass.AI_LIFECYCLE_PHASE,
        "lifecycle": OntologyClass.AI_LIFECYCLE_PHASE,
        "impact_domain": OntologyClass.DOMAIN,
        "domain": OntologyClass.DOMAIN,
    }

    # Edge predicates for connecting inference nodes
    _INFERENCE_EDGE_MAP = {
        "purpose": RelationType.HAS_CAPABILITY,  # AISystem -> hasCapability -> Purpose
        "lifecycle_phase": RelationType.USES_TECHNIQUE,  # Incident -> lifecycle
        "impact_domain": RelationType.AFFECTS,  # Incident -> domain
    }

    fallback_doc_id = _get_fallback_doc_id(state, event_id)
    inferences = state.get("inferences", {})

    # Find primary AISystem/GPAIModel for connecting Purpose
    ai_systems = [n for n in nodes if n.entity_type in (OntologyClass.AI_SYSTEM, OntologyClass.GPAI_MODEL)]
    primary_system = max(ai_systems, key=lambda s: s.support_count) if ai_systems else None

    for field_name, inf_data in inferences.items():
        if not inf_data.get("value"):
            continue
        etype = _INFERENCE_TYPE_MAP.get(field_name)
        if not etype:
            continue

        doc_id = _resolve_source_doc_id(inf_data, fallback_doc_id)
        confidence = float(inf_data.get("confidence", 0.7))

        node = EntityNode(
            id=generate_id(event_id, field_name, inf_data["value"][:20]),
            name=inf_data["value"],
            entity_type=etype,
            evidence=[EvidenceItem(
                evidence_id=generate_uuid(),
                evidence_sentence=inf_data.get("evidence", ""),
                source_doc_id=doc_id,
                confidence=confidence,
            )],
            extraction_mode=ExtractionMode.INFERRED,
            confidence=confidence,
            source_doc_ids=[doc_id],
            reasoning=inf_data.get("reasoning"),
        )
        nodes.append(node)

        # Connect to incident or AISystem
        if field_name == "purpose" and primary_system:
            edges.append(_make_edge(
                primary_system.id, RelationType.HAS_CAPABILITY, node.id,
                node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                subject_name=primary_system.name, object_name=node.name,
            ))
        else:
            # Connect to incident via generic edge
            edges.append(_make_edge(
                event_id, RelationType.HAS_EVIDENCE, node.id,
                node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                subject_name=incident_name, object_name=node.name,
            ))


def _build_knowledge_statements(edges: list[RelationEdge]) -> list[KnowledgeStatement]:
    statements: list[KnowledgeStatement] = []
    for edge in edges:
        for ev in edge.evidence:
            statements.append(KnowledgeStatement(
                id=generate_uuid(),
                subject_id=edge.subject_id,
                predicate=edge.predicate,
                object_id=edge.object_id,
                evidence=ev,
                extraction_mode=edge.extraction_mode,
                confidence=edge.confidence,
                support_count=edge.support_count,
            ))
    return statements


def _make_edge(
    subject_id: str,
    predicate: RelationType,
    object_id: str,
    evidence: list[EvidenceItem],
    mode: ExtractionMode,
    confidence: float,
    source_doc_ids: list[str],
    subject_name: str = "",
    object_name: str = "",
) -> RelationEdge:
    return RelationEdge(
        id=generate_id(subject_id, predicate.value, object_id),
        subject_id=subject_id,
        subject_name=subject_name,
        predicate=predicate,
        object_id=object_id,
        object_name=object_name,
        evidence=evidence[:3],
        extraction_mode=mode,
        confidence=round(confidence, 2),
        source_doc_ids=source_doc_ids[:5],
    )


def _add_governance_nodes(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    """Add governance, documentation, and trend nodes from Stage 4 L3 inference."""
    governance = state.get("governance", {})
    if not governance:
        return

    fallback_doc_id = _get_fallback_doc_id(state, event_id)
    ai_systems = [n for n in nodes if n.entity_type in (OntologyClass.AI_SYSTEM, OntologyClass.GPAI_MODEL)]

    _GOV_TYPE_MAP = {
        "obligation": OntologyClass.OBLIGATION,
        "compliance_requirement": OntologyClass.COMPLIANCE_REQUIREMENT,
        "enforcement_action": OntologyClass.ENFORCEMENT_ACTION,
        "code_of_conduct": OntologyClass.CODE_OF_CONDUCT,
        "trend": OntologyClass.TREND,
        "documentation": OntologyClass.DOCUMENTATION,
        "technical_documentation": OntologyClass.TECHNICAL_DOCUMENTATION,
        "verification_test": OntologyClass.VERIFICATION_TEST,
    }

    for field_name, etype in _GOV_TYPE_MAP.items():
        data = governance.get(field_name)
        if not data or not data.get("value"):
            continue

        doc_id = _resolve_source_doc_id(data, fallback_doc_id)
        confidence = float(data.get("confidence", 0.7))
        node = EntityNode(
            id=generate_id(event_id, field_name, data["value"][:30]),
            name=data["value"],
            entity_type=etype,
            evidence=[EvidenceItem(
                evidence_id=generate_uuid(),
                evidence_sentence=data.get("evidence", ""),
                source_doc_id=doc_id,
                confidence=confidence,
            )],
            extraction_mode=ExtractionMode.INFERRED,
            confidence=confidence,
            source_doc_ids=[doc_id],
            reasoning=data.get("reasoning"),
        )
        nodes.append(node)

        # EnforcementAction -> respondsToIncident -> AIRiskIncident
        if field_name == "enforcement_action":
            edges.append(_make_edge(
                node.id, RelationType.RESPONDS_TO_INCIDENT, event_id,
                node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                subject_name=node.name, object_name=incident_name,
            ))
        # Obligation -> imposed by Regulation (if present)
        elif field_name == "obligation":
            regulations = [n for n in nodes if n.entity_type == OntologyClass.REGULATION]
            if regulations:
                reg = regulations[0]
                edges.append(_make_edge(
                    reg.id, RelationType.IMPOSES_REQUIREMENT, node.id,
                    node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                    subject_name=reg.name, object_name=node.name,
                ))
        # TechnicalDocumentation / Documentation -> connected to AISystem
        elif field_name in ("documentation", "technical_documentation", "verification_test"):
            if ai_systems:
                primary = max(ai_systems, key=lambda s: s.support_count)
                edges.append(_make_edge(
                    primary.id, RelationType.HAS_EVIDENCE, node.id,
                    node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                    subject_name=primary.name, object_name=node.name,
                ))
            else:
                edges.append(_make_edge(
                    event_id, RelationType.HAS_EVIDENCE, node.id,
                    node.evidence, ExtractionMode.INFERRED, confidence, [doc_id],
                    subject_name=incident_name, object_name=node.name,
                ))


def _add_technical_role_edges(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    """Generate develops/provides/deploys/uses, enforces, investigates,
    compliesWithRegulation, and conformsToStandard edges."""
    ai_systems = [n for n in nodes if n.entity_type in (OntologyClass.AI_SYSTEM, OntologyClass.GPAI_MODEL)]
    ai_models = [n for n in nodes if n.entity_type == OntologyClass.AI_MODEL]
    ai_regulators = [n for n in nodes if n.entity_type == OntologyClass.REGULATOR]
    stakeholders = {n.name: n for n in nodes if n.entity_type in _STAKEHOLDER_TYPES}

    _ROLE_TO_RELATION: dict[str, tuple[RelationType, tuple[OntologyClass, ...]]] = {
        "AIDeveloper": (RelationType.DEVELOPS, (OntologyClass.AI_SYSTEM, OntologyClass.AI_MODEL, OntologyClass.GPAI_MODEL)),
        "AIProvider": (RelationType.PROVIDES, (OntologyClass.AI_SYSTEM, OntologyClass.AI_MODEL, OntologyClass.GPAI_MODEL)),
        "AIDeployer": (RelationType.DEPLOYS, (OntologyClass.AI_SYSTEM,)),
        "AIUser": (RelationType.USES, (OntologyClass.AI_SYSTEM,)),
    }

    # --- Role-based technical edges ---
    role_assignments = state.get("role_assignments", [])
    for ra in role_assignments:
        name = ra.get("stakeholder_name", "")
        role = ra.get("role", "")
        mapping = _ROLE_TO_RELATION.get(role)
        if not name or not mapping:
            continue

        relation, target_types = mapping
        snode = stakeholders.get(name)
        if not snode:
            for sname, snodeCandidate in stakeholders.items():
                if sname.lower() == name.lower():
                    snode = snodeCandidate
                    break
        if not snode:
            continue

        confidence = float(ra.get("confidence", 0.8))
        doc_id = ra.get("source_doc_id", "")

        targets = [n for n in (ai_systems + ai_models) if n.entity_type in target_types]
        for target in targets:
            evidence = [
                EvidenceItem(
                    evidence_id=generate_uuid(),
                    evidence_sentence=ra.get("evidence", ""),
                    source_doc_id=doc_id,
                    confidence=confidence,
                )
            ] if ra.get("evidence") else []
            edges.append(_make_edge(
                snode.id, relation, target.id,
                evidence, ExtractionMode.INFERRED, confidence,
                [doc_id] if doc_id else snode.source_doc_ids,
                subject_name=snode.name, object_name=target.name,
            ))

    # --- Regulator -> enforces -> Regulation ---
    for reg in [n for n in nodes if n.entity_type == OntologyClass.REGULATION]:
        for regulator in ai_regulators:
            reg_evidence = regulator.evidence[:1] if regulator.evidence else []
            edges.append(_make_edge(
                regulator.id, RelationType.ENFORCES, reg.id,
                reg_evidence, ExtractionMode.INFERRED, 0.7,
                regulator.source_doc_ids,
                subject_name=regulator.name, object_name=reg.name,
            ))

    # --- Regulator -> investigates -> AIRiskIncident ---
    for regulator in ai_regulators:
        reg_evidence = regulator.evidence[:1] if regulator.evidence else []
        edges.append(_make_edge(
            regulator.id, RelationType.INVESTIGATES, event_id,
            reg_evidence, ExtractionMode.INFERRED, 0.7,
            regulator.source_doc_ids,
            subject_name=regulator.name, object_name=incident_name,
        ))

    # --- AISystem -> compliesWithRegulation -> Regulation ---
    if ai_systems:
        primary_system = max(ai_systems, key=lambda s: s.support_count)
        for reg in [n for n in nodes if n.entity_type == OntologyClass.REGULATION]:
            edges.append(_make_edge(
                primary_system.id, RelationType.COMPLIES_WITH_REGULATION, reg.id,
                reg.evidence, ExtractionMode.INFERRED, 0.7,
                primary_system.source_doc_ids,
                subject_name=primary_system.name, object_name=reg.name,
            ))

    # --- AISystem -> conformsToStandard -> Standard ---
    if ai_systems:
        primary_system = max(ai_systems, key=lambda s: s.support_count)
        for std in [n for n in nodes if n.entity_type == OntologyClass.STANDARD]:
            edges.append(_make_edge(
                primary_system.id, RelationType.CONFORMS_TO_STANDARD, std.id,
                std.evidence, ExtractionMode.INFERRED, 0.7,
                primary_system.source_doc_ids,
                subject_name=primary_system.name, object_name=std.name,
            ))


def _add_report_nodes(
    state: PipelineState,
    event_id: str,
    incident_name: str,
    nodes: list[EntityNode],
    edges: list[RelationEdge],
) -> None:
    """Add NewsReport and InformationSource nodes to the graph.

    Each document becomes a NewsReport node, and the source (e.g., news outlet)
    becomes an InformationSource node with a HAS_INFORMATION_SOURCE edge.
    """
    documents = state.get("documents", [])

    # Group documents by source
    source_to_docs: dict[str, list] = {}
    for doc in documents:
        source = doc.source_name or "Unknown"
        source_to_docs.setdefault(source, []).append(doc)

    # Build InformationSource nodes
    source_nodes: dict[str, EntityNode] = {}
    for source_name, docs in source_to_docs.items():
        source_id = generate_id(event_id, "source", source_name[:30])
        source_node = EntityNode(
            id=source_id,
            name=source_name,
            entity_type=OntologyClass.INFORMATION_SOURCE,
            description=f"Information source: {source_name}",
            evidence=[],
            extraction_mode=ExtractionMode.EXPLICIT,
            confidence=0.9,
            source_doc_ids=[doc.doc_id for doc in docs],
            attributes={"report_count": len(docs)},
        )
        nodes.append(source_node)
        source_nodes[source_name] = source_node

    # Build NewsReport nodes and connect to incident + source
    for doc in documents:
        report_node = EntityNode(
            id=doc.doc_id,
            name=doc.title or doc.doc_id,
            entity_type=OntologyClass.NEWS_REPORT,
            description=doc.content[:200] if doc.content else None,
            evidence=[
                EvidenceItem(
                    evidence_id=generate_uuid(),
                    evidence_sentence=doc.title or "",
                    source_doc_id=doc.doc_id,
                    confidence=1.0,
                )
            ],
            extraction_mode=ExtractionMode.EXPLICIT,
            confidence=0.9,
            source_doc_ids=[doc.doc_id],
            attributes={
                "url": doc.url,
                "publish_time": doc.publish_time,
                "language": doc.language,
            },
        )
        nodes.append(report_node)

        # Connect report to incident: incident -hasReport-> report
        edges.append(_make_edge(
            event_id,
            RelationType.HAS_REPORT,
            doc.doc_id,
            report_node.evidence,
            ExtractionMode.EXPLICIT,
            0.9,
            [doc.doc_id],
            subject_name=incident_name,
            object_name=report_node.name,
        ))

        # Connect report to source: source -hasInformationSource-> report
        source = doc.source_name or "Unknown"
        source_node = source_nodes.get(source)
        if source_node:
            edges.append(_make_edge(
                source_node.id,
                RelationType.HAS_INFORMATION_SOURCE,
                doc.doc_id,
                [],
                ExtractionMode.EXPLICIT,
                0.9,
                [doc.doc_id],
                subject_name=source_node.name,
                object_name=report_node.name,
            ))

    logger.info(
        f"[Stage 5] Added {len(documents)} NewsReport nodes and "
        f"{len(source_nodes)} InformationSource nodes"
    )


def _apply_system_attributes(
    state: PipelineState,
    nodes: list[EntityNode],
) -> None:
    """Merge inferred system attributes into AISystem nodes' attributes dict."""
    system_attrs = state.get("system_attributes", {})
    if not system_attrs:
        return

    ai_systems = [n for n in nodes if n.entity_type == OntologyClass.AI_SYSTEM]
    if not ai_systems:
        return

    primary = max(ai_systems, key=lambda s: s.support_count)
    for attr_name, attr_data in system_attrs.items():
        if attr_name not in primary.attributes or not primary.attributes[attr_name]:
            primary.attributes[attr_name] = attr_data.get("value", "")
            logger.info(f"[Stage 5] Added attribute {attr_name}={attr_data.get('value', '')} to {primary.name}")
