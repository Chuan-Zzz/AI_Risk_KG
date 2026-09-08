"""Neo4j graph storage for event knowledge subgraphs."""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import get_config
from src.core.models import EntityNode, EventKnowledgeSubgraph, ExtractionMode, OntologyClass, RelationType

logger = logging.getLogger(__name__)

_LABEL_MAP: dict[str, str] = {
    "AIRiskIncident": "AIRiskIncident",
    "AISystem": "AISystem",
    "AIModel": "AIModel",
    "GPAIModel": "GPAIModel",
    "AITechnique": "AITechnique",
    "AICapability": "AICapability",
    "AIComponent": "AIComponent",
    "Data": "Data",
    "Stakeholder": "Stakeholder",
    "AIDeveloper": "AIDeveloper",
    "AIProvider": "AIProvider",
    "AIDeployer": "AIDeployer",
    "AIUser": "AIUser",
    "Regulator": "Regulator",
    "AffectedActor": "AffectedActor",
    "RiskSource": "RiskSource",
    "Risk": "Risk",
    "Hazard": "Hazard",
    "Threat": "Threat",
    "Vulnerability": "Vulnerability",
    "Consequence": "Consequence",
    "Impact": "Impact",
    "RiskControl": "RiskControl",
    "Regulation": "Regulation",
    "Standard": "Standard",
    "Obligation": "Obligation",
    "ComplianceRequirement": "ComplianceRequirement",
    "EnforcementAction": "EnforcementAction",
    "CodeOfConduct": "CodeOfConduct",
    "NewsReport": "NewsReport",
    "InformationSource": "InformationSource",
    "Evidence": "Evidence",
    "KnowledgeStatement": "KnowledgeStatement",
    "RoleAssignment": "RoleAssignment",
    "Purpose": "Purpose",
    "AILifecyclePhase": "AILifecyclePhase",
    "Domain": "Domain",
    "Trend": "Trend",
    "Documentation": "Documentation",
    "TechnicalDocumentation": "TechnicalDocumentation",
    "VerificationTest": "VerificationTest",
}

_VALID_REL_TYPES = frozenset(rt.value.upper() for rt in RelationType)

# Entity types that represent real-world entities and can be genuinely shared
# across events. Only these types create cross-event connections via name-based MERGE.
_ALIGNABLE_TYPES = frozenset({
    OntologyClass.AI_SYSTEM,
    OntologyClass.AI_MODEL,
    OntologyClass.AI_TECHNIQUE,
    OntologyClass.AI_CAPABILITY,
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
    OntologyClass.AFFECTED_ACTOR,
    OntologyClass.REGULATION,
    OntologyClass.STANDARD,
})

# Stakeholder subtypes normalize to "Stakeholder" label for cross-event merging.
# This ensures "Google" as AIProvider and "Google" as AIDeployer resolve to one node.
_STAKEHOLDER_TYPES = frozenset({
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
    OntologyClass.AFFECTED_ACTOR,
})


def _safe_label(entity_type: OntologyClass) -> str:
    label = _LABEL_MAP.get(entity_type.value)
    if label is None:
        label = "Entity"
    return label.replace("`", "``")


def _safe_rel_type(predicate: RelationType) -> str:
    val = predicate.value.upper()
    if val not in _VALID_REL_TYPES:
        raise ValueError(f"Unknown relation type: {val}")
    return val.replace("`", "``")


def _is_alignable(entity_type: OntologyClass) -> bool:
    return entity_type in _ALIGNABLE_TYPES


def _merge_label(entity_type: OntologyClass) -> str:
    """Neo4j label for MERGE — stakeholder subtypes normalize to Stakeholder."""
    if entity_type in _STAKEHOLDER_TYPES:
        return "Stakeholder"
    return _safe_label(entity_type)


def _build_node_match(alias: str, node: EntityNode, event_id: str) -> tuple[str, dict[str, Any]]:
    """Build a Cypher MATCH clause and parameters for a node.

    - AIRiskIncident  → MATCH by {event_id}
    - Alignable types → MATCH by {name} with type label
    - Non-alignable   → MATCH by {id} (unique per event)
    """
    p = f"{alias}_"
    if node.entity_type == OntologyClass.AI_RISK_INCIDENT:
        return (
            f"({alias}:AIRiskIncident {{event_id: ${p}event_id}})",
            {f"{p}event_id": event_id},
        )
    if _is_alignable(node.entity_type):
        ml = _merge_label(node.entity_type)
        return (
            f"({alias}:`{ml}` {{name: ${p}name}})",
            {f"{p}name": node.name},
        )
    label = _safe_label(node.entity_type)
    return (
        f"({alias}:`{label}` {{id: ${p}id}})",
        {f"{p}id": node.id},
    )


class Neo4jStore:
    def __init__(self) -> None:
        config = get_config()
        self._uri = config.get("neo4j.uri", "")
        self._user = config.get("neo4j.user", "")
        self._password = config.get("neo4j.password", "")
        self._database = config.get("neo4j.database", "neo4j")
        self._driver = None
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            if not self._uri:
                return
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._driver.verify_connectivity()
            logger.info("Neo4j connected")
        except Exception as e:
            logger.warning("Neo4j connection failed: %s", type(e).__name__)
            self._driver = None

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self):
        return self._driver

    def connect(self) -> bool:
        if self._driver is None:
            self._connect()
        return self._driver is not None

    def clear_database(self) -> None:
        """Remove all nodes and relationships. Call before re-running with new alignment."""
        if not self._driver:
            return
        with self._driver.session(database=self._database) as session:
            # Delete in batches to avoid OOM on large graphs
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n "
                    "RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
        logger.info("Neo4j: database cleared")

    def get_stats(self) -> dict[str, Any]:
        if not self._driver:
            return {
                "total_nodes": 0,
                "total_relations": 0,
                "nodes_by_type": {},
            }

        with self._driver.session(database=self._database) as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            total_relations = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            by_type_result = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY c DESC"
            )
            nodes_by_type = {record["label"]: record["c"] for record in by_type_result}

        return {
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "nodes_by_type": nodes_by_type,
        }

    def add_event_subgraph(self, subgraph: EventKnowledgeSubgraph) -> None:
        if not self._driver:
            return

        node_lookup: dict[str, EntityNode] = {n.id: n for n in subgraph.nodes}

        with self._driver.session(database=self._database) as session:
            # --- Nodes ---
            for node in subgraph.nodes:
                label = _safe_label(node.entity_type)
                evidence_texts = [ev.evidence_sentence for ev in node.evidence[:3]]
                evidence_doc_ids = [ev.source_doc_id for ev in node.evidence[:3]]

                if node.entity_type == OntologyClass.AI_RISK_INCIDENT:
                    # Incident: unique per event — MERGE on event_id
                    session.run(
                        f"MERGE (n:`{label}` {{event_id: $event_id}}) "
                        f"SET n.name = $name, n.id = $id, n.entity_type = $entity_type, "
                        f"n.extraction_mode = $mode, n.confidence = $conf, "
                        f"n.evidence = $evidence, n.source_doc_ids = $source_doc_ids, "
                        f"n.evidence_doc_ids = $evidence_doc_ids",
                        id=node.id,
                        name=node.name,
                        entity_type=node.entity_type.value,
                        mode=node.extraction_mode.value,
                        conf=node.confidence,
                        evidence=evidence_texts,
                        source_doc_ids=node.source_doc_ids,
                        evidence_doc_ids=evidence_doc_ids,
                        event_id=subgraph.event_id,
                    )

                elif _is_alignable(node.entity_type):
                    # Alignable: cross-event shared entity — MERGE on name within type.
                    # evidence / source_doc_ids / evidence_doc_ids are accumulated across
                    # events rather than overwritten, so shared entities preserve the
                    # full provenance from every event that mentions them.
                    ml = _merge_label(node.entity_type)
                    session.run(
                        f"MERGE (n:`{ml}` {{name: $name}}) "
                        f"ON CREATE SET n.evidence = [], n.source_doc_ids = [], "
                        f"n.evidence_doc_ids = [], n.event_ids = [] "
                        f"SET n.id = $id, n.entity_type = $entity_type, "
                        f"n.extraction_mode = $mode, "
                        f"n.confidence = CASE WHEN $conf > coalesce(n.confidence, 0) "
                        f"THEN $conf ELSE n.confidence END, "
                        f"n.evidence = [x IN n.evidence WHERE NOT x IN $evidence] + $evidence, "
                        f"n.source_doc_ids = [x IN n.source_doc_ids WHERE NOT x IN $source_doc_ids] + $source_doc_ids, "
                        f"n.evidence_doc_ids = [x IN n.evidence_doc_ids WHERE NOT x IN $evidence_doc_ids] + $evidence_doc_ids",
                        id=node.id,
                        name=node.name,
                        entity_type=node.entity_type.value,
                        mode=node.extraction_mode.value,
                        conf=node.confidence,
                        evidence=evidence_texts,
                        source_doc_ids=node.source_doc_ids,
                        evidence_doc_ids=evidence_doc_ids,
                    )
                    # Accumulate event_ids on shared entity
                    session.run(
                        f"MATCH (n:`{ml}` {{name: $name}}) "
                        f"SET n.event_ids = CASE WHEN n.event_ids IS NULL "
                        f"THEN [$event_id] "
                        f"ELSE CASE WHEN $event_id IN n.event_ids "
                        f"THEN n.event_ids ELSE n.event_ids + $event_id END END",
                        name=node.name,
                        event_id=subgraph.event_id,
                    )

                else:
                    # Non-alignable: event-specific — MERGE on id (unique per event)
                    session.run(
                        f"MERGE (n:`{label}` {{id: $id}}) "
                        f"SET n.name = $name, n.entity_type = $entity_type, "
                        f"n.extraction_mode = $mode, n.confidence = $conf, "
                        f"n.evidence = $evidence, n.source_doc_ids = $source_doc_ids, "
                        f"n.evidence_doc_ids = $evidence_doc_ids, "
                        f"n.event_id = $event_id",
                        id=node.id,
                        name=node.name,
                        entity_type=node.entity_type.value,
                        mode=node.extraction_mode.value,
                        conf=node.confidence,
                        evidence=evidence_texts,
                        source_doc_ids=node.source_doc_ids,
                        evidence_doc_ids=evidence_doc_ids,
                        event_id=subgraph.event_id,
                    )

            # --- Edges ---
            for edge in subgraph.edges:
                rel_type = _safe_rel_type(edge.predicate)
                evidence_texts = [ev.evidence_sentence for ev in edge.evidence[:2]]
                evidence_doc_ids = [ev.source_doc_id for ev in edge.evidence[:2]]

                subj_node = node_lookup.get(edge.subject_id)
                obj_node = node_lookup.get(edge.object_id)

                if not subj_node or not obj_node:
                    logger.warning("Skipping edge %s: could not resolve nodes", rel_type)
                    continue

                subj_clause, subj_params = _build_node_match("a", subj_node, subgraph.event_id)
                obj_clause, obj_params = _build_node_match("b", obj_node, subgraph.event_id)

                params: dict[str, Any] = {
                    **subj_params,
                    **obj_params,
                    "mode": edge.extraction_mode.value,
                    "conf": edge.confidence,
                    "evidence": evidence_texts,
                    "source_doc_ids": edge.source_doc_ids,
                    "evidence_doc_ids": evidence_doc_ids,
                }

                session.run(
                    f"MATCH {subj_clause}, {obj_clause} "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) "
                    f"ON CREATE SET r.evidence = [], r.source_doc_ids = [], "
                    f"r.evidence_doc_ids = [] "
                    f"SET r.extraction_mode = $mode, "
                    f"r.confidence = CASE WHEN $conf > coalesce(r.confidence, 0) "
                    f"THEN $conf ELSE r.confidence END, "
                    f"r.evidence = [x IN r.evidence WHERE NOT x IN $evidence] + $evidence, "
                    f"r.source_doc_ids = [x IN r.source_doc_ids WHERE NOT x IN $source_doc_ids] + $source_doc_ids, "
                    f"r.evidence_doc_ids = [x IN r.evidence_doc_ids WHERE NOT x IN $evidence_doc_ids] + $evidence_doc_ids",
                    **params,
                )

        logger.info("Neo4j: stored %d nodes, %d edges", len(subgraph.nodes), len(subgraph.edges))
