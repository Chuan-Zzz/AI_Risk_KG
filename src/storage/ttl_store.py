"""TTL/RDF output for event knowledge subgraphs using rdflib."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from src.core.models import EventKnowledgeSubgraph, ExtractionMode, OntologyClass

AIRO = Namespace("https://w3id.org/airo#")
BASE = Namespace("https://w3id.org/airo/incident/")


def write_ttl(subgraph: EventKnowledgeSubgraph, path: Path) -> None:
    g = Graph()
    g.bind("airo", AIRO)
    g.bind("incident", BASE)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    event_uri = BASE[f"event/{subgraph.event_id}"]
    g.add((event_uri, RDF.type, AIRO.AIRiskIncident))

    # Add all nodes with evidence provenance
    for node in subgraph.nodes:
        node_uri = _make_uri(node.entity_type, node.id)
        class_uri = AIRO[node.entity_type.value]
        g.add((node_uri, RDF.type, class_uri))
        g.add((node_uri, RDFS.label, Literal(node.name)))

        if node.description:
            g.add((node_uri, AIRO["hasDescription"], Literal(node.description)))

        if node.extraction_mode:
            g.add((node_uri, AIRO.extractionMode, Literal(node.extraction_mode.value)))

        if node.confidence > 0:
            g.add((node_uri, AIRO.confidenceScore, Literal(node.confidence, datatype=XSD.float)))

        if node.support_count > 1:
            g.add((node_uri, AIRO.hasSupportCount, Literal(node.support_count, datatype=XSD.integer)))

        if node.reasoning:
            g.add((node_uri, AIRO["hasReasoning"], Literal(node.reasoning)))

        # Write source_doc_ids as hasSourceDocument relations
        for doc_id in node.source_doc_ids:
            doc_uri = BASE[f"document/{doc_id}"]
            g.add((node_uri, AIRO.hasSourceDocument, doc_uri))

        # Write individual evidence items with source_doc_id
        for ev in node.evidence:
            ev_uri = BASE[f"evidence/{ev.evidence_id}"]
            g.add((ev_uri, RDF.type, AIRO.Evidence))
            g.add((ev_uri, AIRO.evidenceText, Literal(ev.evidence_sentence)))
            g.add((ev_uri, AIRO["sourceDocId"], Literal(ev.source_doc_id)))
            g.add((node_uri, AIRO.hasEvidence, ev_uri))

    # Add all edges
    for edge in subgraph.edges:
        subj_uri = _find_uri(g, edge.subject_id, subgraph)
        obj_uri = _find_uri(g, edge.object_id, subgraph)
        if subj_uri and obj_uri:
            pred_uri = AIRO[edge.predicate.value]
            g.add((subj_uri, pred_uri, obj_uri))

    # Add knowledge statements with evidence
    for ks in subgraph.knowledge_statements:
        ks_uri = BASE[f"ks/{ks.id}"]
        g.add((ks_uri, RDF.type, AIRO.KnowledgeStatement))
        g.add((ks_uri, AIRO.extractionMode, Literal(ks.extraction_mode.value)))
        g.add((ks_uri, AIRO.confidenceScore, Literal(ks.confidence, datatype=XSD.float)))

        ev_uri = BASE[f"evidence/{ks.evidence.evidence_id}"]
        g.add((ev_uri, RDF.type, AIRO.Evidence))
        g.add((ev_uri, AIRO.evidenceText, Literal(ks.evidence.evidence_sentence)))
        g.add((ev_uri, AIRO["sourceDocId"], Literal(ks.evidence.source_doc_id)))
        g.add((ks_uri, AIRO.hasEvidence, ev_uri))

    # Add event metadata
    if subgraph.first_seen:
        g.add((event_uri, AIRO.firstSeen, Literal(subgraph.first_seen, datatype=XSD.dateTime)))
    if subgraph.last_seen:
        g.add((event_uri, AIRO.lastSeen, Literal(subgraph.last_seen, datatype=XSD.dateTime)))
    g.add((event_uri, AIRO.hasSupportCount, Literal(subgraph.report_count, datatype=XSD.integer)))

    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")


def _make_uri(entity_type: OntologyClass, entity_id: str) -> URIRef:
    safe_id = entity_id.replace(" ", "_").replace("/", "_")
    return BASE[f"{entity_type.value}/{safe_id}"]


def _find_uri(g: Graph, node_id: str, subgraph: EventKnowledgeSubgraph) -> URIRef | None:
    for node in subgraph.nodes:
        if node.id == node_id:
            return _make_uri(node.entity_type, node.id)
    return None
