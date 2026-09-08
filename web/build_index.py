import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
TARGET = Path(__file__).resolve().parent / "data" / "events.json"

events = []
types = Counter()
chains = Counter()
documents = set()

for path in sorted(OUTPUT.glob("pred_*/event_subgraph.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    incident = data.get("incident_node") or {}
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    for node in nodes:
        entity_type = node.get("entity_type") or "Unknown"
        types[entity_type] += 1
        documents.update(node.get("source_doc_ids") or [])
    edge_types = {str(edge.get("relation_type") or edge.get("predicate") or "").lower() for edge in edges}
    if "hasrisk" in edge_types or "causes" in edge_types:
        chains["risk"] += 1
    if "leadsto" in edge_types or "leads_to" in edge_types:
        chains["consequence"] += 1
    if "impacts" in edge_types or "hasimpact" in edge_types:
        chains["impact"] += 1
    if "mitigates" in edge_types:
        chains["control"] += 1
    events.append({
        "id": data.get("event_id") or path.parent.name,
        "title": incident.get("name") or data.get("event_id") or path.parent.name,
        "date": data.get("first_seen") or "",
        "reports": data.get("report_count") or len(incident.get("evidence") or []),
        "nodes": len(nodes),
        "edges": len(edges),
        "risk": next((node.get("name") for node in nodes if node.get("entity_type") == "Risk"), "Risk chain"),
        "riskType": next((node.get("entity_type") for node in nodes if node.get("entity_type") in {"Risk", "RiskSource", "Impact", "Hazard"}), "Risk"),
    })

payload = {"events": events, "stats": {"events": len(events), "nodes": sum(x["nodes"] for x in events), "edges": sum(x["edges"] for x in events), "documents": len(documents), "types": dict(types), "chains": dict(chains)}}
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {len(events)} events to {TARGET}")
