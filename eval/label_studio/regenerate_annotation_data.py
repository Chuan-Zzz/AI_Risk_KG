"""Regenerate Label Studio annotation_tasks.json from pipeline output.

Reads event_subgraph.json + gold_standard_100_events.json + eval_cases.jsonl
and produces annotation data for level1_entity, level2_risk_chain, level3_inference.

Changes from previous version:
  - Level 1: only show the document with the most entity mentions (instead of all docs)
  - All levels: no Chinese translation interleaving (English only)
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from collections import defaultdict
from html import escape as h, unescape as unescape_html
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path("output")
GOLD_FILE = Path("data/gold_standard_100_events.json")
CASES_FILE = Path("data/eval_cases.jsonl")
LEVEL1_DIR = Path("eval/label_studio/level1_entity")
LEVEL2_DIR = Path("eval/label_studio/level2_risk_chain")
LEVEL3_DIR = Path("eval/label_studio/level3_inference")

# ── Entity type display info ──────────────────────────────────────────────

ENTITY_TYPE_META: dict[str, tuple[str, str]] = {
    # (English label, border color)
    "AISystem": ("AISystem", "#ff6b6b"),
    "AIModel": ("AIModel", "#ffa94d"),
    "GPAIModel": ("GPAIModel", "#ffa94d"),
    "AITechnique": ("AITechnique", "#ffd43b"),
    "AICapability": ("AICapability", "#ff8787"),
    "AIDeveloper": ("AIDeveloper", "#20c997"),
    "AIProvider": ("AIProvider", "#339af0"),
    "AIDeployer": ("AIDeployer", "#51cf66"),
    "AIUser": ("AIUser", "#38d9a9"),
    "Regulator": ("Regulator", "#e64980"),
    "AffectedActor": ("AffectedActor", "#f06595"),
    "Stakeholder": ("Stakeholder", "#9775fa"),
    "Regulation": ("Regulation", "#364fc7"),
    "Standard": ("Standard", "#4263eb"),
    "RiskSource": ("RiskSource", "#e67700"),
    "Risk": ("Risk", "#c92a2a"),
    "Misuse": ("Misuse", "#a51c30"),
    "Hazard": ("Hazard", "#b5301a"),
    "Threat": ("Threat", "#8b2500"),
    "Consequence": ("Consequence", "#2d6a4f"),
    "Impact": ("Impact", "#1b4332"),
    "RiskControl": ("RiskControl", "#0c5460"),
    "Vulnerability": ("Vulnerability", "#6f42c1"),
    "Obligation": ("Obligation", "#495057"),
    "InformationSource": ("InformationSource", "#868e96"),
}

# Risk chain slot order (for level2)
RISK_CHAIN_SLOTS = [
    ("risk_source_pre", "RiskSource", "RiskSource", "#fff3cd", "#856404"),
    ("risk_pre", "Risk", "Risk", "#f8d7da", "#721c24"),
    ("consequence_pre", "Consequence", "Consequence", "#d4edda", "#155724"),
    ("impact_pre", "Impact", "Impact", "#cce5ff", "#004085"),
    ("affected_actor_pre", "AffectedActor", "AffectedActor", "#e2e3e5", "#383d41"),
    ("risk_control_pre", "RiskControl", "RiskControl", "#d1ecf1", "#0c5460"),
]


def load_gold_events() -> list[dict]:
    with open(GOLD_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gold_standard", [])


def load_cases() -> dict[str, dict]:
    cases = {}
    with open(CASES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("id") or obj.get("original_id", ""))
            if cid:
                cases[cid] = obj
    return cases


def load_subgraph(event_id: str) -> dict | None:
    p = OUTPUT_DIR / event_id / "event_subgraph.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Text assembly (EN only, no Chinese) ───────────────────────────────────

def _format_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return h(text)
    return "\n".join(f"<p>{h(p)}</p>" for p in paragraphs)


def build_text_en_only(doc_ids: list[str], cases: dict[str, dict]) -> tuple[str, dict[str, int]]:
    """Build combined text for an event, English only (no CN interleaving).

    Returns (html, doc_offsets) where doc_offsets maps doc_id -> start offset of body text.
    """
    result = ""
    offsets = {}
    current_pos = 0
    for did in doc_ids:
        c = cases.get(did)
        if not c:
            continue
        title = c.get("title", f"Doc {did}")
        raw_text = c.get("text", c.get("new_summary", c.get("description", "")))

        header = (
            f'<div style="margin:16px 0 8px 0;padding:10px 14px;'
            f'background:#2c3e50;color:#fff;border-radius:8px;'
            f'font-weight:bold;font-size:15px;">'
            f'Document {did}: {title}</div>'
        )
        if result:
            current_pos += 1
        result += header
        current_pos += len(header)

        body = (
            f'<div style="padding:10px 14px;background:#fff;border:1px solid #e0e0e0;'
            f'border-radius:4px;margin-bottom:8px;line-height:1.8;font-size:14px;">'
            f'{_format_paragraphs(raw_text)}</div>'
        )
        offsets[did] = current_pos
        result += body
        current_pos += len(body)

    return result, offsets


# ── Level 1: Entity pre-extraction HTML ──────────────────────────────────

def _annotate_text(text: str, mentions: list[tuple]) -> str:
    parts = []
    prev_end = 0
    for start, end, mtext, etype, conf in mentions:
        if start < prev_end or end > len(text):
            continue
        parts.append(h(text[prev_end:start]))
        label = ENTITY_TYPE_META.get(etype, (etype, "#999"))[0]
        bg = ENTITY_TYPE_META.get(etype, (etype, "#999"))[1]
        annotated = h(text[start:end])
        tag = (
            f'<span style="background:{bg}22;border-bottom:2px solid {bg};'
            f'padding:1px 2px;border-radius:2px;cursor:default" title="{h(etype)}">'
            f'{annotated}'
            f'<span style="background:{bg};color:#fff;padding:1px 5px;'
            f'border-radius:8px;font-size:10px;margin-left:2px;'
            f'vertical-align:middle;font-weight:600">{conf:.02f}</span>'
            f'<span style="background:{bg}33;color:{bg};padding:1px 5px;'
            f'border-radius:8px;font-size:10px;margin-left:2px;'
            f'vertical-align:middle;font-weight:600">{label}</span>'
            f'</span>'
        )
        parts.append(tag)
        prev_end = end
    parts.append(h(text[prev_end:]))
    return "".join(parts)


def build_entity_html_single_doc(subgraph: dict, doc_id: str, cases: dict[str, dict]) -> str:
    """Build annotated HTML for a single document (the one with most entity mentions)."""
    nodes = subgraph.get("nodes", [])
    c = cases.get(doc_id)
    if not c:
        return ""

    title = c.get("title", f"Doc {doc_id}")
    raw_text = c.get("text", c.get("new_summary", c.get("description", "")))

    # Collect mentions for this doc
    doc_mentions: list[tuple] = []
    for n in nodes:
        etype = n.get("entity_type", "")
        if etype not in ENTITY_TYPE_META:
            continue
        for ev in n.get("evidence", []):
            ms = ev.get("mention_span")
            if not ms or not isinstance(ms, dict):
                continue
            start = ms.get("start", -1)
            end = ms.get("end", -1)
            mtext = ms.get("text", "")
            ev_doc_id = str(ev.get("source_doc_id", ""))
            if ev_doc_id != doc_id:
                continue
            if start < 0 or end <= start or not mtext:
                continue
            conf = ev.get("confidence", 0.8)
            doc_mentions.append((start, end, mtext, etype, conf))

    doc_mentions.sort(key=lambda x: x[0])
    annotated = _annotate_text(raw_text, doc_mentions)

    parts = [
        f'<div style="margin:16px 0 8px 0;padding:10px 14px;'
        f'background:#2c3e50;color:#fff;border-radius:8px;'
        f'font-weight:bold;font-size:15px;">'
        f'Document {doc_id}: {h(title)}</div>',
        f'<div style="padding:10px 14px;background:#fff;border:1px solid #e0e0e0;'
        f'border-radius:4px;margin-bottom:8px;line-height:1.8;font-size:14px;">'
        f'{annotated}</div>',
    ]

    return (
        '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.6">'
        + "".join(parts)
        + "</div>"
    )


def _rendered_text_from_html(html_text: str) -> str:
    no_tags = re.sub(r'<[^>]+>', '', html_text)
    return unescape_html(no_tags)


def _count_entity_mentions_for_doc(subgraph: dict, doc_id: str) -> int:
    """Count entity mentions in a specific document."""
    nodes = subgraph.get("nodes", [])
    count = 0
    for n in nodes:
        if n.get("entity_type", "") not in ENTITY_TYPE_META:
            continue
        for ev in n.get("evidence", []):
            ms = ev.get("mention_span")
            if not ms or not isinstance(ms, dict):
                continue
            if str(ev.get("source_doc_id", "")) == doc_id:
                count += 1
    return count


def build_level1_task(subgraph: dict, doc_ids: list[str], cases: dict[str, dict],
                      meta_info: str) -> dict:
    """Build Level 1 annotation task.

    Only shows the document with the most entity mentions (single doc view).
    """
    # Pick the doc with most entity mentions
    best_doc = None
    best_count = -1
    for did in doc_ids:
        count = _count_entity_mentions_for_doc(subgraph, did)
        if count > best_count:
            best_count = count
            best_doc = did

    if best_doc is None:
        best_doc = doc_ids[0] if doc_ids else ""

    # Build entity HTML for single doc
    html_text = build_entity_html_single_doc(subgraph, best_doc, cases)

    # Build predictions
    nodes = subgraph.get("nodes", [])

    # Build stakeholder -> role mapping
    stakeholder_roles: dict[str, list[str]] = {}
    for n in nodes:
        if n.get("entity_type") != "RoleAssignment":
            continue
        ra_name = n.get("name", "")
        for suffix in ["_AIDeveloper", "_AIProvider", "_AIDeployer", "_AIUser",
                       "_Regulator", "_AffectedActor", "_Stakeholder"]:
            if ra_name.endswith(suffix):
                stakeholder_name = ra_name[: -len(suffix)]
                role_type = suffix[1:]
                if stakeholder_name.lower() not in stakeholder_roles:
                    stakeholder_roles[stakeholder_name.lower()] = []
                stakeholder_roles[stakeholder_name.lower()].append(role_type)
                break

    # Collect mentions from the best doc only
    all_mentions: list[dict] = []
    for n in nodes:
        etype = n.get("entity_type", "")
        if etype not in ENTITY_TYPE_META:
            continue
        node_name = n.get("name", "")
        display_types = [etype]
        if etype == "Stakeholder" and node_name.lower() in stakeholder_roles:
            display_types = stakeholder_roles[node_name.lower()]

        for ev in n.get("evidence", []):
            ms = ev.get("mention_span")
            if not ms or not isinstance(ms, dict):
                continue
            start = ms.get("start", -1)
            end = ms.get("end", -1)
            mtext = ms.get("text", "")
            ev_doc_id = str(ev.get("source_doc_id", ""))
            conf = ev.get("confidence", 0.8)
            if ev_doc_id != best_doc:
                continue
            if start < 0 or end <= start or not mtext:
                continue
            for dt in display_types:
                all_mentions.append({
                    "text": mtext,
                    "entity_type": dt,
                    "doc_id": ev_doc_id,
                    "confidence": conf,
                })

    type_to_label = {
        "AISystem": "AISystem", "AIModel": "AIModel", "GPAIModel": "GPAIModel",
        "AITechnique": "AITechnique", "AICapability": "AICapability",
        "AIDeveloper": "AIDeveloper", "AIProvider": "AIProvider",
        "AIDeployer": "AIDeployer", "AIUser": "AIUser",
        "Regulator": "Regulator", "AffectedActor": "AffectedActor",
        "Stakeholder": "Stakeholder", "Regulation": "Regulation", "Standard": "Standard",
    }

    # Dedup mentions
    mention_best: dict[tuple, dict] = {}
    for m in all_mentions:
        key = (m["doc_id"], m["text"])
        if key not in mention_best:
            mention_best[key] = {"labels": set(), "confidence": 0.0}
        mention_best[key]["labels"].add(m["entity_type"])
        mention_best[key]["confidence"] = max(mention_best[key]["confidence"], m["confidence"])

    # Build predictions using rendered text offsets
    rendered_text = _rendered_text_from_html(html_text)
    result_id = str(uuid.uuid4())

    predictions = []
    for dedup_key, m_info in mention_best.items():
        text = dedup_key[1]
        header = f"Document {best_doc}:"
        hpos = rendered_text.find(header)
        if hpos < 0:
            continue
        abs_pos = rendered_text.find(text, hpos)
        if abs_pos < 0:
            continue
        labels = [type_to_label.get(lt, lt) for lt in m_info["labels"]]
        predictions.append({
            "id": str(uuid.uuid4())[:8],
            "result_id": result_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": abs_pos,
                "end": abs_pos + len(text),
                "text": text,
                "labels": labels,
            },
        })

    # Update meta_info to reflect single-doc
    meta_info = meta_info.replace(f"文档数: {len(doc_ids)}", f"文档数: 1 (of {len(doc_ids)})")

    return {
        "data": {
            "meta_info": meta_info,
            "event_id": subgraph.get("event_id", ""),
            "text": html_text,
        },
        "predictions": [{"result": predictions}],
    }


# ── Level 2: Risk chain HTML (EN only) ───────────────────────────────────

SLOT_COLORS: dict[str, str] = {
    "RiskSource": "#856404",
    "Risk": "#c92a2a",
    "Consequence": "#155724",
    "Impact": "#004085",
    "AffectedActor": "#383d41",
    "RiskControl": "#0c5460",
}

SLOT_BG_COLORS: dict[str, str] = {
    "RiskSource": "#fff3cd",
    "Risk": "#f8d7da",
    "Consequence": "#d4edda",
    "Impact": "#cce5ff",
    "AffectedActor": "#e2e3e5",
    "RiskControl": "#d1ecf1",
}


def _is_chinese(text: str) -> bool:
    """Check if text is predominantly Chinese."""
    if not text:
        return False
    sample = text[:300]
    cn_chars = sum(1 for c in sample if '一' <= c <= '鿿')
    return cn_chars > len(sample) * 0.15


def _strip_chinese(text: str) -> str:
    """Return empty string if text contains Chinese, else return as-is."""
    if not text:
        return ""
    if _is_chinese(text):
        return ""
    return text


def _slot_html(value: str, evidence: str) -> str:
    if not value:
        return '<div style="color:#999;font-style:italic;">Not extracted</div>'
    return (
        f'<div style="padding:8px;background:#fff;border-radius:4px;">'
        f'<div style="font-weight:600;color:#333;margin-bottom:6px;">{h(value)}</div>'
        f'<div style="font-size:12px;color:#666;padding:6px;background:#f9f9f9;'
        f'border-left:3px solid #ddd;border-radius:2px;">'
        f'<span style="color:#999;">Evidence:</span> {h(evidence)}'
        f'</div></div>'
    )


def _collect_risk_evidence(subgraph: dict) -> dict[str, list[dict[str, str]]]:
    nodes = subgraph.get("nodes", [])
    risk_types = {"RiskSource", "Risk", "Consequence", "Impact",
                  "AffectedActor", "RiskControl"}

    type_nodes: dict[str, dict] = {}
    for n in nodes:
        etype = n.get("entity_type", "")
        if etype in risk_types and etype not in type_nodes:
            type_nodes[etype] = n

    doc_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for etype, n in type_nodes.items():
        value = n.get("name", "")
        best_ev = None
        best_conf = -1
        for ev in n.get("evidence", []):
            sentence = ev.get("evidence_sentence", "")
            doc_id = str(ev.get("source_doc_id", ""))
            if not sentence or not doc_id:
                continue
            conf = ev.get("confidence", 0)
            if conf > best_conf:
                best_conf = conf
                best_ev = ev
        if best_ev is None:
            continue
        sentence = best_ev.get("evidence_sentence", "")
        doc_id = str(best_ev.get("source_doc_id", ""))
        sentence = re.sub(r'^\s*\[\d+\]\s*', '', sentence)
        if sentence:
            doc_evidence[doc_id].append({
                "sentence": sentence,
                "slot_type": etype,
                "value": value,
            })
    return doc_evidence


def build_level2_text(subgraph: dict, doc_ids: list[str], cases: dict[str, dict]) -> str:
    """Build text with risk chain evidence highlighted inline, EN only."""
    doc_evidence = _collect_risk_evidence(subgraph)
    if not doc_evidence:
        return ""

    parts: list[str] = []
    for did in doc_ids:
        c = cases.get(did)
        if not c:
            continue
        evidences = doc_evidence.get(did, [])
        if not evidences:
            continue
        title = c.get("title", f"Doc {did}")
        raw_text = c.get("text", c.get("new_summary", c.get("description", "")))

        parts.append(
            f'<div style="margin:16px 0 8px 0;padding:10px 14px;'
            f'background:#2c3e50;color:#fff;border-radius:8px;'
            f'font-weight:bold;font-size:15px;">'
            f'Document {did}: {h(title)}</div>'
        )

        en_paras = [p.strip() for p in raw_text.split("\n") if p.strip()]

        parts.append(
            f'<div style="padding:10px 14px;background:#fff;border:1px solid #e0e0e0;'
            f'border-radius:4px;margin-bottom:8px;line-height:1.8;font-size:14px;">'
        )

        en_pos = 0
        for en_para in en_paras:
            actual_start = raw_text.find(en_para, en_pos)
            if actual_start < 0:
                continue
            actual_end = actual_start + len(en_para)

            evidence_positions: list[tuple[int, int, str, str]] = []
            for ev in evidences:
                sentence = ev["sentence"]
                idx = en_para.find(sentence)
                if idx >= 0:
                    evidence_positions.append((idx, idx + len(sentence), ev["slot_type"], ev["value"]))

            evidence_positions.sort(key=lambda x: x[0])

            annotated_parts = []
            prev_end = 0
            for estart, eend, slot_type, value in evidence_positions:
                if estart < prev_end:
                    continue
                annotated_parts.append(h(en_para[prev_end:estart]))
                color = SLOT_COLORS.get(slot_type, "#495057")
                bg = SLOT_BG_COLORS.get(slot_type, "#f8f9fa")
                annotated_parts.append(
                    f'<span style="background:{bg};border-bottom:2px solid {color};'
                    f'padding:1px 2px;border-radius:2px;cursor:default" '
                    f'title="{h(slot_type)}: {h(value)}">'
                    f'{h(en_para[estart:eend])}'
                    f'<span style="background:{color};color:#fff;padding:1px 5px;'
                    f'border-radius:8px;font-size:10px;margin-left:2px;'
                    f'vertical-align:middle;font-weight:600">{slot_type}</span>'
                    f'</span>'
                )
                prev_end = eend
            annotated_parts.append(h(en_para[prev_end:]))

            annotated_en = "".join(annotated_parts)
            parts.append(
                f'<div style="padding:4px 0;color:#1a1a1a;border-bottom:1px solid #eee;">'
                f'{annotated_en}</div>'
            )

            en_pos = actual_end

        parts.append("</div>")

    if not parts:
        return ""
    result = '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.6">'
    result += "".join(parts)
    result += "</div>"
    return result


def build_risk_chain_data(subgraph: dict) -> dict:
    nodes = subgraph.get("nodes", [])
    type_map: dict[str, dict] = {}
    for n in nodes:
        etype = n.get("entity_type", "")
        if etype in ("RiskSource", "Risk", "Consequence", "Impact",
                       "AffectedActor", "RiskControl"):
            if etype not in type_map:
                evs = n.get("evidence", [])
                evidence_str = evs[0].get("evidence_sentence", "") if evs else ""
                value = _strip_chinese(n.get("name", ""))
                evidence_str = _strip_chinese(evidence_str)
                if value:
                    type_map[etype] = {
                        "value": value,
                        "evidence": evidence_str,
                    }

    slot_mapping = {
        "risk_source_pre": "RiskSource",
        "risk_pre": "Risk",
        "consequence_pre": "Consequence",
        "impact_pre": "Impact",
        "affected_actor_pre": "AffectedActor",
        "risk_control_pre": "RiskControl",
    }

    result = {}
    for slot_key, etype in slot_mapping.items():
        info = type_map.get(etype)
        if info:
            result[slot_key] = _slot_html(info["value"], info["evidence"])
        else:
            result[slot_key] = '<div style="color:#999;font-style:italic;">Not extracted</div>'

    return result


# ── Level 3: Inference HTML (EN only) ────────────────────────────────────

INFERENCE_SLOT_COLORS = {
    "Purpose": "#856404",
    "AILifecyclePhase": "#155724",
    "Domain": "#004085",
}
INFERENCE_SLOT_BG = {
    "Purpose": "#fff3cd",
    "AILifecyclePhase": "#d4edda",
    "Domain": "#cce5ff",
}
INFERENCE_SLOT_LABELS = {
    "Purpose": "Purpose",
    "AILifecyclePhase": "Lifecycle",
    "Domain": "Domain",
}


def _inference_html(value: str, evidence: str, reasoning: str, mode: str, confidence: float) -> str:
    if not value:
        return '<div style="color:#999;font-style:italic;">Not inferred</div>'
    return (
        f'<div style="padding:8px;background:#fff;border-radius:4px;">'
        f'<div style="font-weight:600;color:#333;margin-bottom:6px;">{h(value)}</div>'
        f'<div style="font-size:12px;color:#666;padding:6px;background:#f9f9f9;'
        f'border-left:3px solid #ddd;border-radius:2px;">'
        f'<span style="color:#999;">Evidence:</span> {h(evidence)}</div>'
        f'<div style="font-size:12px;color:#8b5e3c;padding:6px;background:#fef9f0;'
        f'border-left:3px solid #d4a574;border-radius:2px;margin-top:4px;">'
        f'<span style="color:#999;">Reasoning:</span> {h(reasoning)}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:4px;">'
        f'Mode: {h(mode)} | Confidence: {confidence}</div></div>'
    )


def _collect_inference_evidence(subgraph: dict) -> dict[str, list[dict[str, str]]]:
    nodes = subgraph.get("nodes", [])
    inference_types = {"Purpose", "AILifecyclePhase", "Domain"}

    doc_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for n in nodes:
        etype = n.get("entity_type", "")
        if etype not in inference_types:
            continue
        evs = n.get("evidence", [])
        best_ev = None
        best_conf = -1
        for ev in evs:
            conf = ev.get("confidence", 0)
            sentence = ev.get("evidence_sentence", "")
            doc_id = str(ev.get("source_doc_id", ""))
            if sentence and doc_id and conf > best_conf:
                best_conf = conf
                best_ev = ev
        if best_ev is None:
            continue
        sentence = best_ev.get("evidence_sentence", "")
        doc_id = str(best_ev.get("source_doc_id", ""))
        sentence = re.sub(r'^\s*\[\d+\]\s*', '', sentence)
        if sentence:
            doc_evidence[doc_id].append({
                "sentence": sentence,
                "slot_type": etype,
                "value": n.get("name", ""),
            })
    return doc_evidence


def build_level3_text(subgraph: dict, doc_ids: list[str], cases: dict[str, dict]) -> str:
    """Build text with inference evidence highlighted inline, EN only."""
    doc_evidence = _collect_inference_evidence(subgraph)
    if not doc_evidence:
        return ""

    parts: list[str] = []
    for did in doc_ids:
        c = cases.get(did)
        if not c:
            continue
        evidences = doc_evidence.get(did, [])
        if not evidences:
            continue
        title = c.get("title", f"Doc {did}")
        raw_text = c.get("text", c.get("new_summary", c.get("description", "")))

        parts.append(
            f'<div style="margin:16px 0 8px 0;padding:10px 14px;'
            f'background:#2c3e50;color:#fff;border-radius:8px;'
            f'font-weight:bold;font-size:15px;">'
            f'Document {did}: {h(title)}</div>'
        )

        en_paras = [p.strip() for p in raw_text.split("\n") if p.strip()]

        parts.append(
            f'<div style="padding:10px 14px;background:#fff;border:1px solid #e0e0e0;'
            f'border-radius:4px;margin-bottom:8px;line-height:1.8;font-size:14px;">'
        )

        en_pos = 0
        for en_para in en_paras:
            actual_start = raw_text.find(en_para, en_pos)
            if actual_start < 0:
                continue
            actual_end = actual_start + len(en_para)

            evidence_positions: list[tuple[int, int, str, str]] = []
            for ev in evidences:
                sentence = ev["sentence"]
                idx = en_para.find(sentence)
                if idx < 0 and '...' in sentence:
                    frags = [f.strip() for f in sentence.split('...') if f.strip()]
                    best_frag = max(frags, key=len) if frags else ''
                    if len(best_frag) >= 15:
                        idx = en_para.find(best_frag)
                        if idx >= 0:
                            sentence = best_frag
                elif idx < 0 and len(sentence) > 30:
                    idx = en_para.find(sentence[:30])
                    if idx >= 0:
                        end_idx = min(idx + len(sentence), len(en_para))
                        sentence = en_para[idx:end_idx]
                if idx >= 0:
                    evidence_positions.append((idx, idx + len(sentence), ev["slot_type"], ev["value"]))

            evidence_positions.sort(key=lambda x: x[0])

            annotated_parts = []
            prev_end = 0
            for estart, eend, slot_type, value in evidence_positions:
                if estart < prev_end:
                    continue
                annotated_parts.append(h(en_para[prev_end:estart]))
                color = INFERENCE_SLOT_COLORS.get(slot_type, "#495057")
                bg = INFERENCE_SLOT_BG.get(slot_type, "#f8f9fa")
                label = INFERENCE_SLOT_LABELS.get(slot_type, slot_type)
                annotated_parts.append(
                    f'<span style="background:{bg};border-bottom:2px solid {color};'
                    f'padding:1px 2px;border-radius:2px;cursor:default" '
                    f'title="{h(slot_type)}: {h(value)}">'
                    f'{h(en_para[estart:eend])}'
                    f'<span style="background:{color};color:#fff;padding:1px 5px;'
                    f'border-radius:8px;font-size:10px;margin-left:2px;'
                    f'vertical-align:middle;font-weight:600">{label}</span>'
                    f'</span>'
                )
                prev_end = eend
            annotated_parts.append(h(en_para[prev_end:]))

            annotated_en = "".join(annotated_parts)
            parts.append(
                f'<div style="padding:4px 0;color:#1a1a1a;border-bottom:1px solid #eee;">'
                f'{annotated_en}</div>'
            )

            en_pos = actual_end

        parts.append("</div>")

    if not parts:
        return ""
    result = '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.6">'
    result += "".join(parts)
    result += "</div>"
    return result


def build_inference_data(subgraph: dict) -> dict:
    nodes = subgraph.get("nodes", [])
    result = {"purpose_pre": None, "lifecycle_pre": None, "domain_pre": None}

    type_to_slot = {
        "Purpose": "purpose_pre",
        "AILifecyclePhase": "lifecycle_pre",
        "Domain": "domain_pre",
    }

    for n in nodes:
        etype = n.get("entity_type", "")
        slot = type_to_slot.get(etype)
        if slot and result[slot] is None:
            evs = n.get("evidence", [])
            evidence_str = evs[0].get("evidence_sentence", "") if evs else ""
            reasoning = n.get("reasoning", "") or ""
            mode = n.get("extraction_mode", "inferred")
            confidence = n.get("confidence", 0.0)
            value = _strip_chinese(n.get("name", ""))
            evidence_str = _strip_chinese(evidence_str)
            if value:
                result[slot] = _inference_html(value, evidence_str, reasoning, mode, confidence)

    for slot in result:
        if result[slot] is None:
            result[slot] = '<div style="color:#999;font-style:italic;">Not inferred</div>'

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main(event_id: str = None):
    gold_events = load_gold_events()
    cases = load_cases()
    print(f"Gold standard: {len(gold_events)} events")

    level1_tasks = []
    level2_tasks = []
    level3_tasks = []

    processed = 0
    skipped = 0

    for ge in gold_events:
        eid = ge["event_id"]
        if event_id and eid != event_id:
            continue

        subgraph = load_subgraph(eid)
        if not subgraph:
            skipped += 1
            continue

        doc_ids = ge.get("ids", [])
        if not doc_ids:
            skipped += 1
            continue

        report_count = len(doc_ids)
        lang = ge.get("language", "en")
        meta_info = f"Event ID: {eid}\nDocs: {report_count}\nLanguage: {lang}"

        # ── Level 1 ── (single best-doc view)
        level1_result = build_level1_task(subgraph, doc_ids, cases, meta_info)
        level1_tasks.append(level1_result)

        # ── Level 2 ── (all docs, EN only)
        risk_chain = build_risk_chain_data(subgraph)
        level2_text = build_level2_text(subgraph, doc_ids, cases)
        level2_tasks.append({
            "data": {
                "meta_info": meta_info,
                "event_id": eid,
                "text": level2_text,
                **risk_chain,
            }
        })

        # ── Level 3 ── (all docs, EN only)
        inference = build_inference_data(subgraph)
        level3_text = build_level3_text(subgraph, doc_ids, cases)
        level3_tasks.append({
            "data": {
                "meta_info": meta_info,
                "event_id": eid,
                "text": level3_text,
                **inference,
            }
        })

        processed += 1

    # Write outputs
    for tasks, out_dir in [
        (level1_tasks, LEVEL1_DIR),
        (level2_tasks, LEVEL2_DIR),
        (level3_tasks, LEVEL3_DIR),
    ]:
        out_path = out_dir / "annotation_tasks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(tasks)} tasks to {out_path}")

    print(f"\nProcessed: {processed}, Skipped (no output): {skipped}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=str, default=None)
    args = parser.parse_args()
    main(event_id=args.event_id)
