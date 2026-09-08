"""分层 ESR 计算 —— 内容边 vs 基础设施关联边口径。

背景
====
审稿意见指出(2026-08 评审):消融表中 ESR 的分母(语义边)混合了
内容-内容边与至少一端为基础设施节点的边,且各变体的分母构成不同
(LLM-only 全部为内容边,全系统 57.7% 为基础设施关联边),行间
ESR/GKY 比较受分母构成差异混淆。

本脚本按边的端点类型分层,复用 compute_ablation_metrics.py 的
完全相同的边枚举顺序(保证 LLM judgment 键对齐),输出:

  - content 层:两端均为内容实体(#CEdges 口径,与 GDC 分子一致)
  - infra 层:至少一端为基础设施节点,再细分:
      * hub:      AIRiskIncident 枢纽边(involvesStakeholder/hasRisk/...)
      * role:     RoleAssignment 具体化边(roleHeldBy/roleInvolvesSystem/...)
      * evidence: Evidence 链接边(hasEvidence)
      * other:    其余(NewsReport/InformationSource/KnowledgeStatement 端点)

每层报告 ESR_rule / ESR_llm 与原始计数,并给出内容边口径 GKY
(GKY_content = #CEdges × ESR_llm_content)用于检验结论稳健性。

Usage:
    python eval/scripts/compute_esr_layered.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.datasets import build_documents_for_event, load_cases, load_events

OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"
GOLD_FILE = PROJECT_ROOT / "data" / "gold_standard_100_events.json"
RESULT_FILE = PROJECT_ROOT / "eval" / "results" / "esr_layered_metrics.json"

VARIANTS = [
    "full_ontorisk",
    "ablation_no_ontology",
    "ablation_no_event_aggregation",
    "ablation_no_evidence_constraint",
    "ablation_no_controlled_inference",
    "baseline_llm_only",
]

INFRA_TYPES = frozenset({
    "AIRiskIncident",
    "NewsReport",
    "InformationSource",
    "RoleAssignment",
    "Evidence",
    "KnowledgeStatement",
})

INFRA_PREDICATES = frozenset({"hasReport", "hasInformationSource", "hasSourceDocument"})

MIN_EVIDENCE_LEN = 4

# infra 关联边的子层判定:按端点类型
def _infra_sublayer(
    subject_type: str,
    object_type: str,
    predicate: str,
) -> str:
    types = {subject_type, object_type}
    if predicate == "hasEvidence" or "Evidence" in types:
        return "evidence"
    if "RoleAssignment" in types:
        return "role"
    if "AIRiskIncident" in types:
        return "hub"
    return "other"


def _normalise_text(value: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"\[[^\]]+\]", "", value)
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def _evidence_fully_supported(
    edge: dict[str, Any],
    documents: dict[str, str],
) -> bool:
    evidence_list = edge.get("evidence", [])
    if not evidence_list:
        return False
    for evidence in evidence_list:
        doc_id = str(evidence.get("source_doc_id", ""))
        sentence = _normalise_text(evidence.get("evidence_sentence", ""))
        if not doc_id or doc_id not in documents:
            return False
        if len(sentence) < MIN_EVIDENCE_LEN:
            return False
        document = _normalise_text(documents.get(doc_id, ""))
        if sentence not in document:
            return False
    return True


def main() -> None:
    cases = load_cases()
    events = {event.get("incident_id"): event for event in load_events()}
    with open(GOLD_FILE, encoding="utf-8") as f:
        gold = json.load(f)
    event_ids = sorted(
        item["event_id"]
        for item in gold.get("gold_standard", [])
        if item.get("event_id")
    )

    judgment_file = (
        PROJECT_ROOT / "eval" / "results" / "ablation_llm_judgments.json"
    )
    with open(judgment_file, encoding="utf-8") as f:
        judgments = json.load(f)

    event_documents: dict[str, dict[str, str]] = {}
    for event_id in event_ids:
        event = events.get(event_id)
        if event:
            event_documents[event_id] = {
                report.doc_id: report.content
                for report in build_documents_for_event(event, cases)
            }

    results: dict[str, Any] = {}

    for variant in VARIANTS:
        layer_counts: dict[str, defaultdict[str, int]] = {
            layer: defaultdict(int)
            for layer in ("content", "hub", "role", "evidence", "other")
        }
        pred_by_layer: dict[str, Counter] = {
            layer: Counter()
            for layer in ("content", "hub", "role", "evidence", "other")
        }

        for event_id in event_ids:
            path = OUTPUT_DIR / variant / event_id / "event_subgraph.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                subgraph = json.load(f)

            nodes = {n["id"]: n for n in subgraph.get("nodes", [])}
            incident = subgraph.get("incident_node")
            if incident:
                nodes[incident["id"]] = incident

            docs = event_documents.get(event_id, {})

            # 与 compute_ablation_metrics.py 完全一致的枚举顺序
            semantic_edges = [
                edge
                for edge in subgraph.get("edges", [])
                if edge.get("predicate") not in INFRA_PREDICATES
                and edge.get("extraction_mode") != "completed"
            ]

            for edge_index, edge in enumerate(semantic_edges):
                subject_type = nodes.get(
                    edge.get("subject_id", ""), {}
                ).get("entity_type", "")
                object_type = nodes.get(
                    edge.get("object_id", ""), {}
                ).get("entity_type", "")
                s_infra = subject_type in INFRA_TYPES
                o_infra = object_type in INFRA_TYPES

                if s_infra or o_infra:
                    layer = _infra_sublayer(
                        subject_type, object_type, edge.get("predicate", "")
                    )
                else:
                    layer = "content"

                layer_counts[layer]["total"] += 1
                pred_by_layer[layer][edge.get("predicate", "")] += 1

                if _evidence_fully_supported(edge, docs):
                    layer_counts[layer]["rule_supported"] += 1
                    layer_counts[layer]["llm_supported"] += 1
                else:
                    key = f"{variant}|{event_id}|evidence|{edge_index}"
                    if key in judgments:
                        layer_counts[layer]["llm_judged"] += 1
                        if judgments[key]:
                            layer_counts[layer]["llm_supported"] += 1

        variant_out: dict[str, Any] = {}
        for layer in ("content", "hub", "role", "evidence", "other"):
            c = layer_counts[layer]
            total = c["total"]
            if total == 0:
                variant_out[layer] = {"total": 0}
                continue
            variant_out[layer] = {
                "total": total,
                "rule_supported": c["rule_supported"],
                "llm_judged": c["llm_judged"],
                "llm_supported": c["llm_supported"],
                "ESR_rule": round(c["rule_supported"] / total * 100, 1),
                "ESR_llm": round(c["llm_supported"] / total * 100, 1),
                "top_predicates": dict(
                    pred_by_layer[layer].most_common(6)
                ),
            }

        # 汇总层与内容边口径 GKY
        infra_total = sum(
            layer_counts[l]["total"]
            for l in ("hub", "role", "evidence", "other")
        )
        infra_rule = sum(
            layer_counts[l]["rule_supported"]
            for l in ("hub", "role", "evidence", "other")
        )
        infra_llm = sum(
            layer_counts[l]["llm_supported"]
            for l in ("hub", "role", "evidence", "other")
        )
        c = layer_counts["content"]
        content_total = c["total"]
        esr_llm_content = (
            round(c["llm_supported"] / content_total * 100, 1)
            if content_total
            else None
        )
        variant_out["infra_combined"] = {
            "total": infra_total,
            "ESR_rule": round(infra_rule / infra_total * 100, 1)
            if infra_total
            else None,
            "ESR_llm": round(infra_llm / infra_total * 100, 1)
            if infra_total
            else None,
        }
        variant_out["GKY_content"] = (
            round(content_total * c["llm_supported"] / content_total)
            if content_total
            else 0
        )
        results[variant] = variant_out

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 摘要输出
    for variant in VARIANTS:
        r = results[variant]
        content = r["content"]
        infra = r["infra_combined"]
        print(f"\n== {variant} ==")
        print(
            f"  content: n={content['total']:>6}  "
            f"ESR_rule={content['ESR_rule']:>5}  "
            f"ESR_llm={content['ESR_llm']:>5}  "
            f"GKY_content={r['GKY_content']}"
        )
        print(
            f"  infra:   n={infra['total']:>6}  "
            f"ESR_rule={infra['ESR_rule'] if infra['ESR_rule'] is not None else '—':>5}  "
            f"ESR_llm={infra['ESR_llm'] if infra['ESR_llm'] is not None else '—':>5}"
        )
        for sub in ("hub", "role", "evidence", "other"):
            s = r[sub]
            if s["total"]:
                print(
                    f"    {sub:<8} n={s['total']:>6}  "
                    f"ESR_rule={s['ESR_rule']:>5}  ESR_llm={s['ESR_llm']:>5}"
                )

    print(f"\nSaved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
