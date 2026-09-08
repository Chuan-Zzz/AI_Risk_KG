"""消融实验指标计算 —— 统一口径版。

六项指标均从 event_subgraph.json 离线计算,不依赖 LLM,不依赖人工标注。

指标定义
=========

GDC (Graph Density Coherence) = content_edges / content_nodes
    每实体平均语义边数,衡量图结构密度。
    分子分母均排除基础设施节点(NewsReport / InformationSource / RoleAssignment 等)
    及其连边(hasReport / hasInformationSource / hasSourceDocument)。

KDC (Knowledge Density Coherence) = content_statements / content_nodes
    每实体平均知识断言数,衡量知识密度。
    只统计从 content_edges 展开的知识断言(排除基础设施断言)。

RCC (Risk-Chain Completeness) = events_with_complete_chain / total_events * 100
    完整风险链事件占比。
    要求存在一条连通路径:
        RiskSource --causes--> Risk --leadsTo--> Consequence
        --impacts--> Impact --affects--> AffectedActor
    而非仅检查类型与谓词的存在性。

TDV (Type Diversity) = sum(unique_content_types_per_event) / total_events
    每事件内容实体类型种类数的均值。

ESR (Evidence Support Rate) = fully_supported_edges / semantic_edges * 100
    语义边中"全部 evidence 均能在原文中字面匹配"的占比。
    无 evidence 的边计为不支持;任一条 evidence 无法匹配则计为不支持。
    去除了 12 字符硬阈值(对中文不友好),仅保留 4 字符最低长度防止平凡匹配。

CDC (Cross-Document Consistency) = consistent_groups / total_groups * 100
    规则化跨文档一致性:按 (subject, predicate) 分组,
    多 object 组在以下条件下判为一致:
      - multirole 谓词(involvesStakeholder / roleHeldBy):所有 object 均为利益相关者类型;
      - 非 multirole 谓词:所有 object 的 entity_type 相同(允许多实例,如多个 AICapability)。
    仅当 object 类型冲突时才判为不一致。

Usage:
    python eval/scripts/compute_ablation_metrics.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.datasets import build_documents_for_event, load_cases, load_events
from src.core.ontology import AIROOntology

OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"
GOLD_FILE = PROJECT_ROOT / "data" / "gold_standard_100_events.json"
RESULT_FILE = PROJECT_ROOT / "eval" / "results" / "ablation_metrics.json"
DETAIL_FILE = PROJECT_ROOT / "eval" / "results" / "ablation_metrics_details.json"

VARIANTS = [
    "full_ontorisk",
    "ablation_no_ontology",
    "ablation_no_event_aggregation",
    "ablation_no_evidence_constraint",
    "ablation_no_controlled_inference",
    "baseline_llm_only",
]

VARIANT_LABELS = {
    "full_ontorisk": "Full OntoRisk",
    "ablation_no_ontology": "w/o Ontology Constraint",
    "ablation_no_event_aggregation": "w/o Event Aggregation",
    "ablation_no_evidence_constraint": "w/o Evidence Constraint",
    "ablation_no_controlled_inference": "w/o Controlled Inference",
    "baseline_llm_only": "LLM-only Baseline",
}

# 基础设施节点类型(不计入内容实体)
INFRA_TYPES = frozenset({
    "AIRiskIncident",
    "NewsReport",
    "InformationSource",
    "RoleAssignment",
    "Evidence",
    "KnowledgeStatement",
})

# 基础设施谓词(不计入语义边)
INFRA_PREDICATES = frozenset({"hasReport", "hasInformationSource", "hasSourceDocument"})

# 细分利益相关者类型(用于 RSR 指标)
STAKEHOLDER_SPECIFIC_TYPES = frozenset({
    "AIDeveloper", "AIProvider", "AIDeployer", "AIUser", "Regulator", "AffectedActor",
})

# 利益相关者类型(用于 multirole 谓词的 CDC 判定)
STAKEHOLDER_TYPES = frozenset({
    "Stakeholder", "AIDeveloper", "AIProvider", "AIDeployer",
    "AIUser", "Regulator", "AffectedActor",
})

# multirole 谓词:允许多个不同利益相关者角色作为 object
MULTIROLE_PREDICATES = frozenset({"involvesStakeholder", "roleHeldBy"})

# 风险链连通路径所需的谓词序列
CHAIN_PREDICATES = ("causes", "leadsTo", "impacts", "affects")

# evidence 句子归一化后的最低字符数(防止 "AI" 之类的平凡匹配)
MIN_EVIDENCE_LEN = 4


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _normalise_text(value: str) -> str:
    """NFKC 归一化 + 去除标点/空格,用于字面匹配。"""
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"\[[^\]]+\]", "", value)  # 去除 markdown 链接文本
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_gold_ids() -> list[str]:
    data = _load_json(GOLD_FILE, {})
    return sorted(
        item["event_id"]
        for item in data.get("gold_standard", [])
        if item.get("event_id")
    )


def _load_subgraph(variant: str, event_id: str) -> dict[str, Any] | None:
    path = OUTPUT_DIR / variant / event_id / "event_subgraph.json"
    return _load_json(path, None)


def _node_map(subgraph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """构建 node_id → node 字典(含 incident_node)。"""
    nodes = list(subgraph.get("nodes", []))
    incident = subgraph.get("incident_node")
    if incident:
        nodes.append(incident)
    return {node.get("id", ""): node for node in nodes if node.get("id")}


def _is_content_node(node: dict[str, Any] | None) -> bool:
    return bool(node) and node.get("entity_type") not in INFRA_TYPES


def _semantic_edges(subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    """返回语义边:排除基础设施谓词和 completed 模式的自引用边。"""
    return [
        edge for edge in subgraph.get("edges", [])
        if edge.get("predicate") not in INFRA_PREDICATES
        and edge.get("extraction_mode") != "completed"
    ]


def _content_edges(
    subgraph: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """返回两端均为内容实体的语义边。"""
    return [
        edge for edge in _semantic_edges(subgraph)
        if _is_content_node(nodes.get(edge.get("subject_id", "")))
        and _is_content_node(nodes.get(edge.get("object_id", "")))
    ]


def _content_statements(
    subgraph: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """返回从内容实体出发的知识断言(排除基础设施断言)。"""
    result: list[dict[str, Any]] = []
    for stmt in subgraph.get("knowledge_statements", []):
        if stmt.get("predicate") in INFRA_PREDICATES:
            continue
        if stmt.get("extraction_mode") == "completed":
            continue
        if not _is_content_node(nodes.get(stmt.get("subject_id", ""))):
            continue
        if not _is_content_node(nodes.get(stmt.get("object_id", ""))):
            continue
        result.append(stmt)
    return result


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _has_complete_risk_path(
    subgraph: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> bool:
    """检查是否存在一条完整的五槽连通路径。

    RiskSource --causes--> Risk --leadsTo--> Consequence
    --impacts--> Impact --affects--> AffectedActor
    """
    by_predicate: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for edge in _semantic_edges(subgraph):
        by_predicate[edge.get("predicate", "")][
            edge.get("subject_id", "")
        ].add(edge.get("object_id", ""))

    for source_id, source in nodes.items():
        if source.get("entity_type") != "RiskSource":
            continue
        for risk_id in by_predicate["causes"].get(source_id, set()):
            if nodes.get(risk_id, {}).get("entity_type") != "Risk":
                continue
            for consequence_id in by_predicate["leadsTo"].get(risk_id, set()):
                if nodes.get(consequence_id, {}).get("entity_type") != "Consequence":
                    continue
                for impact_id in by_predicate["impacts"].get(consequence_id, set()):
                    if nodes.get(impact_id, {}).get("entity_type") != "Impact":
                        continue
                    for actor_id in by_predicate["affects"].get(impact_id, set()):
                        if nodes.get(actor_id, {}).get("entity_type") == "AffectedActor":
                            return True
    return False


def _evidence_fully_supported(
    edge: dict[str, Any],
    documents: dict[str, str],
) -> tuple[bool, str]:
    """检查边的全部 evidence 是否都能在原文中字面匹配。

    返回 (是否全部匹配, 原因标签):
      - 无 evidence → (False, "missing_evidence")
      - 全部 evidence 匹配 → (True, "matched")
      - 任一 evidence 未匹配 → (False, "unmatched_evidence")
    """
    evidence_list = edge.get("evidence", [])
    if not evidence_list:
        return False, "missing_evidence"

    for evidence in evidence_list:
        doc_id = str(evidence.get("source_doc_id", ""))
        sentence = _normalise_text(evidence.get("evidence_sentence", ""))
        if not doc_id or doc_id not in documents:
            return False, "unmatched_evidence"
        if len(sentence) < MIN_EVIDENCE_LEN:
            return False, "unmatched_evidence"
        document = _normalise_text(documents.get(doc_id, ""))
        if sentence not in document:
            return False, "unmatched_evidence"

    return True, "matched"


def _statement_groups(
    subgraph: dict[str, Any],
) -> list[dict[str, Any]]:
    """按 (归一化 subject, predicate) 分组语义边,用于 CDC 判定。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in _semantic_edges(subgraph):
        subject = _normalise_text(edge.get("subject_name", ""))
        predicate = edge.get("predicate", "")
        if not subject or not predicate:
            continue
        groups[(subject, predicate)].append(edge)

    result: list[dict[str, Any]] = []
    for (subject, predicate), edges in groups.items():
        objects: dict[str, dict[str, Any]] = {}
        for edge in edges:
            obj = _normalise_text(edge.get("object_name", ""))
            if obj:
                objects.setdefault(obj, edge)
        result.append({
            "key": f"{subject}|{predicate}",
            "subject": edges[0].get("subject_name", ""),
            "predicate": predicate,
            "objects": list(objects.values()),
            "requires_judgment": len(objects) > 1,
        })
    return result


def _rule_based_group_consistent(
    group: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> bool:
    """规则化 CDC 判定。

    - 单 object 组:一致。
    - multirole 谓词多 object 组:所有 object 均为利益相关者类型则一致。
    - 非 multirole 谓词多 object 组:所有 object 的 entity_type 相同则一致
      (允许多实例,如一个 AISystem 有多个 AICapability);
      类型冲突则不一致。
    """
    if not group["requires_judgment"]:
        return True

    object_types = [
        nodes.get(edge.get("object_id", ""), {}).get("entity_type", "")
        for edge in group["objects"]
    ]
    object_types = [t for t in object_types if t]

    if not object_types:
        return True

    if group["predicate"] in MULTIROLE_PREDICATES:
        return all(t in STAKEHOLDER_TYPES for t in object_types)

    # 非 multirole:类型全相同则一致,否则冲突
    return len(set(object_types)) == 1


# ---------------------------------------------------------------------------
# 补充结构指标:TDE / RSR / CDRR / OVR
# ---------------------------------------------------------------------------

def _type_distribution_entropy(
    nodes: dict[str, dict[str, Any]],
) -> float:
    """计算内容实体类型分布的香农熵(以 2 为底)。

    H = -sum(p_i * log2(p_i)), p_i = count(type_i) / N
    类型分布越均匀,熵越高。
    """
    type_counts: dict[str, int] = defaultdict(int)
    total = 0
    for node in nodes.values():
        if not _is_content_node(node):
            continue
        type_counts[node.get("entity_type", "")] += 1
        total += 1
    if total <= 1:
        return 0.0
    import math
    entropy = 0.0
    for count in type_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _role_specificity_rate(
    nodes: dict[str, dict[str, Any]],
) -> float:
    """细分角色率 = 细分角色实体数 / (细分角色 + 通用 Stakeholder) 实体数 * 100。

    衡量利益相关者被分类到具体子类型的比例。
    Full 应高(角色被细分),no_ontology/no_inference 应低(退化成 Stakeholder)。
    """
    specific = 0
    generic = 0
    for node in nodes.values():
        etype = node.get("entity_type", "")
        if etype in STAKEHOLDER_SPECIFIC_TYPES:
            specific += 1
        elif etype == "Stakeholder":
            generic += 1
    denom = specific + generic
    if denom == 0:
        return 0.0
    return specific / denom * 100


def _cross_doc_redundancy_rate(
    nodes: dict[str, dict[str, Any]],
) -> float:
    """跨文档冗余率 = 名称归一化后重复的实体数 / 总内容实体数 * 100。

    "重复"指同一事件内多个内容实体归一化后名称相同(去重后少掉的即为重复)。
    no_agg 应显著高于 Full(因为不做聚合去重)。
    """
    names: list[str] = []
    for node in nodes.values():
        if not _is_content_node(node):
            continue
        name = _normalise_text(node.get("name", ""))
        if name:
            names.append(name)
    if not names:
        return 0.0
    unique = len(set(names))
    duplicates = len(names) - unique
    return duplicates / len(names) * 100


def _ontology_violation_rate(
    semantic_edges: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    ontology: AIROOntology,
) -> int:
    """统计违反 domain/range 约束的语义边数。

    违规判定规则:
      1. 谓词不在 ontology.yml 中 → 违规(使用未定义的关系)
      2. 谓词有 domain 约束但 subject_type 不在 domain 中 → 违规
      3. 谓词有 range 约束但 object_type 不在 range 中 → 违规

    返回违规边数(用于后续计算 OVR = violations / total * 100)。
    no_ontology 变体不使用本体约束,应产生更多未定义谓词和类型违规。
    """
    from src.core.models import OntologyClass

    violations = 0
    for edge in semantic_edges:
        predicate = edge.get("predicate", "")
        subject_type = nodes.get(edge.get("subject_id", ""), {}).get("entity_type", "")
        object_type = nodes.get(edge.get("object_id", ""), {}).get("entity_type", "")
        if not predicate:
            continue

        domain = ontology.get_domain(predicate)
        range_ = ontology.get_range(predicate)

        # 谓词不在 ontology.yml 中 → 违规
        if not domain and not range_:
            violations += 1
            continue

        violated = False
        if domain and subject_type:
            try:
                s_cls = OntologyClass(subject_type)
                if s_cls is not None and s_cls not in domain:
                    violated = True
            except ValueError:
                pass
        if not violated and range_ and object_type:
            try:
                o_cls = OntologyClass(object_type)
                if o_cls is not None and o_cls not in range_:
                    violated = True
            except ValueError:
                pass
        if violated:
            violations += 1
    return violations


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _evaluate_variant(
    variant: str,
    event_ids: list[str],
    event_documents: dict[str, dict[str, str]],
    judgments: dict[str, bool],
    ontology: AIROOntology,
) -> tuple[dict[str, Any], dict[str, Any]]:
    totals: defaultdict[str, int] = defaultdict(int)
    details: dict[str, Any] = {}

    for event_id in event_ids:
        subgraph = _load_subgraph(variant, event_id)
        if subgraph is None:
            continue

        totals["events"] += 1
        nodes = _node_map(subgraph)
        docs = event_documents.get(event_id, {})

        content_nodes_count = sum(
            1 for node in nodes.values() if _is_content_node(node)
        )
        content_edges_list = _content_edges(subgraph, nodes)
        content_stmts = _content_statements(subgraph, nodes)
        semantic_edges = _semantic_edges(subgraph)

        # ESR (规则化:字面匹配) + LLM-ESR (语义判定)
        evidence_counts: defaultdict[str, int] = defaultdict(int)
        fully_supported = 0          # 字面全匹配
        llm_supported = 0            # 字面匹配 OR LLM 判定支持
        llm_judged_count = 0         # 有 LLM 判定的边数
        for edge_index, edge in enumerate(semantic_edges):
            supported, reason = _evidence_fully_supported(edge, docs)
            evidence_counts[reason] += 1
            fully_supported += int(supported)
            if supported:
                llm_supported += 1
            else:
                # 对字面未匹配的边查 LLM judgment
                key = f"{variant}|{event_id}|evidence|{edge_index}"
                if key in judgments:
                    llm_judged_count += 1
                    if judgments[key]:
                        llm_supported += 1

        # CDC (规则化) + LLM-CDC (语义判定)
        groups = _statement_groups(subgraph)
        rule_consistent = 0
        llm_consistent = 0
        multi_object_groups = 0
        llm_judged_groups = 0
        for group in groups:
            if group["requires_judgment"]:
                multi_object_groups += 1
            rule_ok = _rule_based_group_consistent(group, nodes)
            rule_consistent += int(rule_ok)
            if group["requires_judgment"]:
                key = f"{variant}|{event_id}|group|{group['key']}"
                if key in judgments:
                    llm_judged_groups += 1
                    if judgments[key]:
                        llm_consistent += 1
            else:
                llm_consistent += 1  # 单 object 组直接计为一致

        # RCC
        has_chain = _has_complete_risk_path(subgraph, nodes)

        # TDV + TDE
        unique_types = {
            node.get("entity_type", "")
            for node in nodes.values()
            if _is_content_node(node)
        }
        entropy = _type_distribution_entropy(nodes)

        # RSR
        rsr = _role_specificity_rate(nodes)

        # CDRR
        cdrr = _cross_doc_redundancy_rate(nodes)

        # OVR
        ovr_violations = _ontology_violation_rate(semantic_edges, nodes, ontology)

        totals["content_nodes"] += content_nodes_count
        totals["content_edges"] += len(content_edges_list)
        totals["content_statements"] += len(content_stmts)
        totals["semantic_edges"] += len(semantic_edges)
        totals["fully_supported_edges"] += fully_supported
        totals["llm_supported_edges"] += llm_supported
        totals["llm_judged_evidence"] += llm_judged_count
        totals["statement_groups"] += len(groups)
        totals["rule_consistent_groups"] += rule_consistent
        totals["llm_consistent_groups"] += llm_consistent
        totals["llm_judged_groups"] += llm_judged_groups
        totals["multi_object_groups"] += multi_object_groups
        totals["complete_paths"] += int(has_chain)
        totals["type_diversity_sum"] += len(unique_types)
        totals["type_entropy_sum"] += entropy
        totals["rsr_sum"] += rsr
        totals["cdrr_sum"] += cdrr
        totals["ovr_violations"] += ovr_violations
        for reason, count in evidence_counts.items():
            totals[f"evidence_{reason}"] += count

        details[event_id] = {
            "content_nodes": content_nodes_count,
            "content_edges": len(content_edges_list),
            "content_statements": len(content_stmts),
            "semantic_edges": len(semantic_edges),
            "fully_supported_edges": fully_supported,
            "llm_supported_edges": llm_supported,
            "evidence_counts": dict(evidence_counts),
            "statement_groups": len(groups),
            "rule_consistent_groups": rule_consistent,
            "llm_consistent_groups": llm_consistent,
            "multi_object_groups": multi_object_groups,
            "complete_risk_path": has_chain,
            "unique_types": len(unique_types),
            "type_entropy": round(entropy, 3),
            "rsr": round(rsr, 1),
            "cdrr": round(cdrr, 1),
            "ovr_violations": ovr_violations,
        }

    events = totals["events"]
    semantic_edges = totals["semantic_edges"]
    groups = totals["statement_groups"]

    # LLM-ESR 覆盖率:有多少字面未匹配的边被 LLM 判定了
    unmatched = totals["evidence_unmatched_evidence"]
    llm_esr_coverage = (
        round(totals["llm_judged_evidence"] / unmatched * 100, 1)
        if unmatched else 100.0
    )
    # LLM-CDC 覆盖率:有多少多 object 组被 LLM 判定了
    llm_cdc_coverage = (
        round(totals["llm_judged_groups"] / totals["multi_object_groups"] * 100, 1)
        if totals["multi_object_groups"] else 100.0
    )

    metrics = {
        # 结构层(规则化,零成本)
        "GDC": round(
            totals["content_edges"] / totals["content_nodes"], 2
        ) if totals["content_nodes"] else 0.0,
        "KDC": round(
            totals["content_statements"] / totals["content_nodes"], 2
        ) if totals["content_nodes"] else 0.0,
        "RCC": round(
            totals["complete_paths"] / events * 100, 1
        ) if events else 0.0,
        "TDV": round(
            totals["type_diversity_sum"] / events, 1
        ) if events else 0.0,
        # 模块针对性指标(规则化,零成本)
        "TDE": round(
            totals["type_entropy_sum"] / events, 2
        ) if events else 0.0,
        "RSR": round(
            totals["rsr_sum"] / events, 1
        ) if events else 0.0,
        "CDRR": round(
            totals["cdrr_sum"] / events, 1
        ) if events else 0.0,
        "OVR": round(
            totals["ovr_violations"] / semantic_edges * 100, 1
        ) if semantic_edges else 0.0,
        # 证据/一致性(规则化 baseline)
        "ESR_rule": round(
            totals["fully_supported_edges"] / semantic_edges * 100, 1
        ) if semantic_edges else 0.0,
        "CDC_rule": round(
            totals["rule_consistent_groups"] / groups * 100, 1
        ) if groups else 0.0,
        # 证据/一致性(LLM 语义判定,若 judgment 文件存在)
        "ESR_llm": round(
            totals["llm_supported_edges"] / semantic_edges * 100, 1
        ) if semantic_edges and totals["llm_judged_evidence"] > 0 else None,
        "CDC_llm": round(
            totals["llm_consistent_groups"] / groups * 100, 1
        ) if groups and totals["llm_judged_groups"] > 0 else None,
        "ESR_llm_coverage": llm_esr_coverage,
        "CDC_llm_coverage": llm_cdc_coverage,
        # 原始计数
        "events": events,
        "content_nodes": totals["content_nodes"],
        "content_edges": totals["content_edges"],
        "content_statements": totals["content_statements"],
        "semantic_edges": semantic_edges,
        "fully_supported_edges": totals["fully_supported_edges"],
        "llm_supported_edges": totals["llm_supported_edges"],
        "statement_groups": groups,
        "rule_consistent_groups": totals["rule_consistent_groups"],
        "llm_consistent_groups": totals["llm_consistent_groups"],
        "multi_object_groups": totals["multi_object_groups"],
        "complete_risk_paths": totals["complete_paths"],
        "ovr_violations": totals["ovr_violations"],
        "evidence_breakdown": {
            "matched": totals["evidence_matched"],
            "missing_evidence": totals["evidence_missing_evidence"],
            "unmatched_evidence": totals["evidence_unmatched_evidence"],
        },
    }
    return metrics, details


def main() -> None:
    cases = load_cases()
    events = {event.get("incident_id"): event for event in load_events()}
    event_ids = _load_gold_ids()
    ontology = AIROOntology()

    print(f"Gold standard events: {len(event_ids)}")

    # 预加载每个事件的源文档(用于 ESR 字面匹配)
    event_documents: dict[str, dict[str, str]] = {}
    for event_id in event_ids:
        event = events.get(event_id)
        if event:
            event_documents[event_id] = {
                report.doc_id: report.content
                for report in build_documents_for_event(event, cases)
            }

    # 加载已有的 LLM judgment(若存在)
    judgment_file = PROJECT_ROOT / "eval" / "results" / "ablation_llm_judgments.json"
    judgments = _load_json(judgment_file, {})
    if judgments:
        print(f"Loaded {len(judgments)} LLM judgments from {judgment_file}")

    results: dict[str, Any] = {}
    all_details: dict[str, Any] = {}

    for variant in VARIANTS:
        metrics, details = _evaluate_variant(
            variant, event_ids, event_documents, judgments, ontology
        )
        results[variant] = metrics
        all_details[variant] = details
        label = VARIANT_LABELS.get(variant, variant)
        esr_llm_str = (
            f"{metrics['ESR_llm']:5.1f}" if metrics["ESR_llm"] is not None else "  —  "
        )
        cdc_llm_str = (
            f"{metrics['CDC_llm']:5.1f}" if metrics["CDC_llm"] is not None else "  —  "
        )
        print(
            f"  {label:<35} "
            f"GDC={metrics['GDC']:5.2f}  "
            f"KDC={metrics['KDC']:5.2f}  "
            f"RCC={metrics['RCC']:5.1f}  "
            f"TDV={metrics['TDV']:5.1f}  "
            f"TDE={metrics['TDE']:5.2f}  "
            f"RSR={metrics['RSR']:5.1f}  "
            f"CDRR={metrics['CDRR']:5.1f}  "
            f"OVR={metrics['OVR']:5.1f}  "
            f"ESR_r={metrics['ESR_rule']:5.1f}  "
            f"CDC_r={metrics['CDC_rule']:5.1f}  "
            f"ESR_l={esr_llm_str}  "
            f"CDC_l={cdc_llm_str}  "
            f"({metrics['events']} ev)"
        )

    _save_json(RESULT_FILE, results)
    _save_json(DETAIL_FILE, all_details)
    print(f"\nResults saved to {RESULT_FILE}")
    print(f"Details saved to {DETAIL_FILE}")


if __name__ == "__main__":
    main()
