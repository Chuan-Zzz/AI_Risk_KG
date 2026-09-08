#!/usr/bin/env python3
"""Group Evidence Purity (GEP) — 组级证据纯度指标.

动机
----
CDC 只检测组内对象是否相互矛盾: 矛盾是稀有事件, 导致全变体
98.5-99.9% 的天花板效应, 无区分度; 且 CDC 看不见两类真实错误:
  1) 清单倾倒(list-dumping): 列表型文档把大量无关对象挂到同一
     subject 下, 对象间不矛盾 → CDC 放行;
  2) 标题/占位符 subject: 组本身不该存在, CDC 仍按对象间关系判一致。

GEP 把 ESR 的边级证据支持判定聚合到 CDC 的组级单元上, 直接回答
"聚合产生的每个多对象组里, 有多少对象是证据落地的":

  组 g = 同 (归一化 subject, predicate) 下 ≥2 个不同 object (与 CDC 的 H 一致)
  purity(g) = 组内证据支持的对象数 / 组内对象总数

  GEP = mean_g purity(g)          宏平均: 随机抽一组的期望纯度
  PGR = share(purity(g) < 0.5)    污染组率: 过半对象无证据(清单倾倒签名)
  FGG = share(purity(g) = 1.0)    全证据组率

口径
----
对象支持标签与 ESR_LLM 完全同口径(全变体一致, 零新 LLM 调用):
  supported = 字面匹配(_evidence_fully_supported) OR judge 判支持
  (judge 仅对字面未匹配边运行, 覆盖率 ≥99.8%; 未判定且未匹配 → 不支持)

Usage:
    python eval/scripts/compute_group_purity.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_ablation_metrics import (  # noqa: E402
    OUTPUT_DIR,
    VARIANTS,
    VARIANT_LABELS,
    _evidence_fully_supported,
    _load_gold_ids,
    _load_json,
    _load_subgraph,
    _normalise_text,
    _semantic_edges,
)
from src.core.datasets import build_documents_for_event, load_cases, load_events  # noqa: E402

RESULT_FILE = PROJECT_ROOT / "eval" / "results" / "group_purity_metrics.json"
JUDGMENT_FILE = PROJECT_ROOT / "eval" / "results" / "ablation_llm_judgments.json"


def _group_purities(
    variant: str,
    event_ids: list[str],
    event_documents: dict[str, dict[str, str]],
    judgments: dict[str, bool],
) -> list[dict[str, Any]]:
    """返回该变体全部多对象组的纯度明细(跨事件池化)。"""
    groups: list[dict[str, Any]] = []
    for event_id in event_ids:
        subgraph = _load_subgraph(variant, event_id)
        if subgraph is None:
            continue
        docs = event_documents.get(event_id, {})

        # 边级支持标签(与 ESR_LLM 同口径)
        supports: list[bool] = []
        for edge_index, edge in enumerate(_semantic_edges(subgraph)):
            literal_ok, _ = _evidence_fully_supported(edge, docs)
            if literal_ok:
                supports.append(True)
            else:
                key = f"{variant}|{event_id}|evidence|{edge_index}"
                supports.append(bool(judgments.get(key, False)))

        # 按 (归一化subject, predicate) 分组; 组内按归一化object去重
        # (保留首条边, 与 _statement_groups 语义一致)
        grouped: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
        meta: dict[tuple[str, str], str] = {}
        for edge, supported in zip(_semantic_edges(subgraph), supports):
            subject = _normalise_text(edge.get("subject_name", ""))
            predicate = edge.get("predicate", "")
            obj = _normalise_text(edge.get("object_name", ""))
            if not subject or not predicate or not obj:
                continue
            gkey = (subject, predicate)
            grouped[gkey].setdefault(obj, supported)
            meta.setdefault(gkey, edge.get("subject_name", ""))

        for (subject, predicate), objects in grouped.items():
            if len(objects) < 2:  # 只看多对象组(与 CDC 的 H 同单元)
                continue
            vals = list(objects.values())
            groups.append({
                "event_id": event_id,
                "subject": meta[(subject, predicate)],
                "predicate": predicate,
                "n_objects": len(vals),
                "n_supported": sum(vals),
                "purity": sum(vals) / len(vals),
            })
    return groups


def main() -> None:
    cases = load_cases()
    events = {e.get("incident_id"): e for e in load_events()}
    event_ids = _load_gold_ids()
    judgments = _load_json(JUDGMENT_FILE, {})

    event_documents: dict[str, dict[str, str]] = {}
    for event_id in event_ids:
        event = events.get(event_id)
        if event:
            event_documents[event_id] = {
                r.doc_id: r.content
                for r in build_documents_for_event(event, cases)
            }

    ablation = _load_json(
        PROJECT_ROOT / "eval" / "results" / "ablation_metrics.json", {}
    )

    results: dict[str, Any] = {}
    print(f"{'variant':<33}{'GEP%':>7}{'PGR%':>7}{'FGG%':>7}"
          f"{'#grp':>7}{'ESR_llm':>9}{'CDC_llm':>9}")
    for variant in VARIANTS:
        groups = _group_purities(variant, event_ids, event_documents, judgments)
        n = len(groups)
        gep = sum(g["purity"] for g in groups) / n * 100 if n else 0.0
        pgr = sum(g["purity"] < 0.5 for g in groups) / n * 100 if n else 0.0
        fgg = sum(g["purity"] == 1.0 for g in groups) / n * 100 if n else 0.0
        base = ablation.get(variant, {})
        results[variant] = {
            "GEP": round(gep, 1),
            "PGR": round(pgr, 1),
            "FGG": round(fgg, 1),
            "multi_object_groups": n,
            "total_objects": sum(g["n_objects"] for g in groups),
            "supported_objects": sum(g["n_supported"] for g in groups),
            "reference_ESR_llm": base.get("ESR_llm"),
            "reference_CDC_llm": base.get("CDC_llm"),
        }
        print(f"{VARIANT_LABELS.get(variant, variant):<33}"
              f"{gep:7.1f}{pgr:7.1f}{fgg:7.1f}{n:7d}"
              f"{base.get('ESR_llm', float('nan')):9.1f}"
              f"{base.get('CDC_llm', float('nan')):9.1f}")

    Path(RESULT_FILE).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
