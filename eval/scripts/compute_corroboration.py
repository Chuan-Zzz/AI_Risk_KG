#!/usr/bin/env python3
"""Evidence Corroboration Rate (ECR) — 跨文档证据佐证率.

动机
----
CDC 检测"对象间矛盾", 但证据锚定使矛盾通道近乎封死(跨文档组仅
2.2% 不一致), 全变体 98.5-99.9% 天花板, 无区分度。同一命题
("跨文档聚合产出一致的知识")的信息量方向是**正向佐证**:
多篇文档独立支持同一事实。

定义
----
edge 的佐证 = 其 evidence 列表覆盖 ≥2 个不同 source_doc_id
(同文档多条证据不算, 按 distinct doc 计)。

  ECR_all = 佐证边数 / 全部语义边数          (主口径)
  ECR_sup = 佐证边数 / 证据支持边数           (条件口径, 剥离 ESR 影响)
  CDF     = 对象来源覆盖 ≥2 文档的多对象组占比 (组级融合视角)

可证伪的预期: 事件聚合把同事件多篇文档的证据融合到合并实体上,
故 Full >> w/o EventAgg >> LLM-only(文档局部抽取几乎无法佐证)。

零 LLM 调用: 仅统计 event_subgraph.json 中已存的 evidence doc ids。

Usage:
    python eval/scripts/compute_corroboration.py
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

RESULT_FILE = PROJECT_ROOT / "eval" / "results" / "corroboration_metrics.json"


def _edge_doc_ids(edge: dict[str, Any]) -> set[str]:
    return {
        str(ev.get("source_doc_id"))
        for ev in edge.get("evidence", [])
        if ev.get("source_doc_id")
    }


def main() -> None:
    cases = load_cases()
    events = {e.get("incident_id"): e for e in load_events()}
    event_ids = _load_gold_ids()

    event_documents: dict[str, dict[str, str]] = {}
    for eid in event_ids:
        ev = events.get(eid)
        if ev:
            event_documents[eid] = {
                r.doc_id: r.content for r in build_documents_for_event(ev, cases)
            }

    results: dict[str, Any] = {}
    print(f"{'variant':<33}{'edges':>7}{'corrob':>8}{'ECR_all':>9}"
          f"{'ECR_sup':>9}{'CDF%':>7}")
    for variant in VARIANTS:
        tot = corrob = supported = 0
        grp_total = grp_multi = 0
        for eid in event_ids:
            sg = _load_subgraph(variant, eid)
            if sg is None:
                continue
            edges = _semantic_edges(sg)
            tot += len(edges)

            group_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
            for edge in edges:
                doc_ids = _edge_doc_ids(edge)
                if len(doc_ids) >= 2:
                    corrob += 1
                ok, _ = _evidence_fully_supported(edge, event_documents.get(eid, {}))
                if ok:
                    supported += 1
                s = _normalise_text(edge.get("subject_name", ""))
                p = edge.get("predicate", "")
                if s and p:
                    group_docs[(s, p)].update(doc_ids)

            for doc_ids in group_docs.values():
                grp_total += 1
                if len(doc_ids) >= 2:
                    grp_multi += 1

        ecr_all = corrob / tot * 100 if tot else 0.0
        ecr_sup = corrob / supported * 100 if supported else 0.0
        cdf = grp_multi / grp_total * 100 if grp_total else 0.0
        results[variant] = {
            "semantic_edges": tot,
            "corroborated_edges": corrob,
            "evidence_supported_edges": supported,
            "ECR_all": round(ecr_all, 1),
            "ECR_sup": round(ecr_sup, 1),
            "groups": grp_total,
            "multi_doc_groups": grp_multi,
            "CDF": round(cdf, 1),
        }
        print(f"{VARIANT_LABELS.get(variant, variant):<33}"
              f"{tot:>7}{corrob:>8}{ecr_all:>9.1f}{ecr_sup:>9.1f}{cdf:>7.1f}")

    Path(RESULT_FILE).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
