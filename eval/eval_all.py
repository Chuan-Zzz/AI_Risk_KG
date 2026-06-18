"""Evaluation metrics from Label Studio annotation results."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ANNOT_DIR = Path("eval/标注结果")
OUTPUT_DIR = Path("output")


def load_all_tasks(prefix: str) -> list[dict]:
    """Load all annotation tasks from 评测1-4.json files."""
    tasks = []
    for suffix in ['1', '2', '3', '4']:
        path = ANNOT_DIR / f"{prefix}{suffix}.json"
        if path.exists():
            with open(path, encoding='utf-8') as f:
                tasks.extend(json.load(f))
    return tasks


def load_subgraph(event_id: str) -> dict | None:
    p = OUTPUT_DIR / event_id / "event_subgraph.json"
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


# ── Level 1: Entity P/R/F1 ──────────────────────────────────────────────

def eval_level1():
    tasks = load_all_tasks("显式实体评测")
    print(f"\n{'='*60}")
    print("Table 1: Entity Extraction Results")
    print(f"{'='*60}")

    tp_by_type: dict[str, int] = defaultdict(int)
    fp_by_type: dict[str, int] = defaultdict(int)
    fn_by_type: dict[str, int] = defaultdict(int)

    seen_events = set()
    for task in tasks:
        event_id = task['data']['event_id']
        seen_events.add(event_id)

        annotations = task.get('annotations', [])
        if not annotations:
            continue

        result = annotations[0]['result']

        gold_entities = set()
        pred_entities = set()

        for r in result:
            if r['type'] != 'labels':
                continue

            value = r['value']
            text = value.get('text', '')
            labels = value.get('labels', [])
            origin = r.get('origin', '')
            normalized = normalize_name(text)

            for label in labels:
                entity_key = (normalized, label)
                if origin in ('manual', 'prediction-changed'):
                    gold_entities.add(entity_key)
                else:
                    pred_entities.add(entity_key)

        # For entities with both gold and prediction, they're correct (TP)
        matched = gold_entities & pred_entities

        # Count TP, FP, FN per type
        type_counter: dict[str, dict] = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for key in matched:
            etype = key[1]
            type_counter[etype]['tp'] += 1

        for key in pred_entities - gold_entities:
            etype = key[1]
            type_counter[etype]['fp'] += 1

        for key in gold_entities - pred_entities:
            etype = key[1]
            type_counter[etype]['fn'] += 1

        for etype, counts in type_counter.items():
            tp_by_type[etype] += counts['tp']
            fp_by_type[etype] += counts['fp']
            fn_by_type[etype] += counts['fn']

    # Calculate P/R/F1 per type
    all_types = sorted(set(list(tp_by_type.keys()) + list(fp_by_type.keys()) + list(fn_by_type.keys())))
    total_tp, total_fp, total_fn = 0, 0, 0
    results = []

    for etype in all_types:
        tp = tp_by_type[etype]
        fp = fp_by_type[etype]
        fn = fn_by_type[etype]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results.append((etype, tp, fp, fn, p, r, f1))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    print(f"\nEvents evaluated: {len(seen_events)}")
    print(f"\n{'Entity Type':<15} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for etype, tp, fp, fn, p, r, f1 in results:
        print(f"{etype:<15} {tp:>4} {fp:>4} {fn:>4} {p:>9.1%} {r:>9.1%} {f1:>9.1%}")
    print("-" * 60)
    print(f"{'Overall':<15} {total_tp:>4} {total_fp:>4} {total_fn:>4} {overall_p:>9.1%} {overall_r:>9.1%} {overall_f1:>9.1%}")

    return {
        "overall": {"precision": overall_p, "recall": overall_r, "f1": overall_f1},
        "by_type": {r[0]: {"precision": r[4], "recall": r[5], "f1": r[6]} for r in results},
    }


# ── Level 2: Risk Chain Accuracy ────────────────────────────────────────

def eval_level2():
    tasks = load_all_tasks("风险链评测")
    print(f"\n{'='*60}")
    print("Table 2: Risk Chain Accuracy")
    print(f"{'='*60}")

    slot_scores: dict[str, list[int]] = defaultdict(list)
    seen_events = set()

    for task in tasks:
        event_id = task['data']['event_id']
        seen_events.add(event_id)

        annotations = task.get('annotations', [])
        if not annotations:
            continue

        result = annotations[0]['result']
        for r in result:
            if r['type'] == 'rating':
                from_name = r['from_name']
                rating = r['value']['rating']
                slot_scores[from_name].append(rating)

    slot_order = ['risk_source_score', 'risk_score', 'consequence_score',
                  'impact_score', 'affected_actor_score', 'risk_control_score']
    slot_labels = ['RiskSource', 'Risk', 'Consequence', 'Impact', 'AffectedActor', 'RiskControl']

    total_correct, total_partial, total_incorrect, total = 0, 0, 0, 0
    results = []

    for from_name, label in zip(slot_order, slot_labels):
        scores = slot_scores.get(from_name, [])
        if not scores:
            continue
        correct = sum(1 for s in scores if s == 3)
        partial = sum(1 for s in scores if s == 2)
        incorrect = sum(1 for s in scores if s == 1)
        n = len(scores)
        accuracy = (correct + 0.5 * partial) / n if n > 0 else 0
        results.append((label, correct, partial, incorrect, n, accuracy))
        total_correct += correct
        total_partial += partial
        total_incorrect += incorrect
        total += n

    overall_acc = (total_correct + 0.5 * total_partial) / total if total > 0 else 0

    print(f"\nEvents evaluated: {len(seen_events)}")
    print(f"\n{'Slot':<15} {'Correct':>8} {'Partial':>8} {'Incorrect':>10} {'N':>4} {'Accuracy':>10}")
    print("-" * 60)
    for label, c, p, i, n, acc in results:
        print(f"{label:<15} {c:>8} {p:>8} {i:>10} {n:>4} {acc:>9.1%}")
    print("-" * 60)
    print(f"{'Overall':<15} {total_correct:>8} {total_partial:>8} {total_incorrect:>10} {total:>4} {overall_acc:>9.1%}")

    return {
        "overall_accuracy": overall_acc,
        "by_slot": {r[0]: {"correct": r[1], "partial": r[2], "incorrect": r[3], "accuracy": r[5]} for r in results},
    }


# ── Level 3: Inference Accuracy ─────────────────────────────────────────

def eval_level3():
    tasks = load_all_tasks("推理字段评测")
    print(f"\n{'='*60}")
    print("Table 2b: Inference Field Accuracy")
    print(f"{'='*60}")

    field_scores: dict[str, list[int]] = defaultdict(list)
    seen_events = set()

    for task in tasks:
        event_id = task['data']['event_id']
        seen_events.add(event_id)

        annotations = task.get('annotations', [])
        if not annotations:
            continue

        result = annotations[0]['result']
        for r in result:
            if r['type'] == 'rating':
                from_name = r['from_name']
                rating = r['value']['rating']
                field_scores[from_name].append(rating)

    field_order = ['purpose_score', 'lifecycle_score', 'domain_score']
    field_labels = ['Purpose', 'AILifecyclePhase', 'Domain']

    total_correct, total_partial, total_incorrect, total = 0, 0, 0, 0
    results = []

    for from_name, label in zip(field_order, field_labels):
        scores = field_scores.get(from_name, [])
        if not scores:
            continue
        correct = sum(1 for s in scores if s == 3)
        partial = sum(1 for s in scores if s == 2)
        incorrect = sum(1 for s in scores if s == 1)
        n = len(scores)
        accuracy = (correct + 0.5 * partial) / n if n > 0 else 0
        results.append((label, correct, partial, incorrect, n, accuracy))
        total_correct += correct
        total_partial += partial
        total_incorrect += incorrect
        total += n

    overall_acc = (total_correct + 0.5 * total_partial) / total if total > 0 else 0

    print(f"\nEvents evaluated: {len(seen_events)}")
    print(f"\n{'Field':<18} {'Correct':>8} {'Partial':>8} {'Incorrect':>10} {'N':>4} {'Accuracy':>10}")
    print("-" * 62)
    for label, c, p, i, n, acc in results:
        print(f"{label:<18} {c:>8} {p:>8} {i:>10} {n:>4} {acc:>9.1%}")
    print("-" * 62)
    print(f"{'Overall':<18} {total_correct:>8} {total_partial:>8} {total_incorrect:>10} {total:>4} {overall_acc:>9.1%}")

    return {
        "overall_accuracy": overall_acc,
        "by_field": {r[0]: {"correct": r[1], "partial": r[2], "incorrect": r[3], "accuracy": r[5]} for r in results},
    }


# ── Level 4: Graph Reliability (auto from event_subgraph.json) ──────────

def eval_level4():
    print(f"\n{'='*60}")
    print("Table 4: Graph Reliability")
    print(f"{'='*60}")

    total_nodes, nodes_with_evidence = 0, 0
    total_edges, edges_with_evidence = 0, 0
    total_support = 0
    events_count = 0
    gold = json.load(open("data/gold_standard_100_events.json", encoding='utf-8'))
    gold_ids = {e['event_id'] for e in gold.get('gold_standard', [])}

    for event_id in sorted(gold_ids):
        sg = load_subgraph(event_id)
        if not sg:
            continue
        events_count += 1

        for node in sg.get("nodes", []):
            if node.get("entity_type") == "AIRiskIncident":
                continue
            total_nodes += 1
            evidence = node.get("evidence", [])
            if evidence and any(ev.get("evidence_sentence", "") for ev in evidence):
                nodes_with_evidence += 1
            total_support += node.get("support_count", 1)

        for edge in sg.get("edges", []):
            total_edges += 1
            evidence = edge.get("evidence", [])
            if evidence and any(ev.get("evidence_sentence", "") if isinstance(ev, dict) else ev for ev in evidence):
                edges_with_evidence += 1
            total_support += edge.get("support_count", 1)

    node_cov = nodes_with_evidence / total_nodes if total_nodes > 0 else 0
    edge_cov = edges_with_evidence / total_edges if total_edges > 0 else 0
    overall_cov = (nodes_with_evidence + edges_with_evidence) / (total_nodes + total_edges) if (total_nodes + total_edges) > 0 else 0
    avg_support = total_support / (total_nodes + total_edges) if (total_nodes + total_edges) > 0 else 0

    print(f"\nEvents evaluated: {events_count}")
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Node Evidence Coverage':<30} {node_cov:>9.1%}")
    print(f"{'Edge Evidence Coverage':<30} {edge_cov:>9.1%}")
    print(f"{'Overall Evidence Coverage':<30} {overall_cov:>9.1%}")
    print(f"{'Avg Support Strength':<30} {avg_support:>9.2f}")

    return {
        "node_evidence_coverage": node_cov,
        "edge_evidence_coverage": edge_cov,
        "overall_evidence_coverage": overall_cov,
        "avg_support_strength": avg_support,
    }


def main():
    print("AI Risk Knowledge Graph — Evaluation Metrics")
    print("=" * 60)

    r1 = eval_level1()
    r2 = eval_level2()
    r3 = eval_level3()
    r4 = eval_level4()

    # Save all results
    results = {"level1_entity": r1, "level2_risk_chain": r2, "level3_inference": r3, "level4_graph_reliability": r4}
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
