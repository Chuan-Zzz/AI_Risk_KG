"""计算标注员间一致性 (Cohen's Kappa).

标注数据配对关系：
  评测1 (Annotator 26) 和 评测3 (Annotator 44) 标注同一批任务
  评测2 (Annotator 41) 和 评测4 (Annotator 45) 标注同一批任务

任务匹配方式：通过 data.event_id 字段匹配
"""

from __future__ import annotations

import json
import sys
import io
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL_DIR = Path(__file__).resolve().parent / "标注结果"


def load_json(path: str) -> list[dict]:
    with open(EVAL_DIR / path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_event_id(task: dict) -> str:
    return task.get("data", {}).get("event_id", "")


def cohen_kappa(matrix: list[list[int]]) -> float:
    n = len(matrix)
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0

    po = sum(matrix[i][i] for i in range(n)) / total

    row_sums = [sum(matrix[i]) for i in range(n)]
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(n)) / (total * total)

    if (1 - pe) == 0:
        return 1.0

    return (po - pe) / (1 - pe)


def weighted_kappa_3x3(matrix: list[list[int]]) -> float:
    n = 3
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0

    weights = [[0, 1, 4], [1, 0, 1], [4, 1, 0]]

    row_sums = [sum(matrix[i]) for i in range(n)]
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]

    observed_weighted = sum(weights[i][j] * matrix[i][j] for i in range(n) for j in range(n))
    expected_weighted = sum(weights[i][j] * row_sums[i] * col_sums[j] for i in range(n) for j in range(n))

    if expected_weighted == 0:
        return 1.0

    return 1 - (observed_weighted / total) / (expected_weighted / (total * total))


def percent_agreement(matrix: list[list[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0
    return sum(matrix[i][i] for i in range(len(matrix))) / total


def compute_kappa_level1() -> dict:
    print("=" * 80)
    print("Level 1 显式实体 - Cohen's Kappa")
    print("=" * 80)

    results = {}

    for pair_name, fa_name, fb_name, ann_a, ann_b in [
        ("Pair A26 vs A44", "显式实体评测1.json", "显式实体评测3.json", 26, 44),
        ("Pair A41 vs A45", "显式实体评测2.json", "显式实体评测4.json", 41, 45),
    ]:
        fa = load_json(fa_name)
        fb = load_json(fb_name)

        eid_to_a = {}
        for t in fa:
            eid = get_event_id(t)
            entities = set()
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "labels":
                        continue
                    text = r.get("value", {}).get("text", "")
                    labels = r.get("value", {}).get("labels", [])
                    for label in labels:
                        entities.add((text, label))
            eid_to_a[eid] = entities

        eid_to_b = {}
        for t in fb:
            eid = get_event_id(t)
            entities = set()
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "labels":
                        continue
                    text = r.get("value", {}).get("text", "")
                    labels = r.get("value", {}).get("labels", [])
                    for label in labels:
                        entities.add((text, label))
            eid_to_b[eid] = entities

        common_eids = set(eid_to_a.keys()) & set(eid_to_b.keys())
        print(f"\n  {pair_name}: 共同任务数 = {len(common_eids)}")

        if not common_eids:
            print("  无共同任务，跳过")
            continue

        tp = fp = fn = tn = 0
        by_type_matrix = defaultdict(lambda: [[0, 0], [0, 0]])

        for eid in common_eids:
            ea = eid_to_a[eid]
            eb = eid_to_b[eid]
            all_entities = ea | eb

            for entity in all_entities:
                in_a = entity in ea
                in_b = entity in eb
                if in_a and in_b:
                    tp += 1
                elif in_a and not in_b:
                    fp += 1
                elif not in_a and in_b:
                    fn += 1
                else:
                    tn += 1

                etype = entity[1]
                if in_a and in_b:
                    by_type_matrix[etype][0][0] += 1
                elif in_a and not in_b:
                    by_type_matrix[etype][0][1] += 1
                elif not in_a and in_b:
                    by_type_matrix[etype][1][0] += 1
                else:
                    by_type_matrix[etype][1][1] += 1

        matrix = [[tp, fp], [fn, tn]]
        kappa = cohen_kappa(matrix)
        total = tp + fp + fn + tn
        po = (tp + tn) / total if total > 0 else 0

        print(f"  实体级配对: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Po (观察一致率) = {po:.4f}")
        print(f"  Cohen's Kappa = {kappa:.4f}")

        print(f"\n  按实体类型 Kappa:")
        print(f"  {'Entity Type':<16} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Po':<8} {'Kappa':<10}")
        print(f"  {'-'*60}")

        type_kappas = {}
        for etype in sorted(by_type_matrix.keys()):
            m = by_type_matrix[etype]
            k = cohen_kappa(m)
            t = sum(sum(row) for row in m)
            p = percent_agreement(m)
            if t > 0:
                print(f"  {etype:<16} {m[0][0]:<6} {m[0][1]:<6} {m[1][0]:<6} {m[1][1]:<6} {p:<8.4f} {k:.4f}")
            type_kappas[etype] = k

        results[pair_name] = {
            "overall_kappa": kappa,
            "overall_po": po,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "type_kappas": type_kappas,
            "common_tasks": len(common_eids),
        }

    return results


def compute_kappa_level2() -> dict:
    print("\n" + "=" * 80)
    print("Level 2 风险链 - Cohen's Kappa")
    print("=" * 80)

    results = {}

    slots = [
        "risk_source_score", "risk_score", "consequence_score",
        "impact_score", "affected_actor_score", "risk_control_score",
    ]
    slot_display = {
        "risk_source_score": "RiskSource",
        "risk_score": "Risk",
        "consequence_score": "Consequence",
        "impact_score": "Impact",
        "affected_actor_score": "AffectedActor",
        "risk_control_score": "RiskControl",
    }

    for pair_name, fa_name, fb_name in [
        ("Pair A26 vs A44", "风险链评测1.json", "风险链评测3.json"),
        ("Pair A41 vs A45", "风险链评测2.json", "风险链评测4.json"),
    ]:
        fa = load_json(fa_name)
        fb = load_json(fb_name)

        eid_to_a = {}
        for t in fa:
            eid = get_event_id(t)
            ratings = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    from_name = r.get("from_name", "")
                    rating = r.get("value", {}).get("rating", 0)
                    ratings[from_name] = rating
            eid_to_a[eid] = ratings

        eid_to_b = {}
        for t in fb:
            eid = get_event_id(t)
            ratings = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    from_name = r.get("from_name", "")
                    rating = r.get("value", {}).get("rating", 0)
                    ratings[from_name] = rating
            eid_to_b[eid] = ratings

        common_eids = set(eid_to_a.keys()) & set(eid_to_b.keys())
        print(f"\n  {pair_name}: 共同任务数 = {len(common_eids)}")

        if not common_eids:
            print("  无共同任务，跳过")
            continue

        print(f"\n  按槽位 Kappa (3x3 混淆矩阵: 行=标注员A, 列=标注员B):")
        print(f"  {'Slot':<16} {'3-3':<5} {'3-2':<5} {'3-1':<5} {'2-3':<5} {'2-2':<5} {'2-1':<5} {'1-3':<5} {'1-2':<5} {'1-1':<5} {'Po':<7} {'Kappa':<8} {'WK':<8}")
        print(f"  {'-'*95}")

        overall_matrix = [[0]*3 for _ in range(3)]
        slot_kappas = {}

        for slot in slots:
            conf_matrix = [[0]*3 for _ in range(3)]

            for eid in common_eids:
                ra = eid_to_a[eid].get(slot)
                rb = eid_to_b[eid].get(slot)
                if ra is None or rb is None:
                    continue
                ia = 3 - ra
                ib = 3 - rb
                if 0 <= ia < 3 and 0 <= ib < 3:
                    conf_matrix[ia][ib] += 1
                    overall_matrix[ia][ib] += 1

            k = cohen_kappa(conf_matrix)
            wk = weighted_kappa_3x3(conf_matrix)
            po = percent_agreement(conf_matrix)
            flat = [conf_matrix[i][j] for i in range(3) for j in range(3)]
            print(f"  {slot_display[slot]:<16} " + " ".join(f"{v:<5}" for v in flat) + f" {po:<7.3f} {k:.4f}  {wk:.4f}")
            slot_kappas[slot_display[slot]] = {"kappa": k, "weighted_kappa": wk, "po": po}

        overall_k = cohen_kappa(overall_matrix)
        overall_wk = weighted_kappa_3x3(overall_matrix)
        overall_po = percent_agreement(overall_matrix)
        print(f"  {'Overall':<16} " + " ".join(f"{overall_matrix[i][j]:<5}" for i in range(3) for j in range(3)) + f" {overall_po:<7.3f} {overall_k:.4f}  {overall_wk:.4f}")

        results[pair_name] = {
            "overall_kappa": overall_k,
            "overall_weighted_kappa": overall_wk,
            "overall_po": overall_po,
            "slot_kappas": slot_kappas,
            "common_tasks": len(common_eids),
        }

    return results


def compute_kappa_level3() -> dict:
    print("\n" + "=" * 80)
    print("Level 3 推理字段 - Cohen's Kappa")
    print("=" * 80)

    results = {}

    fields = ["purpose_score", "lifecycle_score", "domain_score"]
    field_display = {
        "purpose_score": "Purpose",
        "lifecycle_score": "AILifecyclePhase",
        "domain_score": "Domain",
    }

    for pair_name, fa_name, fb_name in [
        ("Pair A26 vs A44", "推理字段评测1.json", "推理字段评测3.json"),
        ("Pair A41 vs A45", "推理字段评测2.json", "推理字段评测4.json"),
    ]:
        fa = load_json(fa_name)
        fb = load_json(fb_name)

        eid_to_a = {}
        for t in fa:
            eid = get_event_id(t)
            ratings = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    from_name = r.get("from_name", "")
                    rating = r.get("value", {}).get("rating", 0)
                    ratings[from_name] = rating
            eid_to_a[eid] = ratings

        eid_to_b = {}
        for t in fb:
            eid = get_event_id(t)
            ratings = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    from_name = r.get("from_name", "")
                    rating = r.get("value", {}).get("rating", 0)
                    ratings[from_name] = rating
            eid_to_b[eid] = ratings

        common_eids = set(eid_to_a.keys()) & set(eid_to_b.keys())
        print(f"\n  {pair_name}: 共同任务数 = {len(common_eids)}")

        if not common_eids:
            print("  无共同任务，跳过")
            continue

        print(f"\n  按字段 Kappa (3x3 混淆矩阵: 行=标注员A, 列=标注员B):")
        print(f"  {'Field':<20} {'3-3':<5} {'3-2':<5} {'3-1':<5} {'2-3':<5} {'2-2':<5} {'2-1':<5} {'1-3':<5} {'1-2':<5} {'1-1':<5} {'Po':<7} {'Kappa':<8} {'WK':<8}")
        print(f"  {'-'*100}")

        overall_matrix = [[0]*3 for _ in range(3)]
        field_kappas = {}

        for field in fields:
            conf_matrix = [[0]*3 for _ in range(3)]

            for eid in common_eids:
                ra = eid_to_a[eid].get(field)
                rb = eid_to_b[eid].get(field)
                if ra is None or rb is None:
                    continue
                ia = 3 - ra
                ib = 3 - rb
                if 0 <= ia < 3 and 0 <= ib < 3:
                    conf_matrix[ia][ib] += 1
                    overall_matrix[ia][ib] += 1

            k = cohen_kappa(conf_matrix)
            wk = weighted_kappa_3x3(conf_matrix)
            po = percent_agreement(conf_matrix)
            flat = [conf_matrix[i][j] for i in range(3) for j in range(3)]
            print(f"  {field_display[field]:<20} " + " ".join(f"{v:<5}" for v in flat) + f" {po:<7.3f} {k:.4f}  {wk:.4f}")
            field_kappas[field_display[field]] = {"kappa": k, "weighted_kappa": wk, "po": po}

        overall_k = cohen_kappa(overall_matrix)
        overall_wk = weighted_kappa_3x3(overall_matrix)
        overall_po = percent_agreement(overall_matrix)
        print(f"  {'Overall':<20} " + " ".join(f"{overall_matrix[i][j]:<5}" for i in range(3) for j in range(3)) + f" {overall_po:<7.3f} {overall_k:.4f}  {overall_wk:.4f}")

        results[pair_name] = {
            "overall_kappa": overall_k,
            "overall_weighted_kappa": overall_wk,
            "overall_po": overall_po,
            "field_kappas": field_kappas,
            "common_tasks": len(common_eids),
        }

    return results


if __name__ == "__main__":
    r1 = compute_kappa_level1()
    r2 = compute_kappa_level2()
    r3 = compute_kappa_level3()

    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    print("\nLevel 1 Cohen's Kappa:")
    for pair, data in r1.items():
        print(f"  {pair}: Kappa={data['overall_kappa']:.4f}, Po={data['overall_po']:.4f}, 共同任务={data['common_tasks']}")

    print("\nLevel 2 Cohen's Kappa:")
    for pair, data in r2.items():
        print(f"  {pair}: Kappa={data['overall_kappa']:.4f}, WK={data['overall_weighted_kappa']:.4f}, Po={data['overall_po']:.4f}, 共同任务={data['common_tasks']}")

    print("\nLevel 3 Cohen's Kappa:")
    for pair, data in r3.items():
        print(f"  {pair}: Kappa={data['overall_kappa']:.4f}, WK={data['overall_weighted_kappa']:.4f}, Po={data['overall_po']:.4f}, 共同任务={data['common_tasks']}")
