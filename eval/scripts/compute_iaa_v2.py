"""计算更适合本评测任务的 IAA 指标.

Level 1 (实体抽取): Inter-Annotator F1
Level 2 (风险链评分): Gwet's AC2 + Krippendorff's Alpha (ordinal)
Level 3 (推理字段评分): Gwet's AC2 + Krippendorff's Alpha (ordinal)
"""

from __future__ import annotations

import json
import sys
import io
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL_DIR = Path(__file__).resolve().parents[1] / "annotations"


def load_json(path: str) -> list[dict]:
    with open(EVAL_DIR / path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_event_id(task: dict) -> str:
    return task.get("data", {}).get("event_id", "")


# ============================================================
# Inter-Annotator F1 (Level 1)
# ============================================================

def compute_ia_f1_level1() -> dict:
    print("=" * 80)
    print("Level 1: Inter-Annotator F1")
    print("=" * 80)

    results = {}

    for pair_name, fa_name, fb_name in [
        ("A26 vs A44", "level1_entity/annotator_1.json", "level1_entity/annotator_3.json"),
        ("A41 vs A45", "level1_entity/annotator_2.json", "level1_entity/annotator_4.json"),
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

        tp = fp = fn = 0
        by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for eid in common_eids:
            ea = eid_to_a[eid]
            eb = eid_to_b[eid]
            matched = ea & eb
            only_a = ea - eb
            only_b = eb - ea

            tp += len(matched)
            fp += len(only_a)
            fn += len(only_b)

            for e in matched:
                by_type[e[1]]["tp"] += 1
            for e in only_a:
                by_type[e[1]]["fp"] += 1
            for e in only_b:
                by_type[e[1]]["fn"] += 1

        p_a = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_a = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_a = 2 * p_a * r_a / (p_a + r_a) if (p_a + r_a) > 0 else 0

        p_b = tp / (tp + fn) if (tp + fn) > 0 else 0
        r_b = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_b = 2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) > 0 else 0

        mean_f1 = (f1_a + f1_b) / 2

        print(f"\n  {pair_name}:")
        print(f"    以A为gold: P={p_a:.4f}, R={r_a:.4f}, F1={f1_a:.4f}")
        print(f"    以B为gold: P={p_b:.4f}, R={r_b:.4f}, F1={f1_b:.4f}")
        print(f"    Inter-Annotator F1 (mean) = {mean_f1:.4f}")

        print(f"\n    按实体类型:")
        print(f"    {'Type':<16} {'F1(A→B)':<10} {'F1(B→A)':<10} {'Mean F1':<10}")
        print(f"    {'-'*50}")

        type_f1s = {}
        for etype in sorted(by_type.keys()):
            d = by_type[etype]
            tp_t = d["tp"]
            fp_t = d["fp"]
            fn_t = d["fn"]

            p_ab = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            r_ab = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_ab = 2 * p_ab * r_ab / (p_ab + r_ab) if (p_ab + r_ab) > 0 else 0

            p_ba = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            r_ba = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            f1_ba = 2 * p_ba * r_ba / (p_ba + r_ba) if (p_ba + r_ba) > 0 else 0

            mean_t = (f1_ab + f1_ba) / 2
            print(f"    {etype:<16} {f1_ab:<10.4f} {f1_ba:<10.4f} {mean_t:<10.4f}")
            type_f1s[etype] = mean_t

        results[pair_name] = {
            "mean_f1": mean_f1,
            "f1_a_to_b": f1_a,
            "f1_b_to_a": f1_b,
            "type_f1s": type_f1s,
        }

    return results


# ============================================================
# Gwet's AC2 (Level 2 & 3)
# ============================================================

def gwet_ac2(ratings_a: list[int], ratings_b: list[int], n_categories: int = 3) -> float:
    """Gwet's AC2 for two raters.

    AC2 uses a chance agreement model based on the probability that
    a rater classifies a subject into a given category, making it
    robust to the Kappa paradox.
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    po = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    category_counts = defaultdict(int)
    for r in ratings_a + ratings_b:
        category_counts[r] += 1

    total_ratings = 2 * n
    pi = {c: category_counts[c] / total_ratings for c in range(1, n_categories + 1)}

    pe_gwet = (2 * sum(pi[c] * (1 - pi[c]) for c in range(1, n_categories + 1))) / (n_categories - 1) if n_categories > 1 else 0

    if (1 - pe_gwet) == 0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe_gwet) / (1 - pe_gwet)


def gwet_ac2_weighted(ratings_a: list[int], ratings_b: list[int], n_categories: int = 3) -> float:
    """Gwet's AC2 with quadratic weights for ordinal data."""
    n = len(ratings_a)
    if n == 0:
        return 0.0

    max_diff = n_categories - 1

    weighted_po = 0.0
    for a, b in zip(ratings_a, ratings_b):
        diff = abs(a - b)
        w = 1 - (diff / max_diff) ** 2
        weighted_po += w
    weighted_po /= n

    category_counts = defaultdict(int)
    for r in ratings_a + ratings_b:
        category_counts[r] += 1

    total_ratings = 2 * n
    pi = {c: category_counts[c] / total_ratings for c in range(1, n_categories + 1)}

    weighted_pe = 0.0
    for c1 in range(1, n_categories + 1):
        for c2 in range(1, n_categories + 1):
            diff = abs(c1 - c2)
            w = 1 - (diff / max_diff) ** 2
            weighted_pe += pi[c1] * pi[c2] * w

    if (1 - weighted_pe) == 0:
        return 1.0 if weighted_po == 1.0 else 0.0

    return (weighted_po - weighted_pe) / (1 - weighted_pe)


# ============================================================
# Krippendorff's Alpha (ordinal)
# ============================================================

def krippendorff_alpha_ordinal(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Krippendorff's Alpha with ordinal distance for two raters.

    Uses the standard formulation:
    Alpha = 1 - (Do / De)
    where Do = observed disagreement, De = expected disagreement by chance.
    Ordinal distance: (rank_diff)^2 / (n_categories - 1)^2
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    all_ratings = ratings_a + ratings_b
    unique_vals = sorted(set(all_ratings))
    if len(unique_vals) <= 1:
        return 1.0

    rank_map = {v: i for i, v in enumerate(unique_vals)}
    n_levels = len(unique_vals)
    max_rank_diff = n_levels - 1

    def ord_dist(a, b):
        return ((rank_map[a] - rank_map[b]) / max_rank_diff) ** 2

    do = sum(ord_dist(a, b) for a, b in zip(ratings_a, ratings_b)) / n

    value_counts = defaultdict(int)
    for r in all_ratings:
        value_counts[r] += 1
    total = len(all_ratings)

    de = 0.0
    for v1 in unique_vals:
        for v2 in unique_vals:
            if v1 != v2:
                de += value_counts[v1] * value_counts[v2] * ord_dist(v1, v2)
    de /= (total * (total - 1)) if total > 1 else 1

    if de == 0:
        return 1.0 if do == 0 else 0.0

    return 1 - do / de


# ============================================================
# Level 2 & 3: Compute all metrics
# ============================================================

def compute_rating_metrics(level_name: str, prefix: str, fields: list[str], field_display: dict) -> dict:
    print(f"\n{'=' * 80}")
    print(f"{level_name}: Gwet's AC2 + Krippendorff's Alpha")
    print(f"{'=' * 80}")

    results = {}

    for pair_name, fa_name, fb_name in [
        ("A26 vs A44", f"{prefix}1.json", f"{prefix}3.json"),
        ("A41 vs A45", f"{prefix}2.json", f"{prefix}4.json"),
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

        print(f"\n  {'Field':<20} {'Po':<8} {'Kappa':<8} {'AC2':<8} {'AC2(w)':<8} {'Alpha':<8}")
        print(f"  {'-'*65}")

        overall_a = []
        overall_b = []
        field_metrics = {}

        for field in fields:
            ratings_a = []
            ratings_b = []
            for eid in common_eids:
                ra = eid_to_a[eid].get(field)
                rb = eid_to_b[eid].get(field)
                if ra is not None and rb is not None:
                    ratings_a.append(ra)
                    ratings_b.append(rb)

            if not ratings_a:
                continue

            po = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / len(ratings_a)

            n = len(ratings_a)
            matrix = [[0]*3 for _ in range(3)]
            for a, b in zip(ratings_a, ratings_b):
                ia = 3 - a
                ib = 3 - b
                if 0 <= ia < 3 and 0 <= ib < 3:
                    matrix[ia][ib] += 1

            row_sums = [sum(matrix[i]) for i in range(3)]
            col_sums = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
            total = sum(sum(row) for row in matrix)
            pe_kappa = sum(row_sums[i] * col_sums[i] for i in range(3)) / (total * total) if total > 0 else 0
            kappa = (po - pe_kappa) / (1 - pe_kappa) if (1 - pe_kappa) > 0 else 0

            ac2 = gwet_ac2(ratings_a, ratings_b)
            ac2w = gwet_ac2_weighted(ratings_a, ratings_b)
            alpha = krippendorff_alpha_ordinal(ratings_a, ratings_b)

            display = field_display.get(field, field)
            print(f"  {display:<20} {po:<8.3f} {kappa:<8.3f} {ac2:<8.3f} {ac2w:<8.3f} {alpha:<8.3f}")

            field_metrics[display] = {
                "po": po, "kappa": kappa, "ac2": ac2, "ac2w": ac2w, "alpha": alpha,
            }

            overall_a.extend(ratings_a)
            overall_b.extend(ratings_b)

        po_all = sum(1 for a, b in zip(overall_a, overall_b) if a == b) / len(overall_a) if overall_a else 0
        ac2_all = gwet_ac2(overall_a, overall_b)
        ac2w_all = gwet_ac2_weighted(overall_a, overall_b)
        alpha_all = krippendorff_alpha_ordinal(overall_a, overall_b)

        n_all = len(overall_a)
        matrix_all = [[0]*3 for _ in range(3)]
        for a, b in zip(overall_a, overall_b):
            ia = 3 - a
            ib = 3 - b
            if 0 <= ia < 3 and 0 <= ib < 3:
                matrix_all[ia][ib] += 1
        row_sums = [sum(matrix_all[i]) for i in range(3)]
        col_sums = [sum(matrix_all[i][j] for i in range(3)) for j in range(3)]
        total = sum(sum(row) for row in matrix_all)
        pe_all = sum(row_sums[i] * col_sums[i] for i in range(3)) / (total * total) if total > 0 else 0
        kappa_all = (po_all - pe_all) / (1 - pe_all) if (1 - pe_all) > 0 else 0

        print(f"  {'Overall':<20} {po_all:<8.3f} {kappa_all:<8.3f} {ac2_all:<8.3f} {ac2w_all:<8.3f} {alpha_all:<8.3f}")

        results[pair_name] = {
            "po": po_all, "kappa": kappa_all,
            "ac2": ac2_all, "ac2w": ac2w_all, "alpha": alpha_all,
            "field_metrics": field_metrics,
        }

    return results


if __name__ == "__main__":
    r1 = compute_ia_f1_level1()

    r2 = compute_rating_metrics(
        "Level 2 风险链", "level2_risk_chain/annotator_",
        ["risk_source_score", "risk_score", "consequence_score",
         "impact_score", "affected_actor_score", "risk_control_score"],
        {"risk_source_score": "RiskSource", "risk_score": "Risk",
         "consequence_score": "Consequence", "impact_score": "Impact",
         "affected_actor_score": "AffectedActor", "risk_control_score": "RiskControl"},
    )

    r3 = compute_rating_metrics(
        "Level 3 推理字段", "level3_inference/annotator_",
        ["purpose_score", "lifecycle_score", "domain_score"],
        {"purpose_score": "Purpose", "lifecycle_score": "AILifecyclePhase", "domain_score": "Domain"},
    )

    print("\n" + "=" * 80)
    print("汇总对比")
    print("=" * 80)

    print("\nLevel 1 Inter-Annotator F1:")
    for pair, data in r1.items():
        print(f"  {pair}: Mean F1 = {data['mean_f1']:.4f}")

    print("\nLevel 2 指标对比:")
    print(f"  {'配对':<14} {'Po':<8} {'Kappa':<8} {'AC2':<8} {'AC2(w)':<8} {'Alpha':<8}")
    for pair, data in r2.items():
        print(f"  {pair:<14} {data['po']:<8.3f} {data['kappa']:<8.3f} {data['ac2']:<8.3f} {data['ac2w']:<8.3f} {data['alpha']:<8.3f}")

    print("\nLevel 3 指标对比:")
    print(f"  {'配对':<14} {'Po':<8} {'Kappa':<8} {'AC2':<8} {'AC2(w)':<8} {'Alpha':<8}")
    for pair, data in r3.items():
        print(f"  {pair:<14} {data['po']:<8.3f} {data['kappa']:<8.3f} {data['ac2']:<8.3f} {data['ac2w']:<8.3f} {data['alpha']:<8.3f}")

    print("\n指标解读 (Landis & Koch 参考标准):")
    print("  Cohen's Kappa:  <0.00 差, 0.00-0.20 轻微, 0.21-0.40 一般, 0.41-0.60 中等, 0.61-0.80 较好, 0.81-1.00 优秀")
    print("  Gwet's AC2:     <0.00 差, 0.00-0.20 轻微, 0.21-0.40 一般, 0.41-0.60 中等, 0.61-0.80 较好, 0.81-1.00 优秀")
    print("  Kripp. Alpha:   <0.67 不可信, 0.67-0.80 临时可信, >0.80 可信")
    print("  IA-F1:          参考常规 F1 解读: <0.5 差, 0.5-0.7 一般, 0.7-0.8 较好, >0.8 优秀")
