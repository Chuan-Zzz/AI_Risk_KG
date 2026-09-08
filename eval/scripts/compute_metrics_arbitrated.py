"""基于仲裁结果更新评测指标.

策略:
  Level 1: 使用仲裁文件 (显式实体评测仲裁1.json + 显式实体评测仲裁2.json) 计算
           并通过 ann.prediction.result 恢复被标注员删除的预测 (FP)
  Level 2: 标注一致的任务用原文件，不一致的用仲裁结果替换
  Level 3: 标注一致的任务用原文件，不一致的用仲裁结果替换

Level 2/3 的"一致"判断: 两对标注员 (A26 vs A44, A41 vs A45) 评分完全相同时算一致

标注分工:
  文件1 (A26) 与 文件3 (A44) 标注前 50 个事件 (pair 1)
  文件2 (A41) 与 文件4 (A45) 标注后 50 个事件 (pair 2)
  两个 pair 的事件互不重叠 (并集 = 100 个事件)
"""

from __future__ import annotations

import json
import sys
import io
import copy
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL_DIR = Path(__file__).resolve().parents[1] / "annotations"
ARBITRATION_DIR = Path(__file__).resolve().parents[1] / "annotations"


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_event_id(task: dict) -> str:
    return task.get("data", {}).get("event_id", "")


# ============================================================
# 常量定义 (与 compute_metrics.py 保持一致)
# ============================================================

STAKEHOLDER_SUBTYPES = {
    "AIDeveloper", "AIProvider", "AIDeployer", "AIUser",
    "Regulator", "AffectedActor",
}

ENTITY_TYPE_ORDER = [
    "AISystem", "AIModel", "AITechnique", "AICapability",
    "AIDeveloper", "AIProvider", "AIDeployer", "AIUser",
    "Regulator", "AffectedActor", "Stakeholder",
    "Regulation", "Standard",
]

ENTITY_TYPE_ORDER_DISPLAY = [
    "AISystem", "AIModel", "AITechnique", "AICapability",
    "AIDeveloper", "AIProvider", "AIDeployer", "AIUser",
    "Regulator", "AffectedActor", "Stakeholder",
    "Stakeholder(agg)",
    "Regulation", "Standard",
]

RISK_CHAIN_SLOTS = [
    "risk_source_score", "risk_score", "consequence_score",
    "impact_score", "affected_actor_score", "risk_control_score",
]

RISK_CHAIN_DISPLAY = {
    "risk_source_score": "RiskSource",
    "risk_score": "Risk",
    "consequence_score": "Consequence",
    "impact_score": "Impact",
    "affected_actor_score": "AffectedActor",
    "risk_control_score": "RiskControl",
}

INFERENCE_FIELDS = ["purpose_score", "lifecycle_score", "domain_score"]

INFERENCE_DISPLAY = {
    "purpose_score": "Purpose",
    "lifecycle_score": "AILifecyclePhase",
    "domain_score": "Domain",
}


# ============================================================
# Level 1: 使用仲裁文件计算 (含 FP 恢复)
# ============================================================

def _recover_fp(task: dict) -> list[dict]:
    """从 annotation.prediction.result 恢复被标注员删除的预测 (FP).

    Label Studio 在标注员删除预测时, 不会在 annotation.result 中保留记录,
    但 ann.prediction.result 保存了导入时的完整预测列表.
    通过比对 prediction.result 的 id 和 annotation.result 中 origin=prediction/
    prediction-changed 的 id, 找出被删除的预测.
    """
    ann = task.get("annotations", [{}])[0] if task.get("annotations") else {}
    pred_obj = ann.get("prediction")
    if not pred_obj or not isinstance(pred_obj, dict):
        return []

    pred_results = pred_obj.get("result", [])
    kept_ids = set()
    for r in ann.get("result", []):
        if r.get("origin", "manual") in ("prediction", "prediction-changed"):
            kept_ids.add(r.get("id"))

    return [r for r in pred_results if r.get("id") not in kept_ids]


def calc_level1_arbitrated() -> dict:
    print("=" * 80)
    print("Level 1: 基于仲裁结果计算指标 (含 FP 恢复)")
    print("=" * 80)

    # 加载仲裁文件
    arb1 = load_json(ARBITRATION_DIR / "level1_entity/arbitration_1.json")
    arb2 = load_json(ARBITRATION_DIR / "level1_entity/arbitration_2.json")
    data = arb1 + arb2

    print(f"  仲裁1: {len(arb1)} 个任务, 仲裁2: {len(arb2)} 个任务, 合计: {len(data)}")

    by_type = defaultdict(lambda: {"tp_exact": 0, "tp_partial": 0, "fn": 0, "fp": 0})
    total = {"tp_exact": 0, "tp_partial": 0, "fn": 0, "fp": 0}

    for task in data:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "labels":
                    continue

                origin = result.get("origin", "manual")
                labels = result.get("value", {}).get("labels", [])
                primary_label = labels[0] if labels else "Unknown"

                if origin == "prediction":
                    by_type[primary_label]["tp_exact"] += 1
                    total["tp_exact"] += 1
                elif origin == "prediction-changed":
                    by_type[primary_label]["tp_partial"] += 1
                    total["tp_partial"] += 1
                elif origin == "manual":
                    by_type[primary_label]["fn"] += 1
                    total["fn"] += 1

            # 恢复被删除的预测 (FP)
            for fp_result in _recover_fp(task):
                if fp_result.get("type") != "labels":
                    continue
                labels = fp_result.get("value", {}).get("labels", [])
                primary_label = labels[0] if labels else "Unknown"
                by_type[primary_label]["fp"] += 1
                total["fp"] += 1

    print(f"  TP(exact)={total['tp_exact']}, TP(partial)={total['tp_partial']}, "
          f"FN={total['fn']}, FP={total['fp']}")

    results = {}
    for etype in ENTITY_TYPE_ORDER:
        d = by_type[etype]
        tp_exact = d["tp_exact"]
        tp_partial = d["tp_partial"]
        fn = d["fn"]
        fp = d["fp"]

        tp_weighted = tp_exact + 0.5 * tp_partial
        gold_total = tp_exact + tp_partial + fn
        pred_total = tp_exact + tp_partial + fp  # FP 计入 Precision 分母

        precision = tp_weighted / pred_total if pred_total > 0 else 0.0
        recall = tp_weighted / gold_total if gold_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[etype] = {
            "tp_exact": tp_exact,
            "tp_partial": tp_partial,
            "fn": fn,
            "fp": fp,
            "gold_total": gold_total,
            "pred_total": pred_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Stakeholder(agg)
    agg_d = defaultdict(int)
    for sub in STAKEHOLDER_SUBTYPES | {"Stakeholder"}:
        for k in ("tp_exact", "tp_partial", "fn", "fp"):
            agg_d[k] += by_type[sub][k]
    tp_exact = agg_d["tp_exact"]
    tp_partial = agg_d["tp_partial"]
    fn = agg_d["fn"]
    fp = agg_d["fp"]
    tp_weighted = tp_exact + 0.5 * tp_partial
    gold_total = tp_exact + tp_partial + fn
    pred_total = tp_exact + tp_partial + fp
    precision = tp_weighted / pred_total if pred_total > 0 else 0.0
    recall = tp_weighted / gold_total if gold_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    results["Stakeholder(agg)"] = {
        "tp_exact": tp_exact, "tp_partial": tp_partial, "fn": fn, "fp": fp,
        "gold_total": gold_total, "pred_total": pred_total,
        "precision": precision, "recall": recall, "f1": f1,
    }

    # Overall
    tp_exact = total["tp_exact"]
    tp_partial = total["tp_partial"]
    fn = total["fn"]
    fp = total["fp"]
    tp_weighted = tp_exact + 0.5 * tp_partial
    gold_total = tp_exact + tp_partial + fn
    pred_total = tp_exact + tp_partial + fp
    overall_precision = tp_weighted / pred_total if pred_total > 0 else 0.0
    overall_recall = tp_weighted / gold_total if gold_total > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0.0
    )
    results["Overall"] = {
        "tp_exact": tp_exact, "tp_partial": tp_partial, "fn": fn, "fp": fp,
        "gold_total": gold_total, "pred_total": pred_total,
        "precision": overall_precision, "recall": overall_recall, "f1": overall_f1,
    }

    return results


# ============================================================
# Level 2/3: 合并原文件 + 仲裁结果
# ============================================================

def build_arbitrated_data(level_name: str, prefix: str, fields: list[str],
                          arbitration_file: str) -> list[dict]:
    """合并原始标注数据和仲裁结果.

    策略:
      1. 加载4个原始标注文件 (1-4)
      2. 找出两对标注员 (1vs3, 2vs4) 评分不一致的任务
      3. 不一致的任务用仲裁结果替换
      4. 一致的任务保留原始标注
    """
    print(f"\n{'=' * 80}")
    print(f"{level_name}: 合并原始标注 + 仲裁结果")
    print(f"{'=' * 80}")

    # 加载仲裁结果
    arb_data = load_json(ARBITRATION_DIR / arbitration_file)
    arb_by_eid = {}
    for t in arb_data:
        eid = get_event_id(t)
        arb_by_eid[eid] = t

    print(f"  仲裁文件: {arbitration_file}, 任务数: {len(arb_data)}")

    # 加载4个原始文件
    original_files = {}
    for i in range(1, 5):
        fname = f"{prefix}{i}.json"
        data = load_json(EVAL_DIR / fname)
        original_files[i] = {get_event_id(t): t for t in data}
        print(f"  原始文件 {fname}: {len(data)} 个任务")

    # 找出不一致的任务
    disagree_eids = set()

    # 配对1: 文件1 vs 文件3
    common_1_3 = set(original_files[1].keys()) & set(original_files[3].keys())
    for eid in common_1_3:
        ratings_1 = _extract_ratings(original_files[1][eid], fields)
        ratings_3 = _extract_ratings(original_files[3][eid], fields)
        if ratings_1 != ratings_3:
            disagree_eids.add(eid)

    # 配对2: 文件2 vs 文件4
    common_2_4 = set(original_files[2].keys()) & set(original_files[4].keys())
    for eid in common_2_4:
        ratings_2 = _extract_ratings(original_files[2][eid], fields)
        ratings_4 = _extract_ratings(original_files[4][eid], fields)
        if ratings_2 != ratings_4:
            disagree_eids.add(eid)

    print(f"  不一致任务数: {len(disagree_eids)}")

    # 构建合并数据: 使用文件1和文件2 (A26和A41的标注)
    # 一致的任务保留原标注，不一致的用仲裁结果替换
    # 注意: 文件1 (pair 1) 和文件2 (pair 2) 覆盖互不重叠的事件集
    merged = []
    replaced = 0
    missing_arb = 0

    # 文件1 (A26) 的任务: pair 1 事件
    for eid, task in original_files[1].items():
        if eid in disagree_eids:
            if eid in arb_by_eid:
                merged.append(arb_by_eid[eid])
                replaced += 1
            else:
                # 仲裁缺失,保留原标注 (pair1 有 5 个不一致事件未被仲裁)
                merged.append(task)
                missing_arb += 1
        else:
            merged.append(task)

    # 文件2 (A41) 的任务: pair 2 事件 (与 pair 1 不重叠)
    for eid, task in original_files[2].items():
        if eid in disagree_eids:
            if eid in arb_by_eid:
                merged.append(arb_by_eid[eid])
                replaced += 1
            else:
                merged.append(task)
                missing_arb += 1
        else:
            merged.append(task)

    print(f"  合并后任务数: {len(merged)}, 其中仲裁替换: {replaced}, "
          f"仲裁缺失保留原标注: {missing_arb}")

    return merged


def _extract_ratings(task: dict, fields: list[str]) -> dict[str, int]:
    """提取任务的评分字典."""
    ratings = {}
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            if r.get("type") != "rating":
                continue
            fname = r.get("from_name", "")
            if fname in fields:
                ratings[fname] = r.get("value", {}).get("rating", 0)
    return ratings


# ============================================================
# Level 2: 计算指标
# ============================================================

def calc_level2_arbitrated() -> dict:
    merged = build_arbitrated_data(
        "Level 2 风险链", "level2_risk_chain/annotator_",
        RISK_CHAIN_SLOTS, "level2_risk_chain/arbitration.json",
    )

    by_slot = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
    total = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}

    for task in merged:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "rating":
                    continue
                from_name = result.get("from_name", "")
                if from_name not in RISK_CHAIN_SLOTS:
                    continue

                rating = result.get("value", {}).get("rating", 0)
                if rating == 3:
                    by_slot[from_name]["correct"] += 1
                    total["correct"] += 1
                elif rating == 2:
                    by_slot[from_name]["partial"] += 1
                    total["partial"] += 1
                elif rating == 1:
                    by_slot[from_name]["incorrect"] += 1
                    total["incorrect"] += 1

                by_slot[from_name]["total"] += 1
                total["total"] += 1

    results = {}
    for slot_key in RISK_CHAIN_SLOTS:
        d = by_slot[slot_key]
        n = d["total"]
        accuracy = (d["correct"] + 0.5 * d["partial"]) / n if n > 0 else 0.0
        results[RISK_CHAIN_DISPLAY[slot_key]] = {
            "correct": d["correct"], "partial": d["partial"],
            "incorrect": d["incorrect"], "total": n, "accuracy": accuracy,
        }

    n = total["total"]
    overall_accuracy = (total["correct"] + 0.5 * total["partial"]) / n if n > 0 else 0.0
    results["Overall"] = {
        "correct": total["correct"], "partial": total["partial"],
        "incorrect": total["incorrect"], "total": n, "accuracy": overall_accuracy,
    }

    return results


# ============================================================
# Level 3: 计算指标
# ============================================================

def calc_level3_arbitrated() -> dict:
    merged = build_arbitrated_data(
        "Level 3 推理字段", "level3_inference/annotator_",
        INFERENCE_FIELDS, "level3_inference/arbitration.json",
    )

    by_field = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
    total = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}

    for task in merged:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "rating":
                    continue
                from_name = result.get("from_name", "")
                if from_name not in INFERENCE_FIELDS:
                    continue

                rating = result.get("value", {}).get("rating", 0)
                if rating == 3:
                    by_field[from_name]["correct"] += 1
                    total["correct"] += 1
                elif rating == 2:
                    by_field[from_name]["partial"] += 1
                    total["partial"] += 1
                elif rating == 1:
                    by_field[from_name]["incorrect"] += 1
                    total["incorrect"] += 1

                by_field[from_name]["total"] += 1
                total["total"] += 1

    results = {}
    for field_key in INFERENCE_FIELDS:
        d = by_field[field_key]
        n = d["total"]
        accuracy = (d["correct"] + 0.5 * d["partial"]) / n if n > 0 else 0.0
        evidence_support = (d["correct"] + d["partial"]) / n if n > 0 else 0.0
        results[INFERENCE_DISPLAY[field_key]] = {
            "correct": d["correct"], "partial": d["partial"],
            "incorrect": d["incorrect"], "total": n,
            "accuracy": accuracy, "evidence_support_rate": evidence_support,
        }

    n = total["total"]
    overall_accuracy = (total["correct"] + 0.5 * total["partial"]) / n if n > 0 else 0.0
    overall_evidence_support = (total["correct"] + total["partial"]) / n if n > 0 else 0.0
    results["Overall"] = {
        "correct": total["correct"], "partial": total["partial"],
        "incorrect": total["incorrect"], "total": n,
        "accuracy": overall_accuracy, "evidence_support_rate": overall_evidence_support,
    }

    return results


# ============================================================
# 输出格式化
# ============================================================

def print_level1(results: dict) -> None:
    print(f"\n{'Entity Type':<16} {'TP(ex)':<8} {'TP(par)':<8} {'FN':<6} {'FP':<6} {'Gold':<6} {'Pred':<6} {'Prec':<10} {'Rec':<10} {'F1':<10}")
    print("-" * 100)
    for etype in ENTITY_TYPE_ORDER_DISPLAY + ["Overall"]:
        r = results[etype]
        p_str = f"{r['precision']:.1%}"
        rc_str = f"{r['recall']:.1%}"
        f1_str = f"{r['f1']:.1%}"
        prefix = "** " if etype == "Overall" else ""
        suffix = " **" if etype == "Overall" else ""
        if etype == "Overall":
            print("-" * 100)
        print(f"{prefix}{etype}{suffix:<16} {r['tp_exact']:<8} {r['tp_partial']:<8} {r['fn']:<6} {r['fp']:<6} {r['gold_total']:<6} {r['pred_total']:<6} {p_str:<10} {rc_str:<10} {f1_str:<10}")


def print_level2(results: dict) -> None:
    print(f"\n{'Slot':<16} {'Correct':<10} {'Partial':<10} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10}")
    print("-" * 64)
    for slot in list(RISK_CHAIN_DISPLAY.values()) + ["Overall"]:
        r = results[slot]
        acc_str = f"{r['accuracy']:.1%}"
        if slot == "Overall":
            print("-" * 64)
        print(f"{'** ' + slot + ' **' if slot == 'Overall' else slot:<16} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10}")


def print_level3(results: dict) -> None:
    print(f"\n{'Field':<20} {'Correct':<10} {'Partial':<10} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10} {'Ev.Support':<12}")
    print("-" * 80)
    for field in list(INFERENCE_DISPLAY.values()) + ["Overall"]:
        r = results[field]
        acc_str = f"{r['accuracy']:.1%}"
        ev_str = f"{r['evidence_support_rate']:.1%}"
        if field == "Overall":
            print("-" * 80)
        print(f"{'** ' + field + ' **' if field == 'Overall' else field:<20} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10} {ev_str:<12}")


def main() -> None:
    print("AI Risk Knowledge Graph - 仲裁后评测指标计算\n")

    l1 = calc_level1_arbitrated()
    l2 = calc_level2_arbitrated()
    l3 = calc_level3_arbitrated()

    print("\n" + "=" * 80)
    print("Level 1 结果 (仲裁后)")
    print("=" * 80)
    print_level1(l1)

    print("\n" + "=" * 80)
    print("Level 2 结果 (仲裁后)")
    print("=" * 80)
    print_level2(l2)

    print("\n" + "=" * 80)
    print("Level 3 结果 (仲裁后)")
    print("=" * 80)
    print_level3(l3)

    # 汇总
    print("\n" + "=" * 80)
    print("仲裁后评测结果汇总")
    print("=" * 80)
    l1o = l1["Overall"]
    l2o = l2["Overall"]
    l3o = l3["Overall"]
    print(f"  Level 1: Precision={l1o['precision']:.1%}, Recall={l1o['recall']:.1%}, F1={l1o['f1']:.1%}")
    print(f"  Level 2: Accuracy={l2o['accuracy']:.1%}")
    print(f"  Level 3: Accuracy={l3o['accuracy']:.1%}, Ev.Support={l3o['evidence_support_rate']:.1%}")

    # 保存
    output = {
        "level1_entity_extraction": l1,
        "level2_risk_chain": l2,
        "level3_inference": l3,
    }
    output_path = EVAL_DIR.parent / "results" / "eval_results_arbitrated.json"
    save_json(output, output_path)
    print(f"\n仲裁后评测结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
