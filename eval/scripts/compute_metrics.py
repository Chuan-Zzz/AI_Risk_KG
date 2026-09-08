"""AI Risk Knowledge Graph - 评测指标计算脚本.

基于 Label Studio 标注结果，按评测方案计算三个层级的指标：
  Level 1: 显式实体评测 (Precision / Recall / F1)
  Level 2: 风险链评测 (Accuracy)
  Level 3: 推理字段评测 (Accuracy / Evidence Support Rate)

数据格式说明：
  Level 1 (显式实体): Label Studio NER 标注
    - origin=prediction: 模型预测，标注员确认 → TP (Exact Match)
    - origin=prediction-changed: 模型预测，标注员修改标签 → Partial Match
    - origin=manual: 标注员手动添加 → FN (模型遗漏)
    - 被删除的预测: 不出现在结果中 → FP (模型预测错误)

  Level 2 (风险链): 评分标注 (3=Correct, 2=Partial, 1=Incorrect)
  Level 3 (推理字段): 评分标注 (3=Correct, 2=Partial, 1=Incorrect)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "annotations"

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


def _normalize_type(label: str) -> str:
    if label in STAKEHOLDER_SUBTYPES:
        return "Stakeholder"
    return label


def load_json_files(pattern: str) -> list[dict]:
    files = sorted(EVAL_DIR.glob(pattern))
    all_data = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            all_data.extend(data)
    return all_data


# ============================================================
# Level 1: 显式实体评测 (含 FP 恢复)
# ============================================================

def _recover_fp(task: dict) -> list[dict]:
    """从 annotation.prediction.result 恢复被标注员删除的预测 (FP)."""
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


def calc_level1() -> dict:
    data = load_json_files("level1_entity_*.json")

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

            for fp_result in _recover_fp(task):
                if fp_result.get("type") != "labels":
                    continue
                labels = fp_result.get("value", {}).get("labels", [])
                primary_label = labels[0] if labels else "Unknown"
                by_type[primary_label]["fp"] += 1
                total["fp"] += 1

    results = {}
    for etype in ENTITY_TYPE_ORDER:
        d = by_type[etype]
        tp_exact = d["tp_exact"]
        tp_partial = d["tp_partial"]
        fn = d["fn"]
        fp = d["fp"]

        tp_weighted = tp_exact + 0.5 * tp_partial
        gold_total = tp_exact + tp_partial + fn
        pred_total = tp_exact + tp_partial + fp

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
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    results["Overall"] = {
        "tp_exact": tp_exact,
        "tp_partial": tp_partial,
        "fn": fn,
        "fp": fp,
        "gold_total": gold_total,
        "pred_total": pred_total,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
    }

    return results


# ============================================================
# Level 2: 风险链评测
# ============================================================

def calc_level2() -> dict:
    data = load_json_files("level2_risk_chain_*.json")

    by_slot = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
    total = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}

    for task in data:
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
            "correct": d["correct"],
            "partial": d["partial"],
            "incorrect": d["incorrect"],
            "total": n,
            "accuracy": accuracy,
        }

    n = total["total"]
    overall_accuracy = (total["correct"] + 0.5 * total["partial"]) / n if n > 0 else 0.0
    results["Overall"] = {
        "correct": total["correct"],
        "partial": total["partial"],
        "incorrect": total["incorrect"],
        "total": n,
        "accuracy": overall_accuracy,
    }

    return results


# ============================================================
# Level 3: 推理字段评测
# ============================================================

def calc_level3() -> dict:
    data = load_json_files("level3_inference_*.json")

    by_field = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
    total = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}

    for task in data:
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
            "correct": d["correct"],
            "partial": d["partial"],
            "incorrect": d["incorrect"],
            "total": n,
            "accuracy": accuracy,
            "evidence_support_rate": evidence_support,
        }

    n = total["total"]
    overall_accuracy = (total["correct"] + 0.5 * total["partial"]) / n if n > 0 else 0.0
    overall_evidence_support = (total["correct"] + total["partial"]) / n if n > 0 else 0.0
    results["Overall"] = {
        "correct": total["correct"],
        "partial": total["partial"],
        "incorrect": total["incorrect"],
        "total": n,
        "accuracy": overall_accuracy,
        "evidence_support_rate": overall_evidence_support,
    }

    return results


# ============================================================
# 输出格式化
# ============================================================

def print_level1(results: dict) -> None:
    print("\n" + "=" * 80)
    print("Table 1: Entity Extraction Results (Level 1 - 显式实体评测)")
    print("=" * 80)
    print()
    print(f"{'Entity Type':<16} {'TP(Exact)':<10} {'TP(Partial)':<12} {'FN':<6} {'Gold':<6} {'Pred':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 96)

    display_order = ENTITY_TYPE_ORDER_DISPLAY + ["Overall"]
    for etype in display_order:
        r = results[etype]
        p_str = f"{r['precision']:.1%}"
        rc_str = f"{r['recall']:.1%}"
        f1_str = f"{r['f1']:.1%}"

        if etype == "Overall":
            print("-" * 96)
            print(f"{'** ' + etype + ' **':<16} {r['tp_exact']:<10} {r['tp_partial']:<12} {r['fn']:<6} {r['gold_total']:<6} {r['pred_total']:<6} {p_str:<10} {rc_str:<10} {f1_str:<10}")
        else:
            print(f"{etype:<16} {r['tp_exact']:<10} {r['tp_partial']:<12} {r['fn']:<6} {r['gold_total']:<6} {r['pred_total']:<6} {p_str:<10} {rc_str:<10} {f1_str:<10}")

    print()
    print("说明:")
    print("  TP(Exact)   = origin=prediction (模型预测正确，标注员确认)")
    print("  TP(Partial) = origin=prediction-changed (模型找到实体但标签被修改)")
    print("  FN          = origin=manual (标注员手动添加，模型遗漏)")
    print("  Precision   = (TP_exact + 0.5×TP_partial) / (TP_exact + TP_partial)")
    print("  Recall      = (TP_exact + 0.5×TP_partial) / (TP_exact + TP_partial + FN)")
    print("  F1          = 2×P×R / (P+R)")
    print("  注意: Precision 为上界估计（未计入被标注员删除的错误预测）")


def print_level2(results: dict) -> None:
    print("\n" + "=" * 80)
    print("Table 2: Risk Chain Accuracy (Level 2 - 风险链评测)")
    print("=" * 80)
    print()
    print(f"{'Slot':<16} {'Correct':<10} {'Partial':<10} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10}")
    print("-" * 64)

    for slot in list(RISK_CHAIN_DISPLAY.values()) + ["Overall"]:
        r = results[slot]
        acc_str = f"{r['accuracy']:.1%}"

        if slot == "Overall":
            print("-" * 64)
            print(f"{'** ' + slot + ' **':<16} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10}")
        else:
            print(f"{slot:<16} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10}")

    print()
    print("说明:")
    print("  Accuracy = (N_correct + 0.5×N_partial) / N_total")
    print("  评分标准: 3=Correct, 2=Partially correct, 1=Incorrect")


def print_level3(results: dict) -> None:
    print("\n" + "=" * 80)
    print("Table 2b: Inference Field Accuracy (Level 3 - 推理字段评测)")
    print("=" * 80)
    print()
    print(f"{'Field':<20} {'Correct':<10} {'Partial':<10} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10} {'Ev.Support':<12}")
    print("-" * 80)

    for field in list(INFERENCE_DISPLAY.values()) + ["Overall"]:
        r = results[field]
        acc_str = f"{r['accuracy']:.1%}"
        ev_str = f"{r['evidence_support_rate']:.1%}"

        if field == "Overall":
            print("-" * 80)
            print(f"{'** ' + field + ' **':<20} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10} {ev_str:<12}")
        else:
            print(f"{field:<20} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<10} {r['total']:<8} {acc_str:<10} {ev_str:<12}")

    print()
    print("说明:")
    print("  Accuracy            = (N_correct + 0.5×N_partial) / N_total")
    print("  Evidence Support Rate = (N_correct + N_partial) / N_total")
    print("  评分标准: 3=Correct, 2=Partially correct, 1=Incorrect")


def print_summary(l1: dict, l2: dict, l3: dict) -> None:
    print("\n" + "=" * 80)
    print("评测结果汇总")
    print("=" * 80)
    print()
    print(f"{'评测维度':<24} {'核心指标':<12} {'值':<10}")
    print("-" * 46)

    l1_overall = l1["Overall"]
    l2_overall = l2["Overall"]
    l3_overall = l3["Overall"]

    print(f"{'Level 1: 显式实体':<24} {'Precision':<12} {l1_overall['precision']:.1%}")
    print(f"{'Level 1: 显式实体':<24} {'Recall':<12} {l1_overall['recall']:.1%}")
    print(f"{'Level 1: 显式实体':<24} {'F1':<12} {l1_overall['f1']:.1%}")
    print(f"{'Level 2: 风险链':<24} {'Accuracy':<12} {l2_overall['accuracy']:.1%}")
    print(f"{'Level 3: 推理字段':<24} {'Accuracy':<12} {l3_overall['accuracy']:.1%}")
    print(f"{'Level 3: 推理字段':<24} {'Ev.Support':<12} {l3_overall['evidence_support_rate']:.1%}")

    print()
    print("数据统计:")
    print(f"  Level 1 标注实体总数: {l1_overall['gold_total']} (模型预测 {l1_overall['pred_total']}, 遗漏 {l1_overall['fn']})")
    print(f"  Level 2 评分总数: {l2_overall['total']}")
    print(f"  Level 3 评分总数: {l3_overall['total']}")


def main() -> None:
    print("AI Risk Knowledge Graph - 评测指标计算")
    print(f"数据目录: {EVAL_DIR}")

    l1 = calc_level1()
    l2 = calc_level2()
    l3 = calc_level3()

    print_level1(l1)
    print_level2(l2)
    print_level3(l3)
    print_summary(l1, l2, l3)

    output = {
        "level1_entity_extraction": l1,
        "level2_risk_chain": l2,
        "level3_inference": l3,
    }

    output_path = EVAL_DIR.parent / "results" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评测结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
