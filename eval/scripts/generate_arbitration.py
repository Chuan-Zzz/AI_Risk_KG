"""筛选标注不一致的数据，生成仲裁用 Label Studio 标注文件.

配对关系:
  - 评测1(A26) vs 评测3(A44): 同一批 50 个任务
  - 评测2(A41) vs 评测4(A45): 同一批 50 个任务

输出策略:
  - 每个层级生成一个文件，合并两对标注员的不一致任务
  - Level 2/3: prediction 中一致的槽位预填值（仲裁员无需改），分歧的槽位留空（仲裁员必须打分）
  - Level 1: prediction 中只放分歧实体，一致的实体不放
  - 两位标注员的分歧信息放在 meta_info 中
"""

from __future__ import annotations

import json
import sys
import io
import copy
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL_DIR = Path(__file__).resolve().parents[1] / "标注结果"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "label_studio" / "arbitration"


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
# Level 1: 实体级不一致 — 只放分歧实体到 prediction
# ============================================================

def build_level1_arbitration():
    print("=" * 80)
    print("Level 1: 筛选实体标注不一致的任务")
    print("=" * 80)

    all_disagree = []

    for pair_name, fa_name, fb_name in [
        ("A26_vs_A44", "显式实体评测1.json", "显式实体评测3.json"),
        ("A41_vs_A45", "显式实体评测2.json", "显式实体评测4.json"),
    ]:
        fa = load_json(EVAL_DIR / fa_name)
        fb = load_json(EVAL_DIR / fb_name)

        eid_to_a = {}
        for t in fa:
            eid = get_event_id(t)
            entities = set()
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "labels":
                        continue
                    text = r.get("value", {}).get("text", "")
                    labels = tuple(r.get("value", {}).get("labels", []))
                    entities.add((text, labels))
            eid_to_a[eid] = (t, entities)

        eid_to_b = {}
        for t in fb:
            eid = get_event_id(t)
            entities = set()
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "labels":
                        continue
                    text = r.get("value", {}).get("text", "")
                    labels = tuple(r.get("value", {}).get("labels", []))
                    entities.add((text, labels))
            eid_to_b[eid] = (t, entities)

        common_eids = set(eid_to_a.keys()) & set(eid_to_b.keys())
        agree_count = 0
        disagree_count = 0

        for eid in sorted(common_eids):
            task_a, entities_a = eid_to_a[eid]
            task_b, entities_b = eid_to_b[eid]

            if entities_a == entities_b:
                agree_count += 1
                continue

            disagree_count += 1
            only_a = entities_a - entities_b
            only_b = entities_b - entities_a

            # 收集分歧实体的 result 条目
            results_a = [
                r for r in task_a["annotations"][0].get("result", [])
                if r.get("type") == "labels"
                and (r.get("value", {}).get("text", ""),
                     tuple(r.get("value", {}).get("labels", []))) in only_a
            ]
            results_b = [
                r for r in task_b["annotations"][0].get("result", [])
                if r.get("type") == "labels"
                and (r.get("value", {}).get("text", ""),
                     tuple(r.get("value", {}).get("labels", []))) in only_b
            ]

            arb_task = {
                "data": copy.deepcopy(task_a.get("data", {})),
                "predictions": [{
                    "result": results_a + results_b,
                    "score": 0,
                    "model_version": f"disputed_entities_{pair_name}",
                }],
            }

            # 分歧摘要写入 meta_info
            a_details = ", ".join(f"{e[0]}({e[1][0]})" for e in sorted(only_a))
            b_details = ", ".join(f"{e[0]}({e[1][0]})" for e in sorted(only_b))
            existing_meta = arb_task["data"].get("meta_info", "")
            arb_task["data"]["meta_info"] = (
                f"{existing_meta}\n[ARBITRATION] {pair_name} | A独有: {a_details} | B独有: {b_details}"
                if existing_meta
                else f"[ARBITRATION] {pair_name} | A独有: {a_details} | B独有: {b_details}"
            )

            all_disagree.append(arb_task)

        print(f"  {pair_name}: 共同={len(common_eids)}, 一致={agree_count}, 不一致={disagree_count}")

    return all_disagree


# ============================================================
# Level 2 & 3: 评分级不一致
#   - 一致的槽位: 预填到 prediction（仲裁员看到已填好）
#   - 分歧的槽位: 不出现在 prediction（留空，仲裁员必须打分）
# ============================================================

def build_rating_arbitration(level_name: str, prefix: str, fields: list[str]):
    print(f"\n{'=' * 80}")
    print(f"{level_name}: 筛选评分不一致的任务")
    print(f"{'=' * 80}")

    all_disagree = []

    for pair_name, fa_name, fb_name in [
        ("A26_vs_A44", f"{prefix}1.json", f"{prefix}3.json"),
        ("A41_vs_A45", f"{prefix}2.json", f"{prefix}4.json"),
    ]:
        fa = load_json(EVAL_DIR / fa_name)
        fb = load_json(EVAL_DIR / fb_name)

        eid_to_a = {}
        for t in fa:
            eid = get_event_id(t)
            ratings = {}
            result_map = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    fname = r.get("from_name", "")
                    ratings[fname] = r.get("value", {}).get("rating", 0)
                    result_map[fname] = r
            eid_to_a[eid] = (t, ratings, result_map)

        eid_to_b = {}
        for t in fb:
            eid = get_event_id(t)
            ratings = {}
            result_map = {}
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "rating":
                        continue
                    fname = r.get("from_name", "")
                    ratings[fname] = r.get("value", {}).get("rating", 0)
                    result_map[fname] = r
            eid_to_b[eid] = (t, ratings, result_map)

        common_eids = set(eid_to_a.keys()) & set(eid_to_b.keys())
        agree_count = 0
        disagree_count = 0
        diff_field_counts = defaultdict(int)

        for eid in sorted(common_eids):
            task_a, ratings_a, result_map_a = eid_to_a[eid]
            task_b, ratings_b, result_map_b = eid_to_b[eid]

            diff_fields = []
            agree_fields = []
            for field in fields:
                ra = ratings_a.get(field)
                rb = ratings_b.get(field)
                if ra is not None and rb is not None:
                    if ra != rb:
                        diff_fields.append(field)
                        diff_field_counts[field] += 1
                    else:
                        agree_fields.append((field, ra))

            if not diff_fields:
                agree_count += 1
                continue

            disagree_count += 1

            # prediction: 只放一致的槽位（预填值），分歧的槽位留空
            agreed_results = []
            for field, value in agree_fields:
                r = copy.deepcopy(result_map_a.get(field, result_map_b.get(field)))
                if r:
                    r["value"]["rating"] = value
                    agreed_results.append(r)

            arb_task = {
                "data": copy.deepcopy(task_a.get("data", {})),
                "predictions": [{
                    "result": agreed_results,
                    "score": 0,
                    "model_version": f"agreed_slots_{pair_name}",
                }],
            }

            # 分歧摘要写入 meta_info
            diff_parts = [f"{f}: A={ratings_a.get(f)} B={ratings_b.get(f)}" for f in diff_fields]
            diff_info = " | ".join(diff_parts)
            existing_meta = arb_task["data"].get("meta_info", "")
            arb_task["data"]["meta_info"] = (
                f"{existing_meta}\n[ARBITRATION] {pair_name} | 分歧: {diff_info}"
                if existing_meta
                else f"[ARBITRATION] {pair_name} | 分歧: {diff_info}"
            )

            all_disagree.append(arb_task)

        print(f"  {pair_name}: 共同={len(common_eids)}, 一致={agree_count}, 不一致={disagree_count}")
        if diff_field_counts:
            print(f"    槽位分歧次数: {dict(sorted(diff_field_counts.items(), key=lambda x: -x[1]))}")

    return all_disagree


# ============================================================
# 生成 Label Studio 仲裁文件
# ============================================================

def generate_ls_file(tasks: list[dict], level_dir: str):
    """生成可直接导入 Label Studio 的仲裁任务文件."""
    if not tasks:
        print(f"  {level_dir}: 无不一致任务，跳过")
        return

    out_path = OUTPUT_DIR / level_dir / "arbitration.json"
    save_json(tasks, out_path)
    print(f"  {level_dir}/arbitration.json: {len(tasks)} 个任务")


if __name__ == "__main__":
    r1 = build_level1_arbitration()

    r2 = build_rating_arbitration(
        "Level 2 风险链", "风险链评测",
        ["risk_source_score", "risk_score", "consequence_score",
         "impact_score", "affected_actor_score", "risk_control_score"],
    )

    r3 = build_rating_arbitration(
        "Level 3 推理字段", "推理字段评测",
        ["purpose_score", "lifecycle_score", "domain_score"],
    )

    print(f"\n{'=' * 80}")
    print("生成 Label Studio 仲裁文件")
    print(f"{'=' * 80}")

    generate_ls_file(r1, "level1_entity")
    generate_ls_file(r2, "level2_risk_chain")
    generate_ls_file(r3, "level3_inference")

    # 汇总
    summary = {
        "level1_entity": len(r1),
        "level2_risk_chain": len(r2),
        "level3_inference": len(r3),
    }
    save_json(summary, OUTPUT_DIR / "arbitration_summary.json")
    print(f"\n  汇总: {summary}")

    print("\n" + "=" * 80)
    print("完成！仲裁文件已保存到 eval/label_studio/arbitration/")
    print("  - 一致的槽位已预填值（仲裁员无需修改）")
    print("  - 分歧的槽位留空（仲裁员需要打分）")
    print("  - 分歧详情在 meta_info 中")
    print("=" * 80)
