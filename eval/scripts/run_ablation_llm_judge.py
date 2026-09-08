"""LLM-as-judge:对消融实验输出做语义级 ESR / CDC 判定。

判定范围(只判定规则化方法无法确定的样本,大幅减少 API 调用):

ESR (Evidence Support Rate):
    仅对"字面未匹配"的边调用 LLM,判断 evidence_sentence 是否在语义上
    支持该三元组声明。字面已匹配的边直接计为支持,无需 LLM。

CDC (Cross-Document Consistency):
    仅对"多 object 组"(同一 subject + predicate 对应多个不同 object)
    调用 LLM,判断这些 object 声明是否语义一致。单 object 组直接计为一致。

特性:
    - 批量请求(每批 EVIDENCE_BATCH_SIZE / CDC_GROUP_BATCH_SIZE 条)
    - 并发执行(ThreadPoolExecutor)
    - checkpoint(每完成一批即写入 JSON,中断后可恢复)
    - 直连 API(绕过代理,trust_env=False)

Usage:
    python eval/scripts/run_ablation_llm_judge.py --workers 8
    python eval/scripts/run_ablation_llm_judge.py --workers 8 --max-events 20  # 抽样测试
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

# 绕过系统代理(ClashX 等会干扰长连接)
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                    "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_var, None)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_config
from src.core.datasets import build_documents_for_event, load_cases, load_events

# 复用主脚本的工具函数
from eval.scripts.compute_ablation_metrics import (
    VARIANTS,
    GOLD_FILE,
    OUTPUT_DIR,
    _evidence_fully_supported,
    _load_gold_ids,
    _load_json,
    _load_subgraph,
    _node_map,
    _normalise_text,
    _save_json,
    _semantic_edges,
    _statement_groups,
)

JUDGMENT_FILE = PROJECT_ROOT / "eval" / "results" / "ablation_llm_judgments.json"
EVIDENCE_BATCH_SIZE = 20
CDC_GROUP_BATCH_SIZE = 20


class DirectLLMJudge:
    """直连 OpenAI-compatible LLM 端点,绕过系统代理。"""

    def __init__(
        self,
        model: str | None = None,
        request_timeout: float = 120.0,
        retries: int = 2,
        max_tokens: int = 2048,
    ) -> None:
        config = get_config()
        self.model = model or os.environ.get("LLM_EVAL_MODEL") or "glm-5.2"
        self.base_url = config.get("llm.primary.base_url", "").rstrip("/")
        self.api_key = config.get("llm.primary.api_key", "")
        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM base URL and API key must be configured")
        self.client = httpx.Client(
            timeout=httpx.Timeout(request_timeout, connect=min(20.0, request_timeout)),
            trust_env=False,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.retries = retries
        self.max_tokens = max_tokens

    def judge(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions", json=request
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content", "").strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
        raise RuntimeError(f"LLM judge request failed after {self.retries+1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

ESR_SYSTEM = (
    "你是一个严格的证据审核员。给定一个知识三元组声明(subject-predicate-object)"
    "和它的证据句子(evidence_sentence)及来源文档片段(source_text),"
    "判断证据句子是否在语义上支持该声明。\n\n"
    "判定规则:\n"
    "- 如果证据句子明确陈述了与该三元组相同或等价的语义 → supported=true\n"
    "- 如果证据句子是该声明的改写、转述、总结 → supported=true\n"
    "- 如果证据句子与声明无关,或语义矛盾 → supported=false\n"
    "- 如果证据句子只部分提及,不足以完整支持声明 → supported=false\n\n"
    "返回 JSON,格式: {\"items\": [{\"key\": \"...\", \"supported\": true/false}]}"
)

CDC_SYSTEM = (
    "你是一个跨文档一致性审核员。给定同一主体(subject)在同一关系(predicate)下"
    "的多个声明(来自不同文档),判断这些声明是否相互一致。\n\n"
    "判定规则:\n"
    "- 如果多个 object 是互补的(如不同方面、不同阶段)→ consistent=true\n"
    "- 如果多个 object 是同一实体的不同名称/别名 → consistent=true\n"
    "- 如果多个 object 虽然不同但不矛盾(如涉及不同子系统)→ consistent=true\n"
    "- 如果多个 object 相互排斥或矛盾(如同一属性有冲突值)→ consistent=false\n"
    "- 如果无法判断,倾向于 consistent=true(保守判定)\n\n"
    "返回 JSON,格式: {\"groups\": [{\"key\": \"...\", \"consistent\": true/false}]}"
)


def _build_evidence_payload(
    items: list[dict[str, Any]],
    event_documents: dict[str, str],
) -> dict[str, Any]:
    """构建 ESR 判定的 payload。"""
    payload_items = []
    for item in items:
        doc_text = event_documents.get(item["source_doc_id"], "")
        # 截断过长的文档,只取包含证据句子的上下文窗口
        if doc_text and len(doc_text) > 2000:
            # 尝试找到证据句子在文档中的位置,取前后 1000 字符
            evidence = item.get("evidence_sentence", "")
            idx = doc_text.find(evidence[:50]) if evidence else -1
            if idx >= 0:
                start = max(0, idx - 500)
                end = min(len(doc_text), idx + len(evidence) + 500)
                doc_text = doc_text[start:end]
            else:
                doc_text = doc_text[:2000]
        payload_items.append({
            "key": item["key"],
            "subject": item["subject"],
            "predicate": item["predicate"],
            "object": item["object"],
            "evidence_sentence": item.get("evidence_sentence", ""),
            "source_text": doc_text[:2000],
        })
    return {"items": payload_items}


def _build_cdc_payload(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 CDC 判定的 payload。"""
    payload_groups = []
    for group in groups:
        facts = []
        for edge in group["objects"]:
            source_doc_ids = sorted({
                str(item.get("source_doc_id", ""))
                for item in edge.get("evidence", [])
                if item.get("source_doc_id")
            })
            facts.append({
                "object": edge.get("object_name", ""),
                "source_doc_ids": source_doc_ids,
            })
        payload_groups.append({
            "key": group["key"],
            "subject": group["subject"],
            "predicate": group["predicate"],
            "facts": facts,
        })
    return {"groups": payload_groups}


# ---------------------------------------------------------------------------
# 任务收集
# ---------------------------------------------------------------------------

def _collect_pending_tasks(
    variant: str,
    event_ids: list[str],
    event_documents: dict[str, dict[str, str]],
    judgments: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """收集待判定的 ESR 和 CDC 任务。"""
    evidence_tasks: list[dict[str, Any]] = []
    cdc_tasks: list[dict[str, Any]] = []

    for event_id in event_ids:
        subgraph = _load_subgraph(variant, event_id)
        if subgraph is None:
            continue
        docs = event_documents.get(event_id, {})

        # ESR:只收集字面未匹配的边
        for edge_index, edge in enumerate(_semantic_edges(subgraph)):
            supported, reason = _evidence_fully_supported(edge, docs)
            if supported:
                continue
            if reason != "unmatched_evidence":
                continue  # missing_evidence 无法判定,跳过
            key = f"{variant}|{event_id}|evidence|{edge_index}"
            if key in judgments:
                continue  # 已判定
            # 取第一条 evidence(通常只有一条)
            ev = edge.get("evidence", [{}])[0] if edge.get("evidence") else {}
            evidence_tasks.append({
                "key": key,
                "subject": edge.get("subject_name", ""),
                "predicate": edge.get("predicate", ""),
                "object": edge.get("object_name", ""),
                "evidence_sentence": ev.get("evidence_sentence", ""),
                "source_doc_id": str(ev.get("source_doc_id", "")),
            })

        # CDC:只收集多 object 组
        nodes = _node_map(subgraph)
        for group in _statement_groups(subgraph):
            if not group["requires_judgment"]:
                continue
            key = f"{variant}|{event_id}|group|{group['key']}"
            if key in judgments:
                continue  # 已判定
            cdc_tasks.append({
                "key": key,
                "subject": group["subject"],
                "predicate": group["predicate"],
                "objects": group["objects"],
            })

    return evidence_tasks, cdc_tasks


def _chunks(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


# ---------------------------------------------------------------------------
# 批量判定
# ---------------------------------------------------------------------------

def _judge_evidence_batch(
    llm: DirectLLMJudge,
    items: list[dict[str, Any]],
    event_documents: dict[str, str],
) -> dict[str, bool]:
    """判定一批 evidence 是否语义支持声明。"""
    payload = _build_evidence_payload(items, event_documents)
    data = llm.judge(ESR_SYSTEM, payload)
    requested_keys = {str(item["key"]) for item in items if item.get("key")}
    decisions: dict[str, bool] = {}
    for item in data.get("items", []):
        if isinstance(item, dict) and str(item.get("key", "")) in requested_keys:
            decisions[str(item.get("key"))] = bool(item.get("supported", False))
    if items and not decisions:
        raise RuntimeError("LLM ESR response did not contain any requested keys")
    return decisions


def _judge_cdc_batch(
    llm: DirectLLMJudge,
    groups: list[dict[str, Any]],
) -> dict[str, bool]:
    """判定一批 CDC 组是否语义一致。"""
    payload = _build_cdc_payload(groups)
    data = llm.judge(CDC_SYSTEM, payload)
    requested_keys = {str(g["key"]) for g in groups if g.get("key")}
    decisions: dict[str, bool] = {}
    for item in data.get("groups", []):
        if isinstance(item, dict) and str(item.get("key", "")) in requested_keys:
            decisions[str(item.get("key"))] = bool(item.get("consistent", False))
    if groups and not decisions:
        raise RuntimeError("LLM CDC response did not contain any requested keys")
    return decisions


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge for ablation ESR/CDC evaluation"
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--max-events", type=int, default=None, help="Limit events (smoke test)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout (s)")
    args = parser.parse_args()

    cases = load_cases()
    events = {event.get("incident_id"): event for event in load_events()}
    event_ids = _load_gold_ids()
    if args.max_events is not None:
        event_ids = event_ids[:args.max_events]

    # 预加载源文档
    event_documents: dict[str, dict[str, str]] = {}
    for event_id in event_ids:
        event = events.get(event_id)
        if event:
            event_documents[event_id] = {
                report.doc_id: report.content
                for report in build_documents_for_event(event, cases)
            }

    # 加载已有 judgment(支持恢复)
    judgments = _load_json(JUDGMENT_FILE, {})
    initial_count = len(judgments)
    print(f"Loaded {initial_count} existing judgments from {JUDGMENT_FILE}")

    # 收集所有待判定任务
    all_evidence_tasks: list[tuple[str, list[dict[str, Any]]]] = []
    all_cdc_tasks: list[tuple[str, list[dict[str, Any]]]] = []

    for variant in VARIANTS:
        ev_tasks, cdc_tasks = _collect_pending_tasks(
            variant, event_ids, event_documents, judgments
        )
        if ev_tasks:
            all_evidence_tasks.append((variant, ev_tasks))
        if cdc_tasks:
            all_cdc_tasks.append((variant, cdc_tasks))
        print(f"  {variant}: {len(ev_tasks)} ESR pending, {len(cdc_tasks)} CDC pending")

    total_evidence = sum(len(t) for _, t in all_evidence_tasks)
    total_cdc = sum(len(t) for _, t in all_cdc_tasks)
    total = total_evidence + total_cdc
    print(f"\nTotal pending: {total_evidence} ESR + {total_cdc} CDC = {total} items")

    if total == 0:
        print("All items already judged. Nothing to do.")
        return

    # 构建批量任务列表
    batches: list[dict[str, Any]] = []
    for variant, tasks in all_evidence_tasks:
        for chunk in _chunks(tasks, EVIDENCE_BATCH_SIZE):
            batches.append({
                "type": "evidence",
                "variant": variant,
                "items": chunk,
            })
    for variant, tasks in all_cdc_tasks:
        for chunk in _chunks(tasks, CDC_GROUP_BATCH_SIZE):
            batches.append({
                "type": "cdc",
                "variant": variant,
                "items": chunk,
            })

    print(f"Total batches: {len(batches)} "
          f"(evidence batch={EVIDENCE_BATCH_SIZE}, cdc batch={CDC_GROUP_BATCH_SIZE})")
    print(f"Workers: {args.workers}\n")

    llm = DirectLLMJudge(
        model=args.model,
        request_timeout=args.timeout,
        retries=2,
    )
    print(f"Model: {llm.model}")
    print(f"Endpoint: {llm.base_url}")
    print()

    completed = 0
    failed = 0
    start_time = time.monotonic()
    last_save = 0

    def _process_batch(batch: dict[str, Any]) -> dict[str, bool]:
        """处理一个批次,返回 decision 字典。"""
        variant = batch["variant"]
        if batch["type"] == "evidence":
            docs = event_documents.get(
                # 从 key 中提取 event_id: "variant|event_id|evidence|index"
                batch["items"][0]["key"].split("|")[1] if batch["items"] else "",
                {},
            )
            return _judge_evidence_batch(llm, batch["items"], docs)
        else:
            return _judge_cdc_batch(llm, batch["items"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            batch_type = batch["type"]
            batch_variant = batch["variant"]
            batch_size = len(batch["items"])

            try:
                decisions = future.result()
                judgments.update(decisions)
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {batch_type} batch ({batch_variant}, "
                      f"{batch_size} items): {exc}")

            # 每 10 批或每分钟保存一次 checkpoint
            now = time.monotonic()
            if completed % 10 == 0 or now - last_save > 60:
                _save_json(JUDGMENT_FILE, judgments)
                last_save = now
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(batches) - completed - failed) / rate if rate > 0 else 0
                print(f"  [{completed + failed}/{len(batches)}] "
                      f"done={completed} fail={failed} "
                      f"judgments={len(judgments)} "
                      f"rate={rate:.1f}/s "
                      f"ETA={eta / 60:.1f}min "
                      f"elapsed={elapsed / 60:.1f}min",
                      flush=True)

    # 最终保存
    _save_json(JUDGMENT_FILE, judgments)
    elapsed = time.monotonic() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{len(batches)} batches ({failed} failed)")
    print(f"Judgments: {initial_count} → {len(judgments)} (+{len(judgments) - initial_count})")
    print(f"Time: {elapsed / 60:.1f} min")
    print(f"Saved to {JUDGMENT_FILE}")


if __name__ == "__main__":
    main()
