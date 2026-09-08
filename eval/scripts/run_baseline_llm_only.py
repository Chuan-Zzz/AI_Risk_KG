"""Baseline 1: LLM-only direct KG extraction (论文补实验方案 §5.2).

范式: Documents -> LLM -> triples
  - 无 ontology schema(类型与谓词自由生成, 仅做 alias 级最弱归一化)
  - 无 event aggregation(事件内多文档输出按简单 union 合并)
  - 无 evidence constraint(证据句子为可选字段, 不做强制校验)

任务定义(非 schema)以自然语言给出: 实体、关系、五槽风险传播链。
风险链槽位名属于任务定义, 用于构建链边; 其余类型/谓词保持模型原始输出。

输出: output/experiments/baseline_llm_only/<event_id>/event_subgraph.json
(直接写 JSON dict, 不经过 pydantic 校验 —— 自由类型无法通过枚举校验,
 eval/scripts/compute_ablation_metrics.py 以 raw JSON 读取, 口径一致)

特性:
  - httpx 直连(trust_env=False, 绕过代理与 OpenAI SDK 特征头)
  - 主模型 deepseek-v4-flash, 内容审核拒绝/失败时回退 minimax-m3
  - 并发执行 + 事件级断点续跑(已有输出的事件自动跳过)

Usage:
    python eval/scripts/run_baseline_llm_only.py --workers 4
    python eval/scripts/run_baseline_llm_only.py --workers 4 --max-events 3  # 冒烟测试
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
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

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.core.datasets import build_documents_for_event, load_cases, load_events

OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments" / "baseline_llm_only"
GOLD_FILE = PROJECT_ROOT / "data" / "gold_standard_100_events.json"

PRIMARY_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
FALLBACK_MODEL = os.environ.get("BASELINE_FALLBACK_LLM_MODEL", "minimax-m3")
BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")

MAX_INPUT_CHARS = 3500   # 单次请求输入截断(防 Cloudflare 120s 超时)
MAX_OUTPUT_TOKENS = 6000
REQUEST_TIMEOUT = 170.0  # 略低于 Cloudflare ~120s+重试预算, 交由重试逻辑兜底
RETRIES = 2

# 风险链五槽(任务定义): 槽名 -> (链上类型, 入边谓词, 出边谓词)
# 链结构: source --causes--> risk --leadsTo--> consequence
#         --impacts--> impact --affects--> affected_actor
CHAIN_SLOTS: list[tuple[str, str, str]] = [
    ("risk_source", "RiskSource"),
    ("risk", "Risk"),
    ("consequence", "Consequence"),
    ("impact", "Impact"),
    ("affected_actor", "AffectedActor"),
]
CHAIN_EDGES: list[tuple[int, int, str]] = [
    (0, 1, "causes"),
    (1, 2, "leadsTo"),
    (2, 3, "impacts"),
    (3, 4, "affects"),
]

# 最弱谓词归一化(alias 级, 固定映射表, 不引入本体知识)
PREDICATE_ALIASES: dict[str, str] = {
    "caused": "causes", "caused by": "causes", "cause": "causes",
    "leads to": "leadsTo", "leads_to": "leadsTo", "led to": "leadsTo",
    "results in": "leadsTo", "results_in": "leadsTo", "resulted in": "leadsTo",
    "impacts": "impacts", "impact": "impacts", "impacted": "impacts",
    "affects": "affects", "affect": "affects", "affected": "affects",
}


def _normalise_name(value: str) -> str:
    """NFKC + 小写 + 去标点空格, 用于 union 去重键。"""
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def _norm_predicate(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    # 已是 canonical 链谓词, 原样保留(避免 camelCase 被 lower)
    canonical = {"causes", "leadsTo", "impacts", "affects"}
    if value in canonical:
        return value
    lower = value.lower()
    if lower in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[lower]
    # 空格/下划线转 camelCase(仅格式归一, 不改语义)
    parts = re.split(r"[\s_]+", lower)
    return parts[0] + "".join(p.capitalize() for p in parts[1:] if p)


def _is_refusal(text: str) -> bool:
    lower = (text or "").lower().strip()
    if len(lower) < 200 and "{" not in lower:
        return True
    patterns = (
        "i cannot assist", "i'm sorry", "i am sorry", "i can't assist",
        "against my guidelines", "i'm unable to", "i am unable to",
        "cannot fulfill", "content policy",
    )
    return any(p in lower for p in patterns)


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    # strict=False: 容忍字符串内的原始控制字符(换行/制表符)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # 截断修复: 补齐未闭合的字符串与括号
        repaired = cleaned
        # 去掉最后一个不完整键值对(截断点), 再补括号
        repaired = re.sub(r',?\s*"[^"]*"\s*:?\s*$', "", repaired)
        repaired = re.sub(r',\s*"[^"]*"\s*$', "", repaired)
        for closer, opener in (("}", "{"), ("]", "[")):
            deficit = repaired.count(opener) - repaired.count(closer)
            repaired += closer * max(0, deficit)
        return json.loads(repaired, strict=False)


class DirectClient:
    """httpx 直连 OpenAI 兼容端点(绕过代理与 SDK 特征头)。"""

    def __init__(self) -> None:
        if not BASE_URL or not API_KEY:
            raise RuntimeError("LLM_BASE_URL / LLM_API_KEY must be set (env or .env)")
        self.client = httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=20.0),
            trust_env=False,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    def chat(self, system: str, user: str, model: str) -> str:
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # 注意: 该端点对 response_format 处理有缺陷(原样回显),
            # JSON 约束完全依赖 system prompt 指令
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        last_error: Exception | None = None
        for attempt in range(RETRIES + 1):
            try:
                response = self.client.post(
                    f"{BASE_URL}/chat/completions", json=request
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content", "") or ""
                if content.strip():
                    return content
                last_error = RuntimeError("empty content")
            except Exception as exc:
                last_error = exc
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"request failed after {RETRIES+1} attempts: {last_error}")

    def chat_with_fallback(self, system: str, user: str) -> str:
        """主模型 -> (拒绝/异常时) minimax-m3。"""
        try:
            content = self.chat(system, user, PRIMARY_MODEL)
            if not _is_refusal(content):
                return content
            print(f"    [refusal] {PRIMARY_MODEL}, falling back to {FALLBACK_MODEL}")
        except Exception as exc:
            print(f"    [error] {PRIMARY_MODEL}: {str(exc)[:120]}, trying {FALLBACK_MODEL}")
        return self.chat(system, user, FALLBACK_MODEL)


SYSTEM_PROMPT = (
    "You are an information extraction system. Given one news report about an "
    "AI-related risk incident, extract structured knowledge as JSON.\n\n"
    "Output JSON with exactly these keys:\n"
    '{"entities": [{"name": str, "type": str, "description": str, '
    '"evidence_sentence": str}],\n'
    ' "relations": [{"subject": str, "predicate": str, "object": str, '
    '"evidence_sentence": str}],\n'
    ' "risk_chain": {"risk_source": str, "risk": str, "consequence": str, '
    '"impact": str, "affected_actor": str, "evidence_sentence": str}}\n\n'
    "Rules:\n"
    "- entity type labels and relation predicates are free-form: use whatever "
    "labels the text supports (e.g. Company, Person, Technology, caused, exposed).\n"
    "- every subject and object in relations MUST also appear as an entity name.\n"
    "- risk_chain holds the names of the five chain slots of this incident: "
    "what initiated the risk (risk_source), the risk itself, the direct "
    "consequence, the broader impact, and who was affected (affected_actor). "
    "Use entity names; use empty string when a slot is not stated.\n"
    "- evidence_sentence: one sentence copied verbatim from the report that "
    "supports the item. If no single sentence supports it, use empty string.\n"
    "- respond with JSON only."
)


def _extract_doc(client: DirectClient, doc: dict[str, str]) -> dict[str, Any]:
    """单文档直接抽取, 返回解析后的 JSON dict(失败返回空 dict)。

    解析失败(截断/格式错误)也触发 minimax-m3 兜底重试。
    """
    content = doc["content"][:MAX_INPUT_CHARS]
    user_msg = f"Title: {doc['title']}\n\nReport:\n{content}"
    try:
        raw = client.chat_with_fallback(SYSTEM_PROMPT, user_msg)
        return _parse_json(raw)
    except Exception:
        # 主模型输出非 JSON(截断/控制字符): 用兜底模型重试一次
        try:
            print(f"    [parse-fail] doc {doc['doc_id']}, retrying with {FALLBACK_MODEL}")
            raw = client.chat(SYSTEM_PROMPT, user_msg, FALLBACK_MODEL)
            return _parse_json(raw)
        except Exception as exc:
            print(f"    [fail] doc {doc['doc_id']}: {str(exc)[:120]}")
            return {}


def _evidence_items(sentence: str, doc_id: str) -> list[dict[str, Any]]:
    sentence = (sentence or "").strip()
    if not sentence:
        return []
    return [{
        "evidence_id": f"{doc_id}-{abs(hash(sentence)) % 10**10}",
        "evidence_sentence": sentence,
        "source_doc_id": doc_id,
        "mention_span": None,
        "confidence": 0.8,
    }]


def _merge_union(event_id: str, docs: list[dict[str, str]],
                 per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    """事件内简单 union: 节点按(归一化名, 原始类型), 边按(归一化s, 谓词, 归一化o)。"""
    node_map: dict[tuple[str, str], dict[str, Any]] = {}
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(name: str, entity_type: str, evidence: list[dict[str, Any]],
                 source_doc_id: str) -> str:
        name = (name or "").strip()
        key = (_normalise_name(name), entity_type)
        if not key[0]:
            return ""
        if key in node_map:
            node = node_map[key]
            seen = {(e["source_doc_id"], e["evidence_sentence"]) for e in node["evidence"]}
            node["evidence"].extend(
                e for e in evidence
                if (e["source_doc_id"], e["evidence_sentence"]) not in seen
            )
            if source_doc_id and source_doc_id not in node["source_doc_ids"]:
                node["source_doc_ids"].append(source_doc_id)
            node["support_count"] = max(node["support_count"], len(node["source_doc_ids"]))
        else:
            node_map[key] = {
                "id": f"{event_id}-{len(node_map)}-{key[1]}",
                "name": name,
                "entity_type": entity_type,
                "description": None,
                "evidence": list(evidence),
                "extraction_mode": "explicit",
                "confidence": 0.8,
                "support_count": 1,
                "source_doc_ids": [source_doc_id] if source_doc_id else [],
                "attributes": {},
                "reasoning": None,
            }
        return node_map[key]["id"]

    def add_edge(subject: str, predicate: str, obj: str,
                 evidence: list[dict[str, Any]], source_doc_id: str) -> None:
        predicate = _norm_predicate(predicate)
        subject, obj = (subject or "").strip(), (obj or "").strip()
        if not subject or not obj or not predicate:
            return
        key = (_normalise_name(subject), predicate, _normalise_name(obj))
        if key in edge_map:
            edge = edge_map[key]
            seen = {(e["source_doc_id"], e["evidence_sentence"]) for e in edge["evidence"]}
            edge["evidence"].extend(
                e for e in evidence
                if (e["source_doc_id"], e["evidence_sentence"]) not in seen
            )
            if source_doc_id and source_doc_id not in edge["source_doc_ids"]:
                edge["source_doc_ids"].append(source_doc_id)
            edge["support_count"] = max(edge["support_count"], len(edge["source_doc_ids"]))
        else:
            edge_map[key] = {
                "id": f"{key[0]}_{predicate}_{key[2]}",
                "subject_id": "", "object_id": "",  # 二次解析时回填
                "subject_name": subject,
                "predicate": predicate,
                "object_id_": "",
                "object_name": obj,
                "evidence": list(evidence),
                "extraction_mode": "explicit",
                "confidence": 0.8,
                "support_count": 1,
                "source_doc_ids": [source_doc_id] if source_doc_id else [],
                "reasoning": None,
            }

    for doc, data in zip(docs, per_doc):
        if not data:
            continue
        doc_id = doc["doc_id"]
        name_to_id: dict[str, str] = {}

        for raw in data.get("entities", []) or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            etype = str(raw.get("type", "") or raw.get("entity_type", "")).strip()
            if not name or not etype:
                continue
            ev = _evidence_items(str(raw.get("evidence_sentence", "")), doc_id)
            node_id = add_node(name, etype, ev, doc_id)
            if node_id:
                name_to_id[_normalise_name(name)] = node_id

        chain_names: list[str] = []
        chain_data = data.get("risk_chain") or {}
        if isinstance(chain_data, dict):
            for slot, slot_type in CHAIN_SLOTS:
                slot_name = str(chain_data.get(slot, "")).strip()
                if not slot_name:
                    chain_names.append("")
                    continue
                ev = _evidence_items(str(chain_data.get("evidence_sentence", "")), doc_id)
                node_id = add_node(slot_name, slot_type, ev, doc_id)
                name_to_id.setdefault(_normalise_name(slot_name), node_id)
                chain_names.append(slot_name)
        else:
            chain_names = [""] * len(CHAIN_SLOTS)

        for raw in data.get("relations", []) or []:
            if not isinstance(raw, dict):
                continue
            subject = str(raw.get("subject", "")).strip()
            obj = str(raw.get("object", "")).strip()
            if _normalise_name(subject) not in name_to_id or _normalise_name(obj) not in name_to_id:
                continue  # 端点必须在实体列表中
            ev = _evidence_items(str(raw.get("evidence_sentence", "")), doc_id)
            add_edge(subject, str(raw.get("predicate", "")), obj, ev, doc_id)

        # 链边: 相邻槽位均存在时构建(canonical 谓词由槽位结构决定)
        chain_ev = _evidence_items(str(chain_data.get("evidence_sentence", "")), doc_id) \
            if isinstance(chain_data, dict) else []
        for i, j, pred in CHAIN_EDGES:
            if chain_names[i] and chain_names[j]:
                add_edge(chain_names[i], pred, chain_names[j], chain_ev, doc_id)

    # 回填边的端点 id
    norm_to_id: dict[str, str] = {}
    for (_, etype), node in node_map.items():
        norm_to_id.setdefault(_normalise_name(node["name"]), node["id"])
    for edge in edge_map.values():
        edge["subject_id"] = norm_to_id.get(_normalise_name(edge["subject_name"]), "")
        edge["object_id"] = norm_to_id.get(_normalise_name(edge["object_name"]), "")

    nodes = list(node_map.values())
    edges = [e for e in edge_map.values() if e["subject_id"] and e["object_id"]]
    # 清理临时键
    for e in edges:
        e.pop("object_id_", None)

    # knowledge_statements: 每条边的每个 evidence 一条(与主 pipeline 口径一致)
    statements = []
    for edge in edges:
        for ev in edge["evidence"]:
            statements.append({
                "id": f"stmt-{len(statements)}",
                "subject_id": edge["subject_id"],
                "predicate": edge["predicate"],
                "object_id": edge["object_id"],
                "evidence": ev,
                "extraction_mode": edge["extraction_mode"],
                "confidence": edge["confidence"],
                "support_count": edge["support_count"],
            })

    incident_node = {
        "id": event_id,
        "name": docs[0]["title"] if docs else event_id,
        "entity_type": "AIRiskIncident",
        "description": event_id,
        "evidence": [],
        "extraction_mode": "completed",
        "confidence": 1.0,
        "support_count": len(docs),
        "source_doc_ids": [d["doc_id"] for d in docs],
        "attributes": {},
        "reasoning": None,
    }

    return {
        "event_id": event_id,
        "incident_node": incident_node,
        "nodes": nodes,
        "edges": edges,
        "knowledge_statements": statements,
        "support_statistics": {},
        "report_count": len(docs),
        "first_seen": None,
        "last_seen": None,
    }


def run_event(client: DirectClient, event: dict[str, Any], cases: dict) -> dict[str, Any]:
    documents = build_documents_for_event(event, cases)
    docs = [
        {"doc_id": r.doc_id, "title": r.title or "", "content": r.content}
        for r in documents
    ]
    per_doc = [_extract_doc(client, d) for d in docs]
    return _merge_union(event.get("incident_id", ""), docs, per_doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-only baseline (Baseline 1)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    data = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold_ids = sorted(
        item["event_id"] for item in data.get("gold_standard", []) if item.get("event_id")
    )
    if args.max_events:
        gold_ids = gold_ids[: args.max_events]
    print(f"Gold events: {len(gold_ids)}")
    print(f"Primary model: {PRIMARY_MODEL}, fallback: {FALLBACK_MODEL}")
    print(f"Endpoint: {BASE_URL}")

    events = {e.get("incident_id"): e for e in load_events()}
    todo = []
    for eid in gold_ids:
        out = OUTPUT_DIR / eid / "event_subgraph.json"
        if out.exists() and not args.force:
            continue
        if eid not in events:
            print(f"  [skip] {eid} not in dataset")
            continue
        todo.append(events[eid])
    print(f"Pending events: {len(todo)} (cached skipped)\n")
    if not todo:
        return

    client = DirectClient()
    cases = load_cases()

    start = time.monotonic()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_event, client, ev, cases): ev for ev in todo
        }
        for future in as_completed(futures):
            ev = futures[future]
            eid = ev.get("incident_id", "?")
            try:
                subgraph = future.result()
                out = OUTPUT_DIR / eid / "event_subgraph.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(subgraph, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                done += 1
                n_e = len(subgraph["edges"])
                print(f"  [{done + failed}/{len(todo)}] {eid}: "
                      f"{n_e} edges, {len(subgraph['nodes'])} nodes")
            except Exception as exc:
                failed += 1
                print(f"  [{done + failed}/{len(todo)}] {eid} FAILED: {str(exc)[:150]}")

    elapsed = time.monotonic() - start
    print(f"\nDone: {done} ok, {failed} failed, {elapsed / 60:.1f} min")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
