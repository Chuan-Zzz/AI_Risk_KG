"""Dataset loading helpers for pipeline entry points."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.core.models import NewsReport

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EVENTS_FILE = _PROJECT_ROOT / "data" / "inferred_event_structure_6124.json"
_GOLD_EVENTS_FILE = _PROJECT_ROOT / "data" / "gold_standard_100_events.json"
_EVAL_500_EVENTS_FILE = _PROJECT_ROOT / "data" / "eval_dataset_500.json"
_CASES_JSONL_FILE = _PROJECT_ROOT / "data" / "eval_cases.jsonl"
_CASES_TRANSLATED_FILE = _PROJECT_ROOT / "data" / "translated_docs.json"

# 网页模板导航关键词（Daily Mail 等新闻网站的侧边栏菜单）
_WEB_TEMPLATE_KEYWORDS = {
    "shopping", "best buys", "discounts", "black friday", "my profile",
    "logout", "share selection", "sign in", "subscribe", "newsletter",
    "follow", "topics", "cybersecurity", "celebrities",
}


def _clean_web_template_content(content: str, title: str) -> str:
    """清洗网页抓取内容中的模板导航和无关新闻标题。

    某些文档（如 Daily Mail）的 content 字段包含大量网页侧边栏导航
    和无关新闻标题，导致 LLM 提取错误或触发内容审核拒绝。
    本函数检测并剥离这些模板内容，只保留与标题相关的正文。

    仅当检测到明确的网页模板模式时才执行清洗，避免影响正常文档。
    """
    if not content or len(content) < 500:
        return content

    lines = content.split("\n")

    # 检测前 40 行是否有网页模板导航模式
    template_line_count = 0
    for line in lines[:40]:
        stripped = line.strip().lower()
        if not stripped:
            continue
        if any(kw in stripped for kw in _WEB_TEMPLATE_KEYWORDS):
            template_line_count += 1

    if template_line_count < 3:
        return content

    # 有网页模板，需要找到实际正文起点
    # 策略：从 title 提取关键词，在内容中找到第一个匹配的长行
    title_words = set(re.findall(r"[a-zA-Z]{3,}", title.lower())) if title else set()
    # 过滤掉过于通用的词
    title_words -= {"the", "and", "for", "with", "from", "that", "this", "says", "said"}

    best_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 40:
            continue
        line_lower = stripped.lower()
        # 检查这一行是否包含标题关键词
        if title_words and sum(1 for w in title_words if w in line_lower) >= 2:
            best_start = i
            break

    if best_start < 0:
        # 标题匹配失败，退而求其次：找第一个超过 200 字符的实质段落
        for i, line in enumerate(lines):
            if len(line.strip()) > 200:
                best_start = i
                break

    if best_start > 0:
        cleaned = "\n".join(lines[best_start:])
        # 同时剥离尾部模板（如果尾部有大量空行或短行）
        cleaned_lines = cleaned.rstrip().split("\n")
        # 从尾部往前找，删除短行（<20字符的空行或导航残留）
        while cleaned_lines and len(cleaned_lines[-1].strip()) < 20:
            cleaned_lines.pop()
        return "\n".join(cleaned_lines)

    return content


def load_cases() -> dict[str, dict]:
    """Load case documents from the first available supported dataset file."""
    if _CASES_JSONL_FILE.exists():
        cases: dict[str, dict] = {}
        with open(_CASES_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                case_id = str(obj.get("id") or obj.get("original_id", ""))
                if case_id:
                    cases[case_id] = obj
        return cases

    if _CASES_TRANSLATED_FILE.exists():
        with open(_CASES_TRANSLATED_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cases = {}
        for item in raw.values():
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            cases[doc_id] = {
                "id": doc_id,
                "event_id": item.get("event_id"),
                "title": item.get("title_en") or item.get("title_cn") or "",
                "text": item.get("text_en") or item.get("text_cn") or "",
                "summary": item.get("summary_en") or item.get("summary_cn") or "",
                "source": item.get("source_name") or "translated_docs",
                "language": "en" if item.get("text_en") else "zh",
            }
        return cases

    raise FileNotFoundError(
        "No supported case dataset found. Expected data/eval_cases.jsonl "
        "or data/translated_docs.json."
    )


def load_events(dataset: str = "all") -> list[dict]:
    """Load a normalized event list for the requested built-in dataset.

    Every returned event uses ``incident_id`` and ``ids`` so command entry
    points do not need dataset-specific branches.
    """
    if dataset == "all":
        path, key = _EVENTS_FILE, "events"
    elif dataset == "gold":
        path, key = _GOLD_EVENTS_FILE, "gold_standard"
    elif dataset == "eval500":
        path, key = _EVAL_500_EVENTS_FILE, "events"
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events: list[dict] = []
    for event in data.get(key, []):
        normalized = dict(event)
        normalized["incident_id"] = event.get("incident_id") or event.get("event_id")
        normalized["ids"] = event.get("ids") or event.get("labeling_cases") or []
        events.append(normalized)
    return events


def build_documents_for_event(event: dict, cases: dict[str, dict]) -> list[NewsReport]:
    documents: list[NewsReport] = []
    event_id = event.get("event_id") or event.get("incident_id", "unknown")
    case_ids = event.get("ids", [])

    for cid in case_ids:
        case = cases.get(str(cid))
        if not case:
            continue

        content = (
            case.get("text")
            or case.get("text_en")
            or case.get("new_summary")
            or case.get("description")
            or ""
        )
        title = case.get("title") or case.get("title_en") or case.get("title_cn") or ""
        # 清洗网页模板导航和无关新闻标题
        content = _clean_web_template_content(content, title)
        summary = (
            case.get("summary")
            or case.get("new_summary")
            or case.get("summary_en")
            or case.get("summary_cn")
            or ""
        )
        source_name = case.get("from_database") or case.get("source") or case.get("source_name")
        language = case.get("language")
        if not language:
            language = "en" if case.get("from_database") == "aiid" or case.get("text_en") else "zh"

        documents.append(
            NewsReport(
                doc_id=str(cid),
                event_id=event_id,
                title=title,
                content=content,
                summary=summary if summary else None,
                source_name=source_name,
                url=case.get("case_link") or case.get("from_url") or case.get("url"),
                publish_time=case.get("release_date") or case.get("publish_time"),
                language=language,
            )
        )

    return documents
