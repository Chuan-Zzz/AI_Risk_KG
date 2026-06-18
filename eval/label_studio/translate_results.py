"""Translate Level 2/Level 3 pre-extraction results to bilingual display.

Reads annotation_tasks.json, translates value/evidence/reasoning via LLM API,
and updates the HTML cards with bilingual (EN + CN) display.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from html import escape as h
from pathlib import Path

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings()

sys.stdout.reconfigure(encoding="utf-8")

# ── API config ──────────────────────────────────────────────────────────

API_KEY = os.environ.get("LLM_API_KEY", "sk-0sVPVBiDcmZnUt05OPs486uHW96ctnhSqC3TyKVsUW1AUU7N")
BASE_URL = "https://api.meai.cloud/v1"
MODEL = "glm-5.1"

LEVEL2_DIR = Path("eval/label_studio/level2_risk_chain")
LEVEL3_DIR = Path("eval/label_studio/level3_inference")

SYSTEM_PROMPT = """你是一位专业的AI风险领域翻译专家。请将以下英文内容翻译为流畅准确的中文。

规则：
1. 保持专业术语的准确性
2. 每行输入对应一行输出，保持对应关系
3. 只输出翻译结果，不要添加解释或编号
4. 专有名词（如公司名、人名、产品名）保留英文原文"""


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def translate_batch(texts: list[str], batch_size: int = 30) -> dict[str, str]:
    """Translate a list of English texts to Chinese.

    Returns dict mapping original English -> Chinese translation.
    """
    session = _make_session()
    result: dict[str, str] = {}

    # Filter out empty strings and deduplicate
    unique_texts = list(dict.fromkeys(t for t in texts if t.strip()))
    print(f"  Translating {len(unique_texts)} unique strings...")

    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i:i + batch_size]
        numbered = "\n".join(f"[{j}] {t}" for j, t in enumerate(batch))

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请翻译以下内容：\n\n{numbered}"},
            ],
            "temperature": 0.1,
        }

        for attempt in range(3):
            try:
                resp = session.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                    verify=False,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                break
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    print(f"  Skipping batch starting at index {i}")
                    continue
                time.sleep(2 * (attempt + 1))

        # Parse numbered lines
        lines = content.split("\n")
        parsed: dict[int, str] = {}
        for line in lines:
            m = re.match(r'\[(\d+)\]\s*(.+)', line.strip())
            if m:
                parsed[int(m.group(1))] = m.group(2).strip()

        for j, text in enumerate(batch):
            if j in parsed:
                result[text] = parsed[j]
            else:
                # Fallback: try to match by order
                remaining = [parsed[k] for k in sorted(parsed.keys()) if k not in range(j)]
                if remaining:
                    result[text] = remaining[0]

        if i + batch_size < len(unique_texts):
            time.sleep(0.5)

    return result


def _extract_translatable_strings(html: str) -> list[str]:
    """Extract English strings from slot/inference HTML that need translation."""
    strings = []
    # Value: text immediately after font-weight:600 div opening, before any nested tag
    for m in re.finditer(r'<div style="font-weight:600[^"]*">([^<]+)', html):
        val = m.group(1).strip()
        if val and not val.startswith("未"):
            strings.append(val)
    # Evidence text after 证据：
    for m in re.finditer(r'证据：</span>([^<]+)', html):
        val = m.group(1).strip()
        val = re.sub(r'^\s*\[\d+\]\s*', '', val)  # strip [doc_id] prefix
        if val:
            strings.append(val)
    # Reasoning text after 推理链：
    for m in re.finditer(r'推理链：</span>([^<]+)', html):
        val = m.group(1).strip()
        if val:
            strings.append(val)
    return strings


def _replace_with_bilingual(html: str, trans: dict[str, str]) -> str:
    """Replace English text in HTML with bilingual (EN + CN) display."""

    def _strip_doc_prefix(text: str) -> str:
        return re.sub(r'^\s*\[\d+\]\s*', '', text).strip()

    # Replace value: insert CN line after the EN value text
    def replace_value(m):
        en = m.group(1).strip()
        cn = trans.get(en, "")
        if cn:
            return (
                f'<div style="font-weight:600;color:#333;margin-bottom:6px;">'
                f'{h(en)}'
                f'<div style="color:#1565c0;font-size:12px;margin-top:2px;">{h(cn)}</div>'
                f'</div>'
            )
        return m.group(0)

    result = re.sub(
        r'<div style="font-weight:600[^"]*">([^<]+)',
        replace_value,
        html,
    )

    # Replace evidence: strip [doc_id] prefix, match against stripped key
    def replace_evidence(m):
        en_raw = m.group(1).strip()
        en_clean = _strip_doc_prefix(en_raw)
        cn = trans.get(en_clean, "")
        if cn:
            return (
                f'<span style="color:#999;">证据：</span>{h(en_raw)}'
                f'<div style="color:#1565c0;font-size:11px;margin-top:2px;">{h(cn)}</div>'
            )
        return m.group(0)

    result = re.sub(
        r'证据：</span>([^<]+)',
        replace_evidence,
        result,
    )

    # Replace reasoning
    def replace_reasoning(m):
        en = m.group(1).strip()
        cn = trans.get(en, "")
        if cn:
            return (
                f'<span style="color:#999;">推理链：</span>{h(en)}'
                f'<div style="color:#1565c0;font-size:11px;margin-top:2px;">{h(cn)}</div>'
            )
        return m.group(0)

    result = re.sub(
        r'推理链：</span>([^<]+)',
        replace_reasoning,
        result,
    )

    return result


def process_level(tasks: list[dict], slot_keys: list[str]) -> list[dict]:
    """Process Level 2 or Level 3 tasks: translate and update HTML."""
    # Collect all translatable strings
    all_strings: list[str] = []
    for task in tasks:
        for key in slot_keys:
            html = task["data"].get(key, "")
            if html:
                all_strings.extend(_extract_translatable_strings(html))

    print(f"Found {len(all_strings)} strings to translate")
    trans = translate_batch(all_strings)
    print(f"Got {len(trans)} translations")

    # Apply translations
    updated = 0
    for task in tasks:
        for key in slot_keys:
            html = task["data"].get(key, "")
            if html and "font-weight:600" in html:
                new_html = _replace_with_bilingual(html, trans)
                if new_html != html:
                    task["data"][key] = new_html
                    updated += 1

    return tasks


def main():
    # ── Level 2 ──
    print("=== Level 2: Risk Chain ===")
    l2_path = LEVEL2_DIR / "annotation_tasks.json"
    with open(l2_path, encoding="utf-8") as f:
        l2_tasks = json.load(f)
    l2_keys = ["risk_source_pre", "risk_pre", "consequence_pre",
               "impact_pre", "affected_actor_pre", "risk_control_pre"]
    l2_tasks = process_level(l2_tasks, l2_keys)
    with open(l2_path, "w", encoding="utf-8") as f:
        json.dump(l2_tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(l2_tasks)} tasks to {l2_path}")

    # ── Level 3 ──
    print("\n=== Level 3: Inference ===")
    l3_path = LEVEL3_DIR / "annotation_tasks.json"
    with open(l3_path, encoding="utf-8") as f:
        l3_tasks = json.load(f)
    l3_keys = ["purpose_pre", "lifecycle_pre", "domain_pre"]
    l3_tasks = process_level(l3_tasks, l3_keys)
    with open(l3_path, "w", encoding="utf-8") as f:
        json.dump(l3_tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(l3_tasks)} tasks to {l3_path}")


if __name__ == "__main__":
    main()
