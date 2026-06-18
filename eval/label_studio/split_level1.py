"""Split Level 1 annotation tasks into two balanced halves.

For events with >5 documents, keep only the 5 docs with the most
pre-extracted entity mentions (based on predictions). Then split into
a/b by document count balance.

Predictions are rebuilt by re-finding positions in the new text to
avoid offset remapping errors.

Handles HTML (HyperText) format:
- Predictions use rendered-text offsets (after stripping HTML tags and decoding entities)
- Section boundaries are found in rendered text
- HTML sections are extracted as complete blocks
- Predictions are rebuilt in rendered-text coordinates
"""

from __future__ import annotations

import json
import re
import sys
from html import unescape as _unescape_html
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LEVEL1_DIR = Path("eval/label_studio/level1_entity")
MAX_DOCS = 5


def _html_to_rendered_and_map(text_html: str) -> tuple[str, list[int]]:
    """Convert HTML to rendered text with a rendered→HTML position mapping.

    Returns (rendered_text, mapping) where mapping[ri] = position in text_html
    where rendered char ri was produced from.
    """
    i = 0
    html_len = len(text_html)
    rendered_chars: list[str] = []
    rendered_map: list[int] = []

    while i < html_len:
        if text_html[i] == "<":
            gt = text_html.find(">", i)
            if gt < 0:
                i += 1
            else:
                i = gt + 1
                continue
        elif text_html[i] == "&":
            semi = text_html.find(";", i)
            if semi < 0:
                rendered_chars.append(text_html[i])
                rendered_map.append(i)
                i += 1
            else:
                entity_str = text_html[i:semi + 1]
                decoded = _unescape_html(entity_str)
                for ch in decoded:
                    rendered_chars.append(ch)
                    rendered_map.append(i)
                i = semi + 1
        else:
            rendered_chars.append(text_html[i])
            rendered_map.append(i)
            i += 1

    return "".join(rendered_chars), rendered_map


def _is_html_text(text: str) -> bool:
    """Check if text contains HTML tags."""
    return "<div" in text or "<span" in text


def _find_doc_sections(rendered_text: str) -> list[tuple[str, int, int]]:
    """Find (doc_id, rendered_start, rendered_end) for each Document section."""
    headers = list(re.finditer(r'Document (\d+):', rendered_text))
    sections = []
    for i, m in enumerate(headers):
        did = m.group(1)
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(rendered_text)
        sections.append((did, start, end))
    return sections


def _count_preds_in_rendered_range(predictions: list[dict], start: int, end: int) -> int:
    """Count predictions whose start offset (in rendered-text coords) is in [start, end)."""
    items = predictions[0]["result"] if predictions and "result" in predictions[0] else predictions
    return sum(1 for p in items if start <= p["value"]["start"] < end)


def _find_html_section_boundaries(html_text: str, rendered_map: list[int],
                                   rendered_start: int, rendered_end: int) -> tuple[int, int]:
    """Map rendered-text range [rendered_start, rendered_end) to HTML source range.

    Returns (html_start, html_end) that covers the complete HTML for this section.
    We use the rendered map to find the HTML positions, then expand to include
    any trailing HTML tags (closing divs, etc.).
    """
    html_start = rendered_map[rendered_start] if rendered_start < len(rendered_map) else 0
    html_end = rendered_map[rendered_end] if rendered_end < len(rendered_map) else len(html_text)

    # Expand html_end to include any closing tags that follow
    while html_end < len(html_text) and html_text[html_end:html_end + 6] in ('</div>', '</div>'):
        html_end += 6

    return html_start, html_end


def _rebuild_predictions_for_rendered(new_rendered: str, old_predictions: list[dict],
                                      kept_doc_ids: list[str]) -> list[dict]:
    """Rebuild predictions for new rendered text after doc trimming.

    Predictions use rendered-text offsets. We search for entity text
    in the new rendered text after the appropriate doc header.
    """
    new_predictions = []
    seen: set[tuple] = set()

    for p in old_predictions:
        v = p["value"]
        text_val = v.get("text", "")
        labels = v.get("labels", [])
        if not text_val:
            continue

        dedup = (text_val, tuple(labels))
        if dedup in seen:
            continue

        for did in kept_doc_ids:
            header = f"Document {did}:"
            hpos = new_rendered.find(header)
            if hpos < 0:
                continue
            pos = new_rendered.find(text_val, hpos)
            if pos >= 0:
                key = (text_val, tuple(labels), pos)
                if key in seen:
                    break
                seen.add(key)
                seen.add(dedup)

                new_predictions.append({
                    "id": p.get("id", ""),
                    "result_id": p.get("result_id", ""),
                    "from_name": p.get("from_name", ""),
                    "to_name": p.get("to_name", ""),
                    "type": p.get("type", ""),
                    "value": {
                        "start": pos,
                        "end": pos + len(text_val),
                        "text": text_val,
                        "labels": labels,
                    },
                })
                break

    return new_predictions


def filter_task_docs(task: dict) -> dict:
    """If a task has >MAX_DOCS documents, keep only the top MAX_DOCS by mention count."""
    text = task["data"].get("text", "")
    predictions = task.get("predictions", [])
    if not text or not predictions:
        return task

    is_html = _is_html_text(text)

    if is_html:
        rendered_text, rendered_map = _html_to_rendered_and_map(text)
    else:
        rendered_text = text
        rendered_map = list(range(len(text)))

    sections = _find_doc_sections(rendered_text)
    if len(sections) <= MAX_DOCS:
        return task

    # Count predictions per doc section (using rendered-text coords)
    scored = []
    for did, r_start, r_end in sections:
        count = _count_preds_in_rendered_range(predictions, r_start, r_end)
        scored.append((did, r_start, r_end, count))

    # Keep top MAX_DOCS by prediction count, then sort by original position
    scored.sort(key=lambda x: x[3], reverse=True)
    kept = scored[:MAX_DOCS]
    kept.sort(key=lambda x: x[1])  # sort by rendered-text start

    kept_doc_ids = [did for did, _, _, _ in kept]

    if is_html:
        # Extract complete HTML sections for kept docs
        new_parts = []
        for _, r_start, r_end, _ in kept:
            html_start = rendered_map[r_start] if r_start < len(rendered_map) else 0
            html_end = rendered_map[r_end] if r_end < len(rendered_map) else len(text)
            # Expand to include closing tags
            while html_end < len(text) and text[html_end:html_end + 6] in ('</div>',):
                html_end += 6
            new_parts.append(text[html_start:html_end])
        inner = "".join(new_parts)
        # Re-add the outer wrapper that was lost during section extraction
        new_text = (
            '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.6">'
            + inner
            + '</div>'
        )
        new_rendered, _ = _html_to_rendered_and_map(new_text)
    else:
        new_text = "".join(text[r_start:r_end] for _, r_start, r_end, _ in kept)
        new_rendered = new_text

    # Rebuild predictions for the new text (using rendered-text coords)
    old_preds = task.get("predictions", [])
    if old_preds and isinstance(old_preds[0], dict) and "result" in old_preds[0]:
        old_pred_items = old_preds[0]["result"]
    else:
        old_pred_items = old_preds

    new_pred_items = _rebuild_predictions_for_rendered(
        new_rendered, old_pred_items, kept_doc_ids)

    # Update meta_info
    meta = task["data"].get("meta_info", "")
    meta = re.sub(r'文档数: \d+', f'文档数: {len(kept)}', meta)

    return {
        "data": {
            "meta_info": meta,
            "event_id": task["data"].get("event_id", ""),
            "text": new_text,
        },
        "predictions": [{"result": new_pred_items}],
    }


def count_task_docs(task: dict) -> int:
    """Count documents in the text field."""
    text = task["data"].get("text", "")
    return len(set(m.group(1) for m in re.finditer(r'Document (\d+):', text)))


def main():
    src = LEVEL1_DIR / "annotation_tasks.json"
    with open(src, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"Original tasks: {len(tasks)}")

    # Step 1: Filter to max 5 docs per event
    filtered = []
    trimmed = 0
    for task in tasks:
        original_docs = count_task_docs(task)
        new_task = filter_task_docs(task)
        new_docs = count_task_docs(new_task)
        if original_docs > MAX_DOCS:
            trimmed += 1
        filtered.append(new_task)

    print(f"Trimmed {trimmed} events from >{MAX_DOCS} docs to {MAX_DOCS}")

    # Step 2: Split into a/b by document count balance
    indexed = list(enumerate(filtered))
    indexed.sort(key=lambda x: count_task_docs(x[1]), reverse=True)

    group_a: list[dict] = []
    group_b: list[dict] = []
    docs_a = 0
    docs_b = 0

    for _, task in indexed:
        ndocs = count_task_docs(task)
        if docs_a <= docs_b:
            group_a.append(task)
            docs_a += ndocs
        else:
            group_b.append(task)
            docs_b += ndocs

    def event_sort_key(t):
        eid = t["data"].get("event_id", "")
        parts = eid.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    group_a.sort(key=event_sort_key)
    group_b.sort(key=event_sort_key)

    for suffix, group, total_docs in [("a", group_a, docs_a), ("b", group_b, docs_b)]:
        out = LEVEL1_DIR / f"annotation_tasks_{suffix}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(group, f, ensure_ascii=False, indent=2)
        print(f"  {suffix}: {len(group)} tasks, {total_docs} docs -> {out}")


if __name__ == "__main__":
    main()
