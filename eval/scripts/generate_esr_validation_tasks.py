#!/usr/bin/env python3
"""Generate Label Studio tasks for the ESR human-validation experiment
(advisor's design, 2026-08-18).

Sample N statement-evidence pairs UNIFORMLY AT RANDOM from the semantic
edges of the full-ontorisk evaluation graph that the GLM judge actually
adjudicated (literal-unmatched edges carrying a judgment; 3,257 of the
12,361 edges over the 100 gold-standard events), let two annotators
independently label each pair on a 3-point scale (Supported / Partially
supported / Unsupported), then compute inter-annotator agreement and
GLM-judge vs human precision / recall / F1 with
compute_esr_validation_metrics.py.

Verification target: the GLM judge's verdict on the adjudicated subset
(the only component of ESR_LLM that is not deterministic). The
deterministic literal-match layer (73.6% of edges) is not sampled.

Fidelity: annotators see exactly the statement and ALL its recorded
evidence sentences (with source doc ids) - a pure statement-evidence
pair judgment, no document context is shown.

Blind design: task payloads carry NO verdict, NO literal-match flag;
each task has an opaque val_id. val_id -> edge key (with verdict
metadata) is written to sample + key map for the metric script.

Usage:
    python eval/scripts/generate_esr_validation_tasks.py [--n 500] [--seed 2026]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR / "scripts"))
from compute_ablation_metrics import (  # noqa: E402
    _evidence_fully_supported,
    _load_gold_ids,
    _load_json,
    _load_subgraph,
    _node_map,
    _semantic_edges,
)

PROJECT = EVAL_DIR.parent
DOCS = PROJECT / "data" / "eval_cases.jsonl"
JUDGMENTS = EVAL_DIR / "results" / "ablation_llm_judgments.json"
OUT_DIR = EVAL_DIR / "label_studio" / "esr_validation"
SAMPLE_OUT = EVAL_DIR / "results" / "esr_validation_sample.jsonl"

VARIANT = "full_ontorisk"

CONFIG = """<!--
  ESR human validation (advisor design): 3-point labeling of
  statement-evidence pairs sampled uniformly from all 12,361 semantic
  edges. Blind: no model verdict, no literal-match flag.
-->
<View style="display: grid; grid-template: auto/1fr 1fr; column-gap: 1em">
    <style>
        .left, .right { width: 100%; height: 100vh; overflow: auto; }
    </style>

    <view className="left">
        <Header value="Task Info" />
        <Text name="meta" value="$meta_info" style="font-size: 14px; color: #666; padding: 10px; background: #f0f2f5; border-radius: 8px;"/>

        <Header value="Evidence Sentence(s)" />
        <HyperText name="text" value="$evidence_html" inline="true" valueType="text" style="font-size: 14px; line-height: 1.8; padding: 14px; background: #fff; border: 2px solid #4dabf7; border-radius: 8px;"/>
    </view>

    <view className="right">
        <Header value="Extracted Statement - Is it supported by the evidence?" />
        <HyperText name="statement" value="$statement_html" inline="true" valueType="text" style="padding: 12px; background: #fff3cd; border-radius: 8px; font-size: 15px;"/>

        <View style="box-shadow: 2px 2px 5px #999; padding: 16px; margin-top: 1em; border-radius: 8px; background: #fff;">
            <Choices name="esr_choice" toName="text" required="true" choice="single">
                <Choice value="2 · 支持 Supported"/>
                <Choice value="1 · 部分支持 Partially supported"/>
                <Choice value="0 · 不支持 Unsupported"/>
            </Choices>
        </View>

        <View style="box-shadow: 2px 2px 5px #999; padding: 16px; margin-top: 1em; border-radius: 8px; background: #f8f9fa;">
            <Header value="Guidelines" style="color: #495057; font-size: 14px;"/>
            <View style="padding:10px;background:#fff;border-left:3px solid #495057;border-radius:4px;font-size:12px;line-height:1.8;">
                <Text name="guide" value="Does the evidence sentence(s) support the statement (subject --predicate--&gt; object)?&#10;&#10;2 · 支持 Supported:&#10;- the evidence states the same or equivalent semantics as the statement&#10;- a rewrite / paraphrase / summary of the statement also counts&#10;&#10;1 · 部分支持 Partially supported:&#10;- the evidence covers part of the statement but leaves another part unverified (e.g. confirms the subject and relation but not the exact object, or vice versa)&#10;&#10;0 · 不支持 Unsupported:&#10;- the evidence is unrelated to, or contradicts, the statement&#10;- the evidence is too vague or incomplete to verify the statement&#10;&#10;Judge only from the shown evidence sentence(s); do not speculate beyond them."/>
            </View>
        </View>
    </view>
</View>
"""


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def val_id(edge_key: str) -> str:
    return "V-" + hashlib.sha1(edge_key.encode()).hexdigest()[:12]


def load_docs() -> dict[str, str]:
    docs = {}
    with open(DOCS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs[str(d.get("case_id") or d.get("id"))] = (
                d.get("translated_text") or d.get("text") or ""
            )
    return docs


def collect_edges() -> list[dict]:
    """All semantic edges of the full-ontorisk graph over gold events."""
    judgments = _load_json(JUDGMENTS, {})
    docs = load_docs()
    rows = []
    for event_id in _load_gold_ids():
        sg = _load_subgraph(VARIANT, event_id)
        if sg is None:
            continue
        nodes = _node_map(sg)
        edges = _semantic_edges(sg)
        for edge_index, edge in enumerate(edges):
            sub = nodes.get(edge.get("subject_id", ""), {})
            obj = nodes.get(edge.get("object_id", ""), {})
            key = f"{VARIANT}|{event_id}|evidence|{edge_index}"
            matched, reason = _evidence_fully_supported(edge, docs)
            judged = (not matched) and (key in judgments)
            verdict = True if matched else judgments.get(key)
            rows.append({
                "edge_key": key,
                "event_id": event_id,
                "edge_index": edge_index,
                "subject": sub.get("name", edge.get("subject_name", "?")),
                "predicate": edge.get("predicate", "?"),
                "object": obj.get("name", edge.get("object_name", "?")),
                "evidence": [
                    {"source_doc_id": str(e.get("source_doc_id", "")),
                     "evidence_sentence": e.get("evidence_sentence", "")}
                    for e in edge.get("evidence", [])
                ],
                "literal_matched": matched,
                "literal_reason": reason,
                "judge_judged": judged,
                "judge_verdict": (None if not judged else bool(judgments[key])),
                "esr_llm_verdict": (None if verdict is None else bool(verdict)),
            })
    return rows


def build_task(r: dict, idx: int, total: int) -> dict:
    statement_html = (
        '<div style="font-family:Arial,sans-serif;line-height:1.9">'
        f'<div style="font-size:14px;color:#666;">Statement</div>'
        f'<div style="font-size:16px;margin-top:6px;"><b>{esc(r["subject"])}</b> '
        f'<span style="background:#e2e3e5;padding:2px 8px;border-radius:4px;font-size:13px;">'
        f'--{esc(r["predicate"])}--&gt;</span> <b>{esc(r["object"])}</b></div></div>'
    )
    blocks = []
    for i, ev in enumerate(r["evidence"], 1):
        did = ev["source_doc_id"]
        sent = ev["evidence_sentence"]
        blocks.append(
            f'<div style="margin-top:{10 if i == 1 else 16}px;">'
            f'<span style="background:#2c3e50;color:#fff;padding:2px 10px;'
            f'border-radius:4px;font-size:12px;">doc {esc(did or "?")}</span>'
            f'<div style="margin-top:8px;font-size:15px;background:#fff3cd;'
            f'padding:8px;border-radius:4px;">{esc(sent)}</div></div>'
        )
    if not blocks:
        blocks.append(
            '<div style="color:#c0392b;font-size:13px;">'
            '(no evidence sentence recorded for this edge)</div>'
        )
    evidence_html = (
        '<div style="font-family:Arial,sans-serif;line-height:1.9;">'
        + "".join(blocks) + "</div>"
    )
    meta = (f"Type: ESR validation (3-point)\nItem: {idx}/{total}\n"
            f"Event: {r['event_id']}")
    return {"data": {"val_id": val_id(r["edge_key"]), "meta_info": meta,
                     "statement_html": statement_html,
                     "evidence_html": evidence_html}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    print("collecting all semantic edges ...", file=sys.stderr)
    rows = collect_edges()
    n_total = len(rows)
    n_lit = sum(r["literal_matched"] for r in rows)
    n_judged = sum(r["judge_judged"] for r in rows)
    print(f"total edges: {n_total} | literal-matched: {n_lit} "
          f"| judge-adjudicated: {n_judged}", file=sys.stderr)

    rng = random.Random(args.seed)
    pool = [r for r in rows if r["judge_judged"]]
    if args.n > len(pool):
        sys.exit(f"requested {args.n} but judged pool has only {len(pool)}")
    sample = rng.sample(pool, args.n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "label_config.xml").write_text(CONFIG, encoding="utf-8")

    tasks = [build_task(r, i, len(sample)) for i, r in enumerate(sample, 1)]
    json.dump(tasks, open(OUT_DIR / "annotation_tasks.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    key_map = {val_id(r["edge_key"]): r["edge_key"] for r in sample}
    json.dump(key_map, open(OUT_DIR / "validation_key_map.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # sample record (with verdict metadata, for the metric script; NOT uploaded)
    with open(SAMPLE_OUT, "w", encoding="utf-8") as f:
        for r in sample:
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    s_lit = sum(r["literal_matched"] for r in sample)
    s_judged = sum(r["judge_judged"] for r in sample)
    s_no_ev = sum(1 for r in sample if not r["evidence"])
    print(f"sampled {len(sample)} edges (seed={args.seed}): "
          f"literal-matched {s_lit} | judge-adjudicated {s_judged} "
          f"| no-evidence {s_no_ev}", file=sys.stderr)
    print(f"tasks   -> {OUT_DIR / 'annotation_tasks.json'}")
    print(f"config  -> {OUT_DIR / 'label_config.xml'}")
    print(f"sample  -> {SAMPLE_OUT}")


if __name__ == "__main__":
    main()
