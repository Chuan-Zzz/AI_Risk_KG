#!/usr/bin/env python3
"""Compute human agreement and GLM-judge vs human metrics for the ESR
validation experiment (3-point labeling, judge-adjudicated sample).

Design (advisor, 2026-08-18): statement-evidence pairs sampled uniformly
from the edges the GLM judge adjudicated; annotators label each pair
Supported(2) / Partially(1) / Unsupported(0). With one export (--a only)
the single human labeling is the gold standard; with two exports (--a and
--b) inter-annotator agreement is computed and disagreements go to an
arbitrator. The GLM verdict is compared against the human labels with
precision / recall / F1.

Because the GLM verdict is binary, human 3-point labels are folded two
ways (both reported):
  strict  : positive = {2}                    negative = {0,1}
  lenient : positive = {2,1}                  negative = {0}

Also reports the implied graph-wide ESR estimate: the literal-match layer
is deterministic (9,104/12,361 supported), and the adjudicated layer is
re-estimated from the human consensus positive rate in the sample,
ESR_adj = (9104 + 3257 * p_hat) / 12361, with a bootstrap 95% CI over the
sample. This shows how judge error propagates into the reported 84.4%.

Inputs:
    --sample  eval/results/esr_validation_sample.jsonl
    --map     eval/label_studio/esr_validation/val_key_map.json
    --a / --b Label Studio export JSONs (one annotator each)
    --arb     optional arbitrator export (only disagreeing tasks)
    --out     optional JSON output path
"""
import argparse
import json
import sys
from pathlib import Path

CHOICE_NAME = "esr_choice"
N_EDGES = 12361      # semantic edges in the evaluation graph
N_LIT = 9104         # literal-matched (deterministically supported)
N_JUDGED = 3257      # judge-adjudicated pool the sample is drawn from


def parse_choice(s: str) -> int:
    """'2 · 支持 Supported' -> 2"""
    try:
        return int(str(s).split("·")[0].strip())
    except (ValueError, AttributeError):
        return -1


def load_export(path):
    """Parse one Label Studio export -> {val_id: (completed_by, label)}."""
    data = json.load(open(path, encoding="utf-8"))
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if isinstance(tasks, dict):
        sys.exit(f"{path}: unexpected export shape (dict without 'tasks')")
    out, users = {}, set()
    for t in tasks:
        vid = (t.get("data") or {}).get("val_id")
        if not vid:
            continue
        best = None
        for ann in t.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            val = None
            for res in ann.get("result") or []:
                if res.get("from_name") == CHOICE_NAME:
                    choices = (res.get("value") or {}).get("choices") or []
                    if choices:
                        v = parse_choice(choices[0])
                        if v in (0, 1, 2):
                            val = v
            if val is None:
                continue
            stamp = ann.get("updated_at") or ""
            if best is None or stamp >= best[0]:
                best = (stamp, ann.get("completed_by"), val)
        if best:
            out[vid] = (best[1], best[2])
            users.add(best[1])
    if len(users) > 1:
        sys.exit(f"{path}: contains {len(users)} annotators; "
                 "export one annotator per file")
    return out


def cohen_kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def prf(gold, pred):
    """Binary gold/pred lists -> accuracy/precision/recall/F1 + kappa."""
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g != 1 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p != 1 for g, p in zip(gold, pred))
    tn = sum(g != 1 and p != 1 for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    acc = (tp + tn) / len(gold)
    kap = cohen_kappa(gold, pred, [0, 1])
    return {"n": len(gold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4),
            "f1": None if f1 is None else round(f1, 4),
            "cohens_kappa": round(kap, 4)}


def confusion(human3, judge):
    """3x2 confusion: human label x judge verdict counts."""
    conf = {}
    for h, j in zip(human3, judge):
        conf[(h, j)] = conf.get((h, j), 0) + 1
    return {f"human_{h}|judge_{'sup' if j else 'unsup'}": c
            for (h, j), c in sorted(conf.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--a", required=True, help="annotator A export file")
    ap.add_argument("--b", default=None,
                    help="optional annotator B export file; omit for "
                         "single-annotator mode")
    ap.add_argument("--arb", default=None,
                    help="arbitrator export file for disagreeing tasks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sample = {r["edge_key"]: r
              for r in (json.loads(l) for l in open(args.sample, encoding="utf-8"))}
    keymap = json.load(open(args.map, encoding="utf-8"))  # val_id -> edge_key

    def relabel(path):
        raw = load_export(path)
        return {keymap[v]: lab for v, (_, lab) in raw.items() if v in keymap}

    A = relabel(args.a)
    B = relabel(args.b) if args.b else {}
    arb = relabel(args.arb) if args.arb else {}

    print(f"parsed: A={len(A)}, B={len(B)}, arb={len(arb)} "
          f"(sample={len(sample)})", file=sys.stderr)
    for name, d in (("A", A), ("B", B)):
        if d:
            miss = len(sample) - len(d)
            if miss:
                print(f"WARNING: {name} missing {miss} tasks", file=sys.stderr)

    if B:
        both = [k for k in sample if k in A and k in B]
        disagree = [k for k in both if A[k] != B[k]]
        pending = [k for k in disagree if k not in arb]
        # consensus: agreement -> shared label; disagreement -> arbitrator
        resolved = {k: A[k] for k in both if A[k] == B[k]}
        resolved.update({k: arb[k] for k in disagree if k in arb})

        a3 = [A[k] for k in both]
        b3 = [B[k] for k in both]
        inter = {
            "n": len(both),
            "raw_agreement": round(sum(A[k] == B[k] for k in both) / len(both), 4),
            "cohens_kappa_3class": round(cohen_kappa(a3, b3, [0, 1, 2]), 4),
            "label_distribution_A": {str(l): a3.count(l) for l in (2, 1, 0)},
            "label_distribution_B": {str(l): b3.count(l) for l in (2, 1, 0)},
            "disagreements": len(disagree),
            "arbitrated": len(disagree) - len(pending),
            "pending_arbitration": len(pending),
        }
        design = ("3-point labeling of judge-adjudicated edges; "
                  "GLM verdict vs arbitrated two-annotator human consensus")
    else:
        # single-annotator mode: the one human labeling IS the gold standard
        keys_a = [k for k in sample if k in A]
        resolved = {k: A[k] for k in keys_a}
        inter = {
            "n": len(keys_a),
            "mode": "single_annotator",
            "label_distribution": {str(l): sum(1 for k in keys_a if A[k] == l)
                                   for l in (2, 1, 0)},
        }
        design = ("3-point labeling of judge-adjudicated edges; "
                  "GLM verdict vs a single human annotator (author)")

    keys = sorted(resolved)
    human3 = [resolved[k] for k in keys]
    judge = [int(bool(sample[k]["judge_verdict"])) for k in keys]
    h_strict = [1 if h == 2 else 0 for h in human3]
    h_lenient = [1 if h in (2, 1) else 0 for h in human3]

    result = {
        "design": design,
        "sample_size": len(sample),
        "n_resolved": len(keys),
        "inter_annotator": inter,
        "judge_vs_human": {
            "strict (S=positive, P/U=negative)": prf(h_strict, judge),
            "lenient (S/P=positive, U=negative)": prf(h_lenient, judge),
        },
        "confusion_human3_x_judge": confusion(human3, judge),
    }

    # implied graph-wide ESR estimate from the human consensus
    import random
    if keys:
        n_pos = sum(h_strict)
        p_hat = n_pos / len(keys)
        esr_adj = (N_LIT + N_JUDGED * p_hat) / N_EDGES
        rng = random.Random(42)
        boots = []
        for _ in range(10000):
            s = rng.choices(h_strict, k=len(h_strict))
            ph = sum(s) / len(s)
            boots.append((N_LIT + N_JUDGED * ph) / N_EDGES)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        result["implied_graph_wide_esr"] = {
            "note": f"literal layer {N_LIT}/{N_EDGES} deterministic + "
                    f"adjudicated layer {N_JUDGED} re-estimated at the "
                    f"sample strict-positive rate",
            "sample_strict_positive_rate": round(p_hat, 4),
            "esr_adjusted_pct": round(esr_adj * 100, 1),
            "bootstrap95_ci_pct": [round(lo * 100, 1), round(hi * 100, 1)],
            "reported_esr_llm_pct": 84.4,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nsaved -> {args.out}", file=sys.stderr)
    if B and pending:
        print(f"\n{len(pending)} disagreements await arbitration; "
              f"re-run with --arb", file=sys.stderr)


if __name__ == "__main__":
    main()
