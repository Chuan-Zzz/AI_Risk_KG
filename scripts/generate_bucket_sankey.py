"""Bucket-level risk propagation Sankey (Section 7.3, full bucket taxonomy).

Redraw of the bucketed Sankey with larger fonts for the KBS submission.
Reproduces the reference figure exactly (stage colours, node order,
ribbon style, bottom legend) — only typography is enlarged.

Data (offline, no LLM):
  scripts/data/bucket_summary.csv      per-stage bucket counts (node order/height)
  scripts/data/bucket_transitions.csv  adjacent-stage co-occurrence counts (ribbons)

Node order: count descending, "other" bucket forced last, "missing" dropped
(excluded events: RiskSource 25, Consequence 14, Impact 15, AffectedActor 19,
Risk 0 — recorded here as the data-integrity note).

Layout is measured in two passes: text extents are queried from the renderer
before positioning, so inter-column gaps always clear the longest label and
adjacent stage headers never collide.

Output: archive/paper/figures/risk_propagation_sankey_buckets.{pdf,svg,png}
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# --------------------------------------------------------------------------- #
# nature-figure mandatory font + SVG rules (always first, no exceptions)
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",      # editable text in SVG
    "pdf.fonttype": 42,          # editable TrueType text in PDF
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).parent / "data"
SUMMARY_PATH = DATA_DIR / "bucket_summary.csv"
TRANS_PATH = DATA_DIR / "bucket_transitions.csv"
OUT_DIR = PROJECT_ROOT / "archive" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Stage definitions (order, headers, colours — muted seaborn family)
# --------------------------------------------------------------------------- #
STAGES = ["RiskSource", "Risk", "Consequence", "Impact", "AffectedActor"]
STAGE_HEADERS = {
    "RiskSource": "Risk Source",
    "Risk": "Risk Type",
    "Consequence": "Consequence",
    "Impact": "Impact",
    "AffectedActor": "Affected Actor",
}
STAGE_COLORS = {
    "RiskSource":    "#4C72B0",  # muted blue
    "Risk":          "#DD8452",  # muted orange / salmon
    "Consequence":   "#55A868",  # muted green
    "Impact":        "#8172B3",  # muted purple
    "AffectedActor": "#C44E52",  # muted red / maroon
}

# Short display labels, matching the reference figure verbatim.
BUCKET_LABELS = {
    "RiskSource": {
        "facial_recognition": "Facial recognition",
        "autonomous_system": "Autonomous system",
        "deepfake_ncii": "Deepfake / NCII",
        "bias_discrimination": "Algorithmic bias",
        "content_moderation": "Content moderation",
        "training_data": "Training data",
        "hallucination": "Hallucination",
        "fraud_abuse": "Fraud & abuse",
        "voice_clone": "Voice cloning",
        "other": "Other source",
    },
    "Risk": {
        "malicious_misuse": "Malicious & misuse",
        "ai_safety_failure": "AI safety failure",
        "discrimination_toxicity": "Discrimination & toxicity",
        "misinformation": "Misinformation",
        "privacy_security": "Privacy & security",
        "socioeconomic_environmental": "Socioeconomic & env.",
        "hci": "Human-computer interaction",
        "other": "Other risk type",
    },
    "Consequence": {
        "misinformation_spread": "Misinformation spread",
        "privacy_violation": "Privacy violation",
        "physical_harm": "Physical harm",
        "discrimination": "Discrimination",
        "financial_loss": "Financial loss",
        "legal_violation": "Legal violation",
        "ip_infringement": "IP infringement",
        "security_breach": "Security breach",
        "reputational_damage": "Reputational damage",
        "psychological_harm": "Psychological harm",
        "sector_harm": "Sector harm",
        "social_division": "Social division",
        "democratic_harm": "Democratic harm",
        "other": "Other consequence",
    },
    "Impact": {
        "individual_sustained": "Sustained individual harm",
        "safety_security_risk": "Safety & security risk",
        "trust_erosion": "Trust erosion",
        "market_industry": "Market & industry",
        "democratic_erosion": "Democratic erosion",
        "social_norm_shift": "Social norm shift",
        "regulatory_response": "Regulatory sanctions",
        "developmental_harm": "Developmental harm",
        "public_discourse": "Public discourse shift",
        "ai_development": "AI development impact",
        "other": "Other impact",
    },
    "AffectedActor": {
        "consumers_users": "Consumers & users",
        "general_public": "General public",
        "children_students": "Children & students",
        "companies": "Companies",
        "media": "Media & journalists",
        "minorities": "Minorities",
        "government": "Government",
        "patients_health": "Patients",
        "women": "Women & girls",
        "other": "Other actor",
    },
}

# --------------------------------------------------------------------------- #
# Typography — enlarged relative to the reference figure
# --------------------------------------------------------------------------- #
LABEL_FS = 8.5      # node labels
HEADER_FS = 10.5    # stage headers (bold)
LEGEND_FS = 8.5     # bottom legend
LABEL_COLOR = "#2b2b2b"
HEADER_COLOR = "#1a1a1a"

FIG_W = 7.5         # inches (KBS double column)
FIG_H = 3.05
AXES_RECT = [0.0, 0.085, 1.0, 0.885]   # axes spans full width: 1 x-unit = FIG_W inches

BAR_W = 0.011       # node bar width (x units)
GAP_FRAC = 0.012    # base vertical gap between nodes in a column (y units)
GAP_MAX = 0.034     # adaptive-gap ceiling for label-dense columns
GAP_TOL = 0.10      # tolerated label-bbox overlap fraction
LABEL_OFF = 0.006   # label x-offset from bar edge

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #

def load_summary() -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {s: Counter() for s in STAGES}
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["stage"]][row["bucket_id"]] = int(row["count"])
    return counts


def load_transitions() -> Dict[Tuple[str, str], Counter]:
    trans: Dict[Tuple[str, str], Counter] = {
        (STAGES[i], STAGES[i + 1]): Counter()
        for i in range(len(STAGES) - 1)
    }
    with open(TRANS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["from_stage"], row["to_stage"])
            if key in trans:
                trans[key][(row["from_bucket"], row["to_bucket"])] += int(row["count"])
    return trans


def _ribbon_path(x0, y0_top, y0_bot, x1, y1_top, y1_bot):
    """Cubic-bezier ribbon with smooth sides (same geometry as Fig. 2)."""
    cx0 = x0 + (x1 - x0) * 0.4
    cx1 = x0 + (x1 - x0) * 0.6
    verts = [
        (x0, y0_bot),
        (cx0, y0_bot), (cx1, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (cx1, y1_top), (cx0, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY,
    ]
    return mpath.Path(verts, codes)


def _measure_texts() -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Render all labels/headers on a scratch figure and measure their extents.

    Returns (label width per stage in inches, header width per stage in
    inches, label bbox height in inches).
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    renderer = fig.canvas.get_renderer()

    label_w: Dict[str, float] = {}
    label_h = 0.0
    for stage in STAGES:
        w_max = 0.0
        for label in BUCKET_LABELS[stage].values():
            t = ax.text(0.5, 0.5, label, fontsize=LABEL_FS)
            bb = t.get_window_extent(renderer=renderer)
            w_max = max(w_max, bb.width / fig.dpi)
            label_h = max(label_h, bb.height / fig.dpi)
            t.remove()
        label_w[stage] = w_max

    header_w: Dict[str, float] = {}
    for stage in STAGES:
        t = ax.text(0.5, 0.5, STAGE_HEADERS[stage], fontsize=HEADER_FS,
                    fontweight="bold")
        bb = t.get_window_extent(renderer=renderer)
        header_w[stage] = bb.width / fig.dpi
        t.remove()
    plt.close(fig)
    return label_w, header_w, label_h


def main() -> None:
    summary = load_summary()
    transitions = load_transitions()

    # --- Node selection: count desc, "other" last, "missing" dropped -------
    stage_nodes: Dict[str, List[str]] = {}
    excluded: Dict[str, int] = {}
    for stage in STAGES:
        ctr = summary[stage]
        excluded[stage] = ctr.get("missing", 0)
        nodes = [b for b in ctr if b not in ("other", "missing")]
        nodes.sort(key=lambda b: ctr[b], reverse=True)
        if "other" in ctr:
            nodes.append("other")
        stage_nodes[stage] = nodes

    # --- Data-integrity note: ribbons referencing dropped nodes ------------
    dropped_ribbon_count = 0
    for (ls, rs), trans in transitions.items():
        for (lb, rb), cnt in trans.items():
            if lb not in stage_nodes[ls] or rb not in stage_nodes[rs]:
                dropped_ribbon_count += cnt

    # --- Two-pass layout: measure text, then compute column positions -------
    label_w, header_w, label_h_in = _measure_texts()
    in2x = FIG_W  # 1 x-unit = FIG_W inches

    clr = 0.10    # label-to-next-bar clearance (inches)
    hclr = 0.14   # header-to-header clearance (inches)

    def zone(i: int) -> float:
        """Width of the ribbon gap after column i (inches)."""
        s_l, s_r = STAGES[i], STAGES[i + 1]
        # middle labels of column i extend right into this zone
        label_need = (LABEL_OFF * in2x + label_w[s_l] + clr) if 0 < i < 4 else 0.55
        # adjacent stage headers must not collide
        header_need = header_w[s_l] / 2 + hclr + header_w[s_r] / 2
        return max(label_need, header_need, 0.85)

    x_positions = [0.15 / in2x]
    for i in range(4):
        x_positions.append(x_positions[i] + BAR_W + zone(i) / in2x)

    # --- Figure ---------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_top, y_bottom = 0.90, 0.045
    col_height = y_top - y_bottom
    axes_h_in = FIG_H * AXES_RECT[3]
    label_h = label_h_in / axes_h_in   # label bbox height in y units

    # --- Bar geometry per stage (normalised so every column spans equally) --
    # Label-dense columns get an adaptive inter-node gap so that vertically
    # adjacent labels keep at most GAP_TOL bbox overlap (small tail nodes).
    def column_gap(nodes: List[str], ctr: Counter) -> float:
        total = sum(ctr[b] for b in nodes)
        n_pairs = len(nodes) - 1
        gap = GAP_FRAC
        for _ in range(8):
            avail = col_height - gap * n_pairs
            worst_halfsum = min(
                (ctr[a] + ctr[b]) / 2 / total * avail
                for a, b in zip(nodes, nodes[1:])
            )
            need = label_h * (1 - GAP_TOL) - worst_halfsum
            if need <= gap + 1e-4:
                return min(gap, GAP_MAX)
            gap = min(need, GAP_MAX)
            if gap >= GAP_MAX:
                return gap
        return gap

    stage_gaps = {s: column_gap(stage_nodes[s], summary[s]) for s in STAGES}

    stage_bars: Dict[str, Dict[str, dict]] = {}
    for stage in STAGES:
        nodes = stage_nodes[stage]
        ctr = summary[stage]
        gap = stage_gaps[stage]
        total = sum(ctr[b] for b in nodes)
        avail = col_height - gap * (len(nodes) - 1)
        y_cursor = y_top
        bars = {}
        for b in nodes:
            h = (ctr[b] / total) * avail
            bars[b] = {"y_top": y_cursor, "y_bottom": y_cursor - h}
            y_cursor -= h + gap
        stage_bars[stage] = bars

    # --- Ribbons: stack from the top of each node, bundle by target order ----
    for i in range(len(STAGES) - 1):
        left_stage, right_stage = STAGES[i], STAGES[i + 1]
        x_left = x_positions[i] + BAR_W
        x_right = x_positions[i + 1]
        left_bars, right_bars = stage_bars[left_stage], stage_bars[right_stage]
        trans = transitions[(left_stage, right_stage)]

        ltot = sum(summary[left_stage][b] for b in stage_nodes[left_stage])
        rtot = sum(summary[right_stage][b] for b in stage_nodes[right_stage])
        lavail = col_height - stage_gaps[left_stage] * (len(stage_nodes[left_stage]) - 1)
        ravail = col_height - stage_gaps[right_stage] * (len(stage_nodes[right_stage]) - 1)

        left_cursor = {b: left_bars[b]["y_top"] for b in stage_nodes[left_stage]}
        right_cursor = {b: right_bars[b]["y_top"] for b in stage_nodes[right_stage]}

        for lb in stage_nodes[left_stage]:
            for rb in stage_nodes[right_stage]:
                cnt = trans.get((lb, rb), 0)
                if cnt == 0:
                    continue
                hL = cnt / ltot * lavail
                hR = cnt / rtot * ravail
                y0_top = left_cursor[lb]
                y0_bot = y0_top - hL
                y1_top = right_cursor[rb]
                y1_bot = y1_top - hR
                left_cursor[lb] = y0_bot
                right_cursor[rb] = y1_bot
                ax.add_patch(mpatches.PathPatch(
                    _ribbon_path(x_left, y0_top, y0_bot, x_right, y1_top, y1_bot),
                    facecolor=STAGE_COLORS[left_stage],
                    alpha=0.35,
                    edgecolor="none",
                    zorder=1,
                ))

    # --- Node bars -----------------------------------------------------------
    for i, stage in enumerate(STAGES):
        x = x_positions[i]
        for b in stage_nodes[stage]:
            bar = stage_bars[stage][b]
            ax.add_patch(mpatches.Rectangle(
                (x, bar["y_bottom"]), BAR_W,
                bar["y_top"] - bar["y_bottom"],
                facecolor=STAGE_COLORS[stage],
                edgecolor="none",
                zorder=3,
            ))

    # --- Labels + stage headers ----------------------------------------------
    overlap_report: Dict[str, float] = {}
    for i, stage in enumerate(STAGES):
        x = x_positions[i]
        nodes = stage_nodes[stage]
        prev_mid = None
        for b in nodes:
            bar = stage_bars[stage][b]
            y_mid = (bar["y_top"] + bar["y_bottom"]) / 2
            label = BUCKET_LABELS[stage].get(b, b)
            if i == 0:
                ax.text(x - LABEL_OFF, y_mid, label, ha="right", va="center",
                        fontsize=LABEL_FS, color=LABEL_COLOR, zorder=4)
            else:
                ax.text(x + BAR_W + LABEL_OFF, y_mid, label, ha="left",
                        va="center", fontsize=LABEL_FS, color=LABEL_COLOR,
                        zorder=4)
            if prev_mid is not None:
                spacing = prev_mid - y_mid
                overlap_report[stage] = max(
                    overlap_report.get(stage, 0.0),
                    max(0.0, label_h - spacing) / label_h,
                )
            prev_mid = y_mid
        ax.text(x + BAR_W / 2, y_top + 0.065, STAGE_HEADERS[stage],
                ha="center", va="center", fontsize=HEADER_FS,
                fontweight="bold", color=HEADER_COLOR, zorder=4)

    # --- Legend (bottom centre, stage colours) --------------------------------
    handles = [
        mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="none",
                       label=STAGE_HEADERS[s])
        for s in STAGES
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=5, frameon=False, fontsize=LEGEND_FS,
        handlelength=1.3, handleheight=0.9, columnspacing=1.6,
    )

    # --- Export ----------------------------------------------------------------
    for ext in ("pdf", "svg"):
        fig.savefig(OUT_DIR / f"risk_propagation_sankey_buckets.{ext}")
    fig.savefig(OUT_DIR / "risk_propagation_sankey_buckets.png", dpi=450)
    plt.close(fig)

    # --- QA notes ----------------------------------------------------------------
    print("Node counts per stage:",
          {s: len(stage_nodes[s]) for s in STAGES})
    print("Excluded 'missing' events per stage:", excluded)
    print("Ribbon events referencing dropped nodes:", dropped_ribbon_count)
    print(f"Column x positions: {[round(x, 3) for x in x_positions]}")
    print(f"Rightmost label extent (x units): "
          f"{x_positions[4] + BAR_W + LABEL_OFF + label_w['AffectedActor'] / in2x:.3f}")
    print("Max vertical label overlap per stage (fraction of label height):",
          {s: round(v, 2) for s, v in overlap_report.items() if v > 0})
    print("Adaptive column gaps:", {s: round(g, 3) for s, g in stage_gaps.items()})
    print(f"Saved to {OUT_DIR}/risk_propagation_sankey_buckets.pdf/.svg/.png")


if __name__ == "__main__":
    main()
