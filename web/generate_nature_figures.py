"""Generate publication-quality figures for Section 7 using nature-figure conventions.

Produces:
  - temporal_trend: yearly event counts (bar) + top-domain share (line), 2016--2025
  - risk_propagation_sankey: 5-stage Sankey (RiskSource -> Risk -> Consequence -> Impact -> AffectedActor)
    with ribbons coloured by risk domain

Backend: Python (matplotlib).  Exclusive — no cross-rendering.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# --------------------------------------------------------------------------- #
# nature-figure mandatory font + SVG rules (always first, no exceptions)
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",      # editable text in SVG
    "pdf.fonttype": 42,          # editable TrueType text in PDF
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).parent / "data" / "app_stats.json"
CLASSIFICATION_PATH = Path(__file__).parent / "data" / "risk_classification.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Domain order + NMI-pastel-inspired palette (low saturation, restrained)
# --------------------------------------------------------------------------- #
DOMAIN_ORDER = [
    "Malicious actors & misuse",
    "AI system safety, failures & limitations",
    "Discrimination & toxicity",
    "Misinformation",
    "Privacy & security",
    "Socioeconomic & environmental harms",
    "Human-computer interaction",
]

DOMAIN_SHORT = {
    "Malicious actors & misuse": "Malicious & misuse",
    "AI system safety, failures & limitations": "AI system safety",
    "Discrimination & toxicity": "Discrimination",
    "Misinformation": "Misinformation",
    "Privacy & security": "Privacy & security",
    "Socioeconomic & environmental harms": "Socioeconomic",
    "Human-computer interaction": "Human-computer",
}

# NMI-pastel family adapted for 7 categorical domains
DOMAIN_COLORS = {
    "Malicious actors & misuse":                            "#C44E52",  # muted red
    "AI system safety, failures & limitations":             "#DD8452",  # muted orange
    "Discrimination & toxicity":                            "#55A868",  # muted green
    "Misinformation":                                       "#8172B3",  # muted purple
    "Privacy & security":                                   "#4C72B0",  # muted blue
    "Socioeconomic & environmental harms":                  "#937860",  # muted brown
    "Human-computer interaction":                           "#DA8BC3",  # muted pink
}

SLOT_TYPES = ["RiskSource", "Risk", "Consequence", "Impact", "AffectedActor"]
SLOT_LABELS = ["Risk\nsource", "Risk", "Consequence", "Impact", "Affected\nactor"]

# Colour per rank position (top1 -> top5) for the Sankey
RANK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_stats() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_classifications() -> Dict[str, str]:
    """event_id -> domain."""
    mapping = {}
    if CLASSIFICATION_PATH.exists():
        with open(CLASSIFICATION_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                mapping[entry["event_id"]] = entry["domain"]
    return mapping


def extract_chains() -> List[dict]:
    """Read every event_subgraph_fused.json and extract the 5-slot chain.

    Returns a list of dicts:
        {event_id, domain, slots: {RiskSource: str|None, Risk: str|None, ...}}
    """
    classifications = load_classifications()
    chains: List[dict] = []

    for json_path in sorted(OUTPUT_DIR.glob("pred_*/event_subgraph_fused.json")):
        event_id = json_path.parent.name
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        domain = classifications.get(event_id, "Other")

        # Collect nodes by entity_type
        nodes_by_type: Dict[str, List[str]] = {t: [] for t in SLOT_TYPES}
        for node in data.get("nodes", []):
            etype = node.get("entity_type", "")
            if etype in nodes_by_type:
                name = node.get("name", "").strip()
                if name:
                    nodes_by_type[etype].append(name)

        # For each slot, pick the first node (most events have exactly one per slot)
        slots = {}
        for slot_type in SLOT_TYPES:
            candidates = nodes_by_type.get(slot_type, [])
            slots[slot_type] = candidates[0] if candidates else None

        chains.append({"event_id": event_id, "domain": domain, "slots": slots})

    return chains


def truncate(text: str, max_len: int = 42) -> str:
    """Truncate long node names for Sankey labels."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


# --------------------------------------------------------------------------- #
# Figure 1 — Temporal trend (total event count, 2016--2024)
# --------------------------------------------------------------------------- #
# Figure contract (nature-figure):
#   1. Core conclusion: AI risk incident clusters grew sharply from 2016 to
#      2024, with a pronounced acceleration after 2022 driven by generative
#      AI deployment; 2023–2024 alone account for ~48% of all events.
#   2. Evidence chain: single hero panel — yearly event-cluster counts.
#   3. Archetype: quantitative grid (single trend panel).
#   4. Backend: Python (matplotlib), exclusive.
#   5. Journal/export: SVG (primary, editable text) + PDF + PNG @ 600 dpi;
#      single-column width (~89 mm); 5 pt glyph floor.

def fig_temporal_trend(stats: dict) -> None:
    """Line chart of total AI risk incident clusters per year, 2016--2024."""

    events_by_year = stats["time_distribution"]["events_by_year"]

    years = [y for y in sorted(events_by_year, key=int) if 2016 <= int(y) <= 2024]
    counts = [events_by_year[y] for y in years]
    x = np.arange(len(years), dtype=float)

    # --- Figure & axes (single-column Nature width ≈ 89 mm) ---
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.set_facecolor("white")

    # --- Single trend line (hero series) with restrained area fill ---
    line_color = "#0F4D92"  # PALETTE blue_main — signal family
    ax.plot(x, counts, color=line_color, linewidth=1.6, marker="o",
            markersize=3.8, markeredgecolor="white", markeredgewidth=0.6,
            zorder=5)
    ax.fill_between(x, counts, alpha=0.10, color=line_color, zorder=2)

    # --- Value labels at every data point ---
    y_max = max(counts)
    for xi, cnt in zip(x, counts):
        ax.text(xi, cnt + y_max * 0.035, f"{cnt}",
                fontsize=6, color=line_color, fontweight="bold",
                ha="center", va="bottom", zorder=6)

    # --- Axes configuration ---
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=6.5, rotation=0)
    ax.set_xlim(-0.55, len(years) - 0.45)
    ax.set_ylim(0, y_max * 1.22)

    ax.set_xlabel("Year", fontsize=7.5, labelpad=4)
    ax.set_ylabel("Number of event clusters", fontsize=7.5, labelpad=4)

    # Clean spines: keep only left + bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Light horizontal grid
    ax.grid(axis="y", color="#ECECEC", linewidth=0.5, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(colors="#4D4D4D", labelsize=6, length=2.5, width=0.6)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.20)

    out_svg = FIG_DIR / "temporal_trend.svg"
    fig.savefig(out_svg, format="svg")
    print(f"Saved: {out_svg}")
    out_pdf = FIG_DIR / "temporal_trend.pdf"
    fig.savefig(out_pdf, format="pdf")
    print(f"Saved: {out_pdf}")
    out_png = FIG_DIR / "temporal_trend.png"
    fig.savefig(out_png, format="png", dpi=600)
    print(f"Saved: {out_png}")

    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — Risk domain pie chart (7 domains, full dataset)
# --------------------------------------------------------------------------- #
# Figure contract (nature-figure):
#   1. Core conclusion: the seven risk domains are highly uneven; two domains
#      — Malicious actors & misuse (24.5%) and AI system safety (20.9%) —
#      together account for ~45% of all events, while HCI (4.1%) and
#      Socioeconomic (6.1%) are comparatively rare.
#   2. Evidence chain: single donut panel over the full 2034-event dataset.
#   3. Archetype: quantitative grid (single donut panel).
#   4. Backend: Python (matplotlib), exclusive.
#   5. Journal/export: SVG (primary, editable text) + PDF + PNG @ 600 dpi;
#      ~1.5-column width to fit the right-side legend; 5 pt glyph floor.

def fig_domain_pie(stats: dict) -> None:
    """Donut chart of the seven risk domains over the full dataset."""

    domain_table = stats["risk_domain_distribution"]["table"]
    # Order by count descending
    items = sorted(
        [(d["domain"], d["count"]) for d in domain_table if d["count"] > 0],
        key=lambda x: -x[1],
    )
    domains = [it[0] for it in items]
    counts = [it[1] for it in items]
    total = sum(counts)
    colors = [DOMAIN_COLORS.get(d, "#888888") for d in domains]
    labels = [DOMAIN_SHORT.get(d, d) for d in domains]

    # --- Figure (~1.5-column width for legend room) ---
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.set_facecolor("white")

    # Donut wedges (ring width ~ 0.38)
    wedges, _ = ax.pie(
        counts, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.0),
    )

    # --- Percentage labels inside the ring ---
    for w, cnt in zip(wedges, counts):
        ang = (w.theta2 + w.theta1) / 2.0
        r = 0.81  # inside the ring
        x = r * np.cos(np.deg2rad(ang))
        y = r * np.sin(np.deg2rad(ang))
        pct = cnt / total * 100
        if pct >= 4.0:
            ax.text(x, y, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white", zorder=5)

    # --- Centre total count ---
    ax.text(0, 0.08, f"{total:,}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#272727")
    ax.text(0, -0.10, "event clusters", ha="center", va="center",
            fontsize=6.5, color="#707070")

    # --- Legend on the right ---
    legend_labels = [f"{lab}  ({cnt}, {cnt/total*100:.1f}%)"
                     for lab, cnt in zip(labels, counts)]
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.5,
              handlelength=1.0, handleheight=1.0, labelspacing=0.6)

    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.02, right=0.62, top=0.96, bottom=0.04)

    out_svg = FIG_DIR / "domain_pie.svg"
    fig.savefig(out_svg, format="svg")
    print(f"Saved: {out_svg}")
    out_pdf = FIG_DIR / "domain_pie.pdf"
    fig.savefig(out_pdf, format="pdf")
    print(f"Saved: {out_pdf}")
    out_png = FIG_DIR / "domain_pie.png"
    fig.savefig(out_png, format="png", dpi=600)
    print(f"Saved: {out_png}")

    plt.close(fig)



# --------------------------------------------------------------------------- #
# Figure 2 — Risk propagation Sankey (full dataset, top-5 per stage)
# --------------------------------------------------------------------------- #

def _ribbon_path(x0, y0_top, y0_bot, x1, y1_top, y1_bot):
    """Return a Path for a Sankey ribbon with smooth cubic-bezier sides."""
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


def fig_sankey(chains: List[dict], stats: dict) -> None:
    """Single-panel 5-stage Sankey over the full dataset.

    Stages: RiskSource -> Risk -> Consequence -> Impact -> AffectedActor.
    Each stage shows only the top-5 most frequent slot values (no "Other"
    bucket). Ribbons carry the co-occurrence count between adjacent-stage
    top-5 values. Ribbon colour is inherited from the left-side value's rank.
    """
    TOP_N = 5

    # --- Aggregate over all chains ---
    stage_values: Dict[str, Counter] = {s: Counter() for s in SLOT_TYPES}
    transitions: Dict[Tuple[str, str], Counter] = {
        (SLOT_TYPES[i], SLOT_TYPES[i + 1]): Counter()
        for i in range(len(SLOT_TYPES) - 1)
    }
    for chain in chains:
        slots = chain["slots"]
        prev_val = None
        for i, slot in enumerate(SLOT_TYPES):
            val = slots.get(slot)
            if val:
                val = truncate(val, 22)
                stage_values[slot][val] += 1
                if prev_val is not None:
                    transitions[(SLOT_TYPES[i - 1], slot)][(prev_val, val)] += 1
                prev_val = val
            else:
                prev_val = None

    # --- Top-N per stage ---
    top_values: Dict[str, List[str]] = {
        s: [v for v, _ in stage_values[s].most_common(TOP_N)] for s in SLOT_TYPES
    }
    top_sets: Dict[str, set] = {s: set(vs) for s, vs in top_values.items()}

    # --- Filter transitions to top-N <-> top-N only ---
    filtered_trans: Dict[Tuple[str, str], Counter] = {}
    for (left_slot, right_slot), trans in transitions.items():
        filtered = Counter()
        for (lv, rv), cnt in trans.items():
            if lv in top_sets[left_slot] and rv in top_sets[right_slot]:
                filtered[(lv, rv)] = cnt
        filtered_trans[(left_slot, right_slot)] = filtered

    # --- Layout ---
    n_stages = len(SLOT_TYPES)
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 5 stage columns; first pushed right to leave label space on the left,
    # last pushed left to leave label space on the right.
    x_positions = [0.20, 0.36, 0.52, 0.68, 0.80]
    bar_width = 0.014
    gap_frac = 0.012
    total_bar_height = 0.80
    y_top_start = 0.88

    # --- Compute bar geometry per stage ---
    stage_bars: List[Dict[str, dict]] = []
    for stage_idx, slot in enumerate(SLOT_TYPES):
        vals = top_values[slot]
        counts = [stage_values[slot][v] for v in vals]
        total = sum(counts)
        y_cursor = y_top_start
        bars = {}
        for rank, (val, cnt) in enumerate(zip(vals, counts)):
            height = (cnt / total) * total_bar_height if total > 0 else 0
            bars[val] = {
                "y_top": y_cursor,
                "y_bottom": y_cursor - height,
                "height": height,
                "count": cnt,
                "rank": rank,
                "color": RANK_COLORS[rank % len(RANK_COLORS)],
            }
            y_cursor -= height + gap_frac
        stage_bars.append(bars)

    # --- Draw ribbons between adjacent stages ---
    for stage_idx in range(n_stages - 1):
        left_slot = SLOT_TYPES[stage_idx]
        right_slot = SLOT_TYPES[stage_idx + 1]
        x_left = x_positions[stage_idx] + bar_width
        x_right = x_positions[stage_idx + 1]
        left_bars = stage_bars[stage_idx]
        right_bars = stage_bars[stage_idx + 1]

        left_consumed: Dict[str, float] = {v: b["y_top"] for v, b in left_bars.items()}
        right_consumed: Dict[str, float] = {v: b["y_top"] for v, b in right_bars.items()}

        trans = filtered_trans[(left_slot, right_slot)]
        for lval in top_values[left_slot]:
            for rval in top_values[right_slot]:
                flow = trans.get((lval, rval), 0)
                if flow == 0:
                    continue
                l_top = left_consumed[lval]
                l_bot = l_top - (flow / left_bars[lval]["count"]) * left_bars[lval]["height"]
                r_top = right_consumed[rval]
                r_bot = r_top - (flow / right_bars[rval]["count"]) * right_bars[rval]["height"]
                left_consumed[lval] = l_bot
                right_consumed[rval] = r_bot

                path = _ribbon_path(x_left, l_top, l_bot, x_right, r_top, r_bot)
                patch = mpatches.PathPatch(path, facecolor=left_bars[lval]["color"],
                                           alpha=0.32, edgecolor="none", zorder=2)
                ax.add_patch(patch)

    # --- Draw bars + value labels ---
    for stage_idx, slot in enumerate(SLOT_TYPES):
        x = x_positions[stage_idx]
        for val, bar in stage_bars[stage_idx].items():
            rect = mpatches.Rectangle(
                (x, bar["y_bottom"]), bar_width, bar["height"],
                facecolor=bar["color"], edgecolor="white", linewidth=0.4, zorder=4,
            )
            ax.add_patch(rect)

            y_mid = (bar["y_top"] + bar["y_bottom"]) / 2
            # First stage: label to the left; last stage: label to the right;
            # middle stages: label to the right of the bar (overlays ribbons
            # but rendered on top with zorder).
            if stage_idx == 0:
                ax.text(x - 0.006, y_mid, val, ha="right", va="center",
                        fontsize=6.5, color="#272727", zorder=10, clip_on=False)
            elif stage_idx == n_stages - 1:
                ax.text(x + bar_width + 0.006, y_mid, val, ha="left", va="center",
                        fontsize=6.5, color="#272727", zorder=10, clip_on=False)
            else:
                ax.text(x + bar_width + 0.005, y_mid, val, ha="left", va="center",
                        fontsize=5.8, color="#272727", zorder=10, clip_on=False)

        # Stage header
        ax.text(x + bar_width / 2, y_top_start + 0.03, SLOT_LABELS[stage_idx],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#4D4D4D")

    # --- Rank colour legend ---
    legend_handles = [
        mpatches.Patch(facecolor=RANK_COLORS[r], edgecolor="white", linewidth=0.4,
                       label=f"Top {r + 1}")
        for r in range(TOP_N)
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=TOP_N,
              fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Risk propagation path across the full dataset (top-5 per stage)",
                 fontsize=11, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)

    for ext in ("svg", "pdf", "png"):
        out = FIG_DIR / f"risk_propagation_sankey.{ext}"
        fig.savefig(out, format=ext, dpi=600 if ext == "png" else None)
        print(f"Saved: {out}")

    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading stats...")
    stats = load_stats()

    print("Generating temporal trend figure...")
    fig_temporal_trend(stats)

    print("Generating domain pie figure...")
    fig_domain_pie(stats)

    print("Extracting risk chains from event subgraphs...")
    chains = extract_chains()
    print(f"  Extracted {len(chains)} event chains")

    print("Generating risk propagation Sankey figure...")
    fig_sankey(chains, stats)

    print("All figures generated.")


if __name__ == "__main__":
    main()
