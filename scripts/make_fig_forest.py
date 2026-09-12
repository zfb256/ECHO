"""Plot Figure 3 from reports_zh/per_model_gap_ci.json.

Rebuild data first with: python scripts/compute_per_model_gap_ci.py
"""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports_zh" / "per_model_gap_ci.json"
OUT = ROOT / "paper" / "figures" / "fig_forest"

FALSE, CLEAN = "#DD8452", "#4C72B0"
INK, MUTED, LINE, BAND = "#22252A", "#6B7280", "#C7CBD1", "#4C72B0"

OPEN = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
        "InternLM2.5-7B", "GLM-4-9B", "Yi-1.5-9B"]
HOSTED = ["DeepSeek-V4-Flash", "Doubao-Seed-2.1-Pro"]


def build():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    m, agg = d["models"], d["open_weight_aggregate"]

    rows = ([(n, m[n], CLEAN) for n in OPEN]
            + [("Open-weight aggregate", agg, INK)]
            + [(n, m[n], CLEAN) for n in HOSTED])

    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    ys = list(range(len(rows)))[::-1]

    # pooled open-weight interval, so each model reads against the aggregate
    ax.axvspan(agg["lo"], agg["hi"], color=BAND, alpha=0.09, linewidth=0)
    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)

    for y, (name, v, colour) in zip(ys, rows):
        pooled = name.startswith("Open-weight")
        lw = 2.0 if pooled else 1.3
        ax.plot([v["lo"], v["hi"]], [y, y], color=colour, linewidth=lw,
                solid_capstyle="butt", zorder=3)
        for x in (v["lo"], v["hi"]):          # interval end caps
            ax.plot([x, x], [y - 0.17, y + 0.17], color=colour, linewidth=lw,
                    zorder=3)
        ax.plot([v["gap"]], [y], marker="D" if pooled else "o",
                markersize=5.0 if pooled else 4.2, color=colour,
                markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=7.2)
    for lab, (name, _, _) in zip(ax.get_yticklabels(), rows):
        if name.startswith("Open-weight"):
            lab.set_fontweight("bold")
    ax.set_ylim(-0.8, len(rows) - 0.2)

    ax.set_xlim(-0.03, 0.60)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xlabel("injected $-$ placebo gap (detected)", fontsize=7.6,
                  color=INK, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.0, colors=MUTED, length=2.5)
    ax.tick_params(axis="y", length=0, colors=INK)

    ax.grid(axis="x", color=LINE, linewidth=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.spines["bottom"].set_linewidth(0.7)

    # the two hosted rows are a different population; mark the break
    ax.axhline(1.5, color=LINE, linewidth=0.7, zorder=1)
    ax.text(0.585, 0.5, "hosted", fontsize=6.6, color=MUTED, ha="right",
            va="center", style="italic")

    fig.tight_layout(pad=0.25)
    return fig


def main():
    fig = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("wrote %s" % OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
