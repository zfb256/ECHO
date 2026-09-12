"""Protocol overview figure for the ECHO paper (Figure 1).

Run from anywhere:
    python scripts/make_fig_protocol.py   ->  paper/figures/fig_protocol.pdf

Design notes
------------
- Layout uses a 900x532 grid with the y axis inverted, so coordinates read
  top-down. One grid unit is 0.72pt on the canvas, so text sizes are set
  through the scale below rather than written as raw point sizes.
- Panel (a) is drawn as a branch rather than a pipeline: the three
  continuations share a source, and Clean A feeds both contrasts. That
  structure, not the ordering of stages, is what the identification rests on.
- Each colour means one thing: orange marks exposure to the false
  proposition, blue a clean condition, green a step where a human supplies the
  label. Judges are deliberately not green; they are not ground truth.
- pdf.fonttype 42 embeds TrueType outlines, which ACL camera-ready requires.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig_protocol"

FALSE, CLEAN, HUMAN = "#DD8452", "#4C72B0", "#55A868"
INK, MUTED, LINE = "#22252A", "#6B7280", "#9CA3AF"
PANEL, BORDER = "#FFFFFF", "#E3E6EA"

W, H = 900, 532
FS = 0.72                    # points per grid unit
HEAD, BODY, NOTE = 13.5, 11.0, 9.4      # in grid units; multiplied by FS
LINE_H = 13.5                # baseline step for stacked text


def build():
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def card(x, y, w, h, r=10, ec=BORDER, fc=PANEL, lw=1.0, z=3):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))

    def bar(x, y, w, h, colour, r=2.0, z=4):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor=colour, zorder=z))

    def txt(x, y, s, size, color=INK, ha="left", weight="normal", z=6):
        ax.text(x, y, s, fontsize=size * FS, color=color, ha=ha, va="center",
                zorder=z, weight=weight)

    def elbow(pts, color=LINE, lw=1.3, head=True, z=5, ms=8):
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            ax.add_patch(FancyArrowPatch(
                pts[i], pts[i + 1], arrowstyle="-|>" if (last and head) else "-",
                mutation_scale=ms, linewidth=lw, color=color,
                shrinkA=0, shrinkB=0, joinstyle="round", zorder=z))

    def panel(x, y, w, h, letter, title, accent):
        card(x, y, w, h, r=12, z=1)
        bar(x, y, w, 4, accent, z=2)
        ax.add_patch(Circle((x + 22, y + 24), 9.5, facecolor=accent,
                            linewidth=0, zorder=4))
        txt(x + 22, y + 24.5, letter, BODY, "white", "center", "bold")
        txt(x + 38, y + 24.5, title, HEAD, INK, weight="bold")

    def node(x, y, w, h, title, lines=(), accent=None, r=7):
        """Bold title over muted detail lines, vertically centred as a block."""
        fc = accent + "1A" if accent else "#FAFBFC"
        ec = accent if accent else BORDER
        card(x, y, w, h, r=r, ec=ec, fc=fc, lw=1.1, z=3)
        block = LINE_H * (1 + len(lines))
        top = y + (h - block) / 2 + LINE_H / 2
        txt(x + 11, top, title, BODY, INK, weight="bold")
        for i, s in enumerate(lines, start=1):
            txt(x + 11, top + i * LINE_H, s, NOTE, MUTED)

    # ---------------------------------------------------------------- panel a
    # Arrow colour marks which continuation travels along it; grey marks a plain
    # sequence step that carries no particular condition.
    panel(8, 8, 884, 180, "a", "Injected arm: exposure and headline inference", CLEAN)

    r1, r2, r3 = 68, 110, 152          # contaminated / Clean A / Clean B centres
    node(26, 80, 186, 60, "Same dialogue",
         ("and shared probe", "tested model generates"))

    node(236, r1 - 17, 152, 34, "Contaminated", ("history + false C",), FALSE)
    node(236, r2 - 17, 152, 34, "Clean A", ("history, no C",), CLEAN)
    node(236, r3 - 17, 152, 34, "Clean B", ("history, no C",), CLEAN)
    # Group the saved generations; take the audit branch from its bottom edge.
    card(230, 45, 164, 130, r=9, ec=LINE, fc="none", lw=0.8, z=2)

    txt(448, 58, "Frozen rule scores", NOTE, MUTED, weight="bold")
    node(448, 72, 170, 34, "CCR", ("directed discordance",))
    node(448, 114, 170, 34, "Placebo", ("clean–clean discordance",))
    node(656, 80, 220, 60, "Primary contrast",
         ("CCR $-$ placebo", "clustered intervals"))

    for y in (r1, r2, r3):
        elbow([(212, r2), (224, r2), (224, y), (236, y)], LINE, 1.2)
    elbow([(388, r1), (418, r1), (418, 82), (448, 82)], FALSE, 1.4)
    elbow([(388, r2), (432, r2), (432, 96), (448, 96)], CLEAN, 1.4)
    elbow([(388, r2), (432, r2), (432, 124), (448, 124)], CLEAN, 1.4)
    elbow([(388, r3), (418, r3), (418, 138), (448, 138)], CLEAN, 1.4)
    elbow([(618, 89), (637, 89), (637, 101), (656, 101)], LINE, 1.3)
    elbow([(618, 131), (637, 131), (637, 119), (656, 119)], LINE, 1.3)

    # ---------------------------------------------------------------- panel b
    panel(8, 260, 884, 132, "b", "Measurement audits", HUMAN)

    node(26, 310, 258, 64, "Held-out human labels",
         ("detector precision and recall", "judge calibration"), HUMAN, r=8)
    node(352, 310, 258, 64, "Hunyuan and Kimi",
         ("saved pairs + C and correction", "contaminated / Clean A",
          "Hunyuan also Clean A / B"), r=8)
    node(666, 310, 210, 64, "Judge assessment",
         ("semantic CCR and intervals", "Hunyuan-only placebo gap",
          "cross-judge agreement"), r=8)

    elbow([(610, 342), (666, 342)], LINE, 1.3)
    # Saved generations feed both audits, not the independent self-induced arm.
    node(212, 206, 200, 34, "Saved outputs from (a)")
    elbow([(312, 175), (312, 206)], LINE, 1.2)
    elbow([(312, 240), (312, 300)], LINE, 1.2, head=False)
    elbow([(312, 300), (155, 300), (155, 310)], LINE, 1.2)
    elbow([(312, 300), (481, 300), (481, 310)], LINE, 1.2)
    # Human labels assess predictions externally; they never enter a judge prompt.
    J = [(156, 374), (156, 387), (650, 387), (650, 360), (666, 360)]
    for i in range(len(J) - 1):
        ax.add_patch(FancyArrowPatch(
            J[i], J[i + 1], arrowstyle="-|>" if i == len(J) - 2 else "-",
            mutation_scale=8, linewidth=1.3, color=HUMAN, linestyle=(0, (4, 3)),
            shrinkA=0, shrinkB=0, joinstyle="round", zorder=5))

    # ---------------------------------------------------------------- panel c
    # A separate arm, not a downstream stage: nothing flows into it from (b).
    panel(8, 404, 884, 120, "c", "Self-induced arm: conditional persistence", FALSE)

    node(26, 446, 198, 62, "Stage 1", ("factual question", "greedy model answer"))
    node(252, 446, 198, 62, "Stage 2", ("replay stage-1 answer", "fixed follow-up"))
    node(478, 446, 198, 62, "Human annotation",
         ("stage-1 factual error", "stage-2 inheritance"), HUMAN)
    node(704, 446, 188, 62, "HCR$_{\\rm si}$",
         ("inheritance among", "confirmed errors"))

    for x0, x1 in ((224, 252), (450, 478), (676, 704)):
        elbow([(x0, 477), (x1, 477)], LINE, 1.3)

    return fig


def main():
    fig = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
