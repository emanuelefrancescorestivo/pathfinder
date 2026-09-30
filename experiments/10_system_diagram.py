"""What an advising office would run each week, as one diagram: docs/figures/system.svg.

A diagram, not a result: it shows no numbers. The names of the steps are the modules
that implement them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _common import ROOT  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE = "#2a78d6"
plt.rcParams.update({"svg.hashsalt": "pathfinder", "font.size": 10, "text.color": INK})

STEPS = (
    ("This week's data", "attendance, assignments,\nquizzes, absences, …", "data.py, oulad.py"),
    (
        "Risk of leaving",
        "logistic model, calibrated;\ngrade: censored (Tobit)",
        "models.py, censored.py",
    ),
    ("Who to see first", "top k the office can see,\nor a cost threshold", "decide.py"),
    ("Which office calls", "the service that owns\nthe first reason", "segments.py"),
    (
        "Student card",
        "reasons, and what-if plans\nthat move ≤ 2 levers",
        "explain.py,\ncounterfactual.py",
    ),
)

fig, ax = plt.subplots(figsize=(13, 3.3))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 13)
ax.set_ylim(0.35, 3.5)
ax.axis("off")
w, h, gap, y0 = 2.3, 1.45, 0.3, 1.35
x0 = (13 - 5 * w - 4 * gap) / 2
for i, (title, body, modules) in enumerate(STEPS):
    x = x0 + i * (w + gap)
    colour = BLUE if i == 2 else MUTED
    ax.add_patch(
        FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.1",
            fc=SURFACE,
            ec=colour,
            lw=2 if i == 2 else 1.2,
        )
    )
    ax.text(x + 0.14, y0 + h - 0.27, title, fontsize=11, fontweight="bold", va="center")
    ax.text(x + 0.14, y0 + h - 0.72, body, fontsize=9, color=INK2, va="center", linespacing=1.4)
    ax.text(x + 0.14, y0 + 0.1, modules, fontsize=7.5, color=MUTED, va="bottom", family="monospace")
    if i:
        ax.add_patch(
            FancyArrowPatch(
                (x - gap + 0.03, y0 + h / 2),
                (x - 0.03, y0 + h / 2),
                arrowstyle="-|>",
                mutation_scale=11,
                color=INK2,
                lw=1.3,
            )
        )
ax.text(
    x0,
    3.3,
    "What an advising office would run each week",
    fontsize=13,
    fontweight="bold",
    va="center",
)
ax.text(
    x0,
    0.72,
    "Checked on two datasets: the course's synthetic cohort (the office views, the grade model) "
    "and real Open University registrations, OULAD\n(how early a warning can come, "
    "early_warning.py and survival.py). Every step is tested; every number in the README "
    "is generated from results/*.json.",
    fontsize=9,
    color=INK2,
    va="center",
    linespacing=1.5,
)
out = ROOT / "docs" / "figures" / "system.svg"
fig.savefig(out, bbox_inches="tight", metadata={"Date": None})
plt.close(fig)
print(f"wrote {out.relative_to(ROOT)}")
