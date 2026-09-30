"""Who will leave in the next four weeks? Three scorers compared week by week on OULAD.

Needs the OULAD CSVs in data/raw/oulad/. Trains on 2013B, 2013J and 2014B, scores the
2014J presentation week by week, and writes results/time.json plus two figures:

  docs/figures/time_auc.svg   AUC for "withdraws within 4 weeks", by week and scorer
  docs/figures/lead_time.svg  how many weeks before withdrawing students were first flagged

Nothing here is tuned on 2014J: the scorers, the horizon (4 weeks), the capacity (10% of
the week's cohort) and the "2 weeks ahead" criterion are fixed in this file before it runs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import ROOT, save  # noqa: E402

from pathfinder import oulad  # noqa: E402
from pathfinder.early_warning import (  # noqa: E402
    TEST,
    TRAIN,
    horizon_table,
    lead_summary,
    lead_times,
    weekly_scores,
)

WEEKS = list(range(0, 31))
HORIZON = 4
CAPACITY = 0.10

d = oulad.load(ROOT / "data" / "raw" / "oulad")
scores, labels = weekly_scores(d, WEEKS, horizon=HORIZON)
table = horizon_table(scores, labels, capacity_share=CAPACITY)
leads = {name: lead_times(d, by_week, capacity_share=CAPACITY) for name, by_week in scores.items()}
summary = {name: lead_summary(t) for name, t in leads.items()}
print(table.round(3).to_string(index=False))
for name, s in summary.items():
    print(name, s)

save(
    "time",
    {
        "train_presentations": list(TRAIN),
        "test_presentations": list(TEST),
        "weeks": WEEKS,
        "horizon_weeks": HORIZON,
        "capacity_share": CAPACITY,
        "table": table.to_dict(orient="records"),
        "lead_summary": summary,
        "lead_histogram": {
            name: t["lead_weeks"].fillna(-1).astype(int).value_counts().sort_index().to_dict()
            for name, t in leads.items()
        },
    },
)

# -- figures -----------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOUR = {"landmark": "#8a8984", "landmark+trajectory": "#eb6834", "survival": "#2a78d6"}
LABEL = {
    "landmark": "weekly snapshot",
    "landmark+trajectory": "snapshot + trajectory",
    "survival": "survival (one hazard model)",
}
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.axisbelow": True,
        "svg.hashsalt": "pathfinder",
    }
)
out = ROOT / "docs" / "figures"

fig, ax = plt.subplots(figsize=(10, 3.8))
for name in scores:
    t = table[table["scorer"] == name]
    ax.fill_between(
        t["week"], t["auc_low"], t["auc_high"], color=COLOUR[name], alpha=0.12, linewidth=0
    )
    ax.plot(
        t["week"],
        t["auc"],
        color=COLOUR[name],
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=LABEL[name],
    )
ax.set_xlabel("week of the course (test presentation)")
ax.set_ylabel(f"AUC, withdraws within {HORIZON} weeks")
ax.legend(frameon=False, loc="lower left")
ax.set_title(
    f"Who leaves in the next {HORIZON} weeks? Trained on "
    f"{', '.join(TRAIN)}, tested on {', '.join(TEST)}",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
)
fig.savefig(out / "time_auc.svg", bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.4))
names = list(scores)
width = 0.8 / len(names)
top = 12
for j, name in enumerate(names):
    lw = leads[name]["lead_weeks"]
    bins = np.arange(-1, top + 2)  # never, 0 .. top, and top+1 meaning "more than top"
    counts = [
        int(lw.isna().sum())
        if b == -1
        else int((lw > top).sum())
        if b == top + 1
        else int((lw == b).sum())
        for b in bins
    ]
    ax.bar(
        bins + (j - (len(names) - 1) / 2) * width,
        counts,
        width=width,
        color=COLOUR[name],
        label=LABEL[name],
    )
ax.set_xticks(
    np.arange(-1, top + 2), ["never"] + [str(i) for i in range(top + 1)] + [f"{top + 1}+"]
)
ax.set_xlabel(f"weeks between first being flagged (top {CAPACITY:.0%}) and withdrawing")
ax.set_ylabel("students who withdrew")
ax.legend(frameon=False)
ax.set_title(
    "How much warning would the office have had?",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
)
fig.savefig(out / "lead_time.svg", bbox_inches="tight", metadata={"Date": None})
plt.close(fig)
print("wrote docs/figures/time_auc.svg, docs/figures/lead_time.svg")
