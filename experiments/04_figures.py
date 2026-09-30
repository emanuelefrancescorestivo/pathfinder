"""The README's figures (docs/figures/*.svg), drawn from results/*.json.

The low-band figure recomputes one cross-validation repeat (seed 0, training set
only) because selection.json keeps summaries, not predictions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import ROOT, load  # noqa: E402

from pathfinder import evaluate as ev  # noqa: E402
from pathfinder.data import load_train  # noqa: E402
from pathfinder.models import GRADE_MODELS  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

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
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "svg.hashsalt": "pathfinder",  # deterministic SVG ids
        "axes.axisbelow": True,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# 1. Mean residual by true-grade band: where least squares over-predicts.
tr = load_train()
y = tr.grade.to_numpy()
edges = np.array([30, 50, 60, 70, 80, 90, 99.99, 100.01])
labels = ["30-50", "50-60", "60-70", "70-80", "80-90", "90-99", "100"]
band = np.digitize(y, edges[1:-1])
fig, ax = plt.subplots(figsize=(7, 3.6))
for name, colour, dx in (("OLS + clip", ORANGE, -0.09), ("Tobit", BLUE, 0.09)):
    p = ev.repeated_oof(GRADE_MODELS[name], tr.X, tr.grade, repeats=1, seed=0)[0]
    m = [np.mean(y[band == b] - p[band == b]) for b in range(len(labels))]
    xs = np.arange(len(labels)) + dx
    ax.plot(
        xs,
        m,
        color=colour,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label=name,
    )
    ax.annotate(
        name,
        (xs[0], m[0]),
        xytext=(8, -4 if name == "Tobit" else 6),
        textcoords="offset points",
        color=INK2,
        fontsize=9,
    )
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(np.arange(len(labels)), labels)
ax.set_xlabel("true final grade")
ax.set_ylabel("mean residual (true - predicted)")
ax.set_title(
    "Weak students are over-predicted less once the ceiling is modelled",
    loc="left",
    fontsize=11,
    color=INK,
)
ax.legend(frameon=False, loc="lower right")
save(fig, "residual_by_band")

# 2. Capacity: dropouts found among the k highest-risk test students.
final = load("final")["dropout"]
tab = final["capacity"]["table"]
ks = [r["k"] for r in tab]
fig, ax = plt.subplots(figsize=(7, 3.6))
for key, colour, name in (
    ("precision_at_k", BLUE, "precision@k"),
    ("recall_at_k", ORANGE, "recall@k"),
):
    vals = [r[key] for r in tab]
    ax.plot(
        ks,
        vals,
        color=colour,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label=name,
    )
    ax.annotate(
        name,
        (ks[-1], vals[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=INK2,
        fontsize=9,
        va="center",
    )
for r in tab:
    ax.annotate(
        f"{r['dropouts_found']}/{final['n_dropouts']}",
        (r["k"], r["recall_at_k"]),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        color=INK2,
        fontsize=8,
    )
ax.set_ylim(0, 1.05)
ax.set_xlabel(f"students contacted, highest risk first (test cohort of {final['n_test']})")
ax.set_ylabel("share")
ax.set_title("What an office that can see k students would get", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper center", ncols=2)
save(fig, "capacity")

# 3. The SMOTE defect: precision promised by the original validation vs delivered.
a = load("audit")
rows = [
    (
        "SMOTE before CV\n(original)",
        a["smote_before_cv"]["oof_precision"],
        a["smote_before_cv"]["test_precision"],
    ),
    (
        "SMOTE inside folds",
        a["smote_inside_folds"]["oof_precision"],
        a["smote_inside_folds"]["test_precision"],
    ),
]
fig, ax = plt.subplots(figsize=(7, 3.0))
for i, (_name, promised, delivered) in enumerate(rows):
    for dy, val, colour, lab in (
        (-0.18, promised, ORANGE, "validation estimate"),
        (0.18, delivered, BLUE, "test set"),
    ):
        ax.barh(i + dy, val, height=0.32, color=colour, label=lab if i == 0 else None)
        ax.annotate(
            f"{val:.2f}",
            (val, i + dy),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
            fontsize=9,
        )
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.set_xlabel("precision at the F2-optimal threshold")
ax.set_title(
    "Oversampling before splitting promised a precision the test set did not deliver",
    loc="left",
    fontsize=11,
    color=INK,
)
ax.legend(frameon=False, loc="lower right")
save(fig, "smote_leak")
