"""Figures for readers who run an advising office, and for the questions behind the model.

All drawn from results/*.json. Produces docs/figures/:
  questions.svg        the questions that shaped the project, the evidence, the decision
  feature_questions.svg  feature-engineering questions answered by cross-validation
  model_choice.svg     candidate models, cross-validated
  caseload.svg         one cohort ranked by risk, coloured by the office it routes to
  routing.svg          flagged students per office
  cluster_check.svg    does the flagged population have cluster structure?
  student_card.svg     one student: reasons and what-if plans
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import ROOT, load  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#b9b8b3"
SERVICE_COLOUR = {
    "academic": BLUE,
    "engagement": ORANGE,
    "financial": AQUA,
    "access": YELLOW,
    "none": GRAY,
}
SERVICE_NAME = {
    "academic": "Academic support",
    "engagement": "Engagement follow-up",
    "financial": "Financial aid",
    "access": "Digital & travel access",
    "none": "No dominant driver",
}
PRETTY = {
    "attendance_rate_pct": "attendance",
    "assignment_completion_pct": "assignments completed",
    "quiz_average_pct": "quiz average",
    "hours_self_study_week": "self-study hours",
    "prior_gpa_20": "prior GPA",
    "absences_last30d": "absences (30 days)",
    "tutoring_sessions_month": "tutoring sessions",
    "internet_reliability_score": "internet",
    "commute_minutes": "commute",
    "financial_stress_score": "financial stress",
    "participation_score": "participation",
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
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.axisbelow": True,
        "svg.hashsalt": "pathfinder",
    }
)


def pretty(f: str) -> str:
    if f.startswith("dormitory_block_"):
        return f"dormitory block {f.rsplit('_', 1)[1]}"
    return PRETTY.get(f, f)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def title(ax: plt.Axes, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=12, color=INK, pad=22 if sub else 8, fontweight="bold")
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


syn, sel, fin, aud = load("synthetic"), load("selection"), load("final"), load("audit")
_time_path = ROOT / "results" / "time.json"
if _time_path.exists():
    _t = load("time")
    _s = _t["lead_summary"]["landmark"]
    _aucs = [r["auc"] for r in _t["table"]]
    time_step = (
        "How early can we warn?",
        f"OULAD, week by week: AUC {min(_aucs):.2f}-{max(_aucs):.2f} for leaving within "
        f"{_t['horizon_weeks']} weeks; {_s['flagged_2_weeks_ahead']['value']:.0%} of "
        "withdrawals flagged 2+ weeks ahead. Trends and survival add nothing clear.",
        "Keep the plain weekly model; state how little warning there is.",
        "kept",
    )
else:
    time_step = (
        "How early can we warn?",
        "The course data has no time axis.",
        "Weekly snapshots on OULAD (pending the data).",
        "open",
    )
fq, adm = load("feature_questions"), load("admin_views")

# -- 1. The questions that shaped the project ---------------------------------------
g = sel["grade"]["models"]
b, i_ = aud["smote_before_cv"], aud["smote_inside_folds"]
kc = adm["kmeans_check"]["train"]
best_sil = max(r["silhouette"] for r in kc)
dq = {q["id"]: q for q in fq["questions"]}
n_hurt = sum(q["grade_verdict"] == "hurts" for q in fq["questions"])
steps = [
    (
        "Is this real student data?",
        f"{syn['n_features_consistent_with_uniform_p_gt_0_5']} features are indistinguishable "
        f"from uniform draws; dropouts have final grades.",
        "Treat as synthetic. Say so first.",
        "changed",
    ),
    (
        "Why do predictions exceed 100?",
        f"{syn['share_grade_at_100']:.0%} of grades sit exactly at 100: the target is censored.",
        "Model the ceiling (Tobit) instead of clipping.",
        "changed",
    ),
    (
        "Why are weak students over-predicted?",
        f"Below 65, mean over-prediction {-g['OLS + clip']['mean_residual_low_band']:.1f} pts "
        f"with clipping, {-g['Tobit']['mean_residual_low_band']:.1f} with Tobit.",
        "Mostly the censoring, not the features.",
        "changed",
    ),
    (
        "Do trees beat a linear model?",
        f"CV RMSE {g['Gradient boosting']['rmse']['value']:.2f} (boosting) vs "
        f"{g['Tobit']['rmse']['value']:.2f} (Tobit).",
        "Keep the linear family: the generator is near-linear.",
        "kept",
    ),
    (
        "Do the notebook's engineered features help?",
        f"log(absences), a tutoring flag, the 6-feature set, interactions: {n_hurt} of "
        f"{len(fq['questions'])} hurt in every CV repeat.",
        "Keep the raw features.",
        "rejected",
    ),
    (
        "Why did validation promise 0.81 precision?",
        f"SMOTE before CV: {b['oof_precision']:.2f} promised, {b['test_precision']:.2f} "
        f"delivered. Inside folds: {i_['oof_precision']:.2f} promised.",
        "No resampling; honest probabilities.",
        "changed",
    ),
    (
        "Which threshold?",
        "F2 weights recall 4x precision; nobody chose that.",
        "Top-k for a capacity, Bayes threshold for a cost.",
        "changed",
    ),
    (
        "Are there dropout archetypes?",
        f"K-means silhouette at most {best_sil:.2f} for k = 2..6, and unstable.",
        "Route by each student's first reason.",
        "changed",
    ),
    time_step,
]
CHIP = {
    "changed": (BLUE, "changed"),
    "kept": (AQUA, "kept"),
    "rejected": (ORANGE, "rejected"),
    "open": (GRAY, "open"),
}
H = 0.9
fig, ax = plt.subplots(figsize=(12, H * len(steps) + 0.9))
ax.set_xlim(0, 12)
ax.set_ylim(0, len(steps) * H + 0.1)
ax.axis("off")
ax.grid(False)
for c, head in ((0.1, "question"), (3.6, "evidence"), (7.6, "decision")):
    ax.text(c, len(steps) * H + 0.05, head.upper(), fontsize=8, color=MUTED, va="bottom")
for n, (q, ev_, dec, kind) in enumerate(steps):
    y = (len(steps) - n - 1) * H + 0.1
    ax.add_patch(
        FancyBboxPatch(
            (0.05, y + 0.06),
            11.9,
            H - 0.14,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor="#f3f2ee",
            edgecolor="none",
        )
    )
    ax.text(
        0.2,
        y + H / 2,
        f"{n + 1}",
        fontsize=13,
        color=MUTED,
        va="center",
        ha="center",
        fontweight="bold",
    )
    ax.text(
        0.45,
        y + H / 2,
        "\n".join(textwrap.wrap(q, 34)),
        fontsize=9.5,
        va="center",
        color=INK,
        fontweight="bold",
    )
    ax.text(
        3.6, y + H / 2, "\n".join(textwrap.wrap(ev_, 46)), fontsize=8.8, va="center", color=INK2
    )
    colour, label = CHIP[kind]
    ax.add_patch(
        FancyBboxPatch(
            (7.6, y + H / 2 - 0.13),
            0.78,
            0.26,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=colour,
            edgecolor="none",
        )
    )
    ax.text(
        7.99,
        y + H / 2,
        label,
        fontsize=7.5,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(8.5, y + H / 2, "\n".join(textwrap.wrap(dec, 36)), fontsize=9, va="center", color=INK)
ax.set_title(
    "Nine questions that turned a course notebook into this repository",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
    pad=14,
)
save(fig, "questions")

# -- 2. Feature-engineering questions ------------------------------------------------
qs = fq["questions"]
labels = [
    "log(1 + absences)",
    "'uses tutoring' flag",
    "drop dormitory block",
    "6-feature compact set",
    "pairwise interactions",
]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, key, name, better in (
    (axes[0], "grade_rmse_delta", "grade: change in CV RMSE (points)", "lower is better"),
    (axes[1], "dropout_auc_delta", "dropout: change in CV AUC", "higher is better"),
):
    for j, q in enumerate(qs):
        d = np.array(q[key])
        good = (d < 0).all() if key.startswith("grade") else (d > 0).all()
        bad = (d > 0).all() if key.startswith("grade") else (d < 0).all()
        colour = AQUA if good else ORANGE if bad else GRAY
        ax.plot([d.min(), d.max()], [j, j], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(
            d.mean(), j, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2
        )
        fmt = "{:+.3f}" if key.startswith("grade") else "{:+.4f}"
        ax.annotate(
            fmt.format(d.mean()),
            (d.mean(), j),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK2,
        )
    ax.axvline(0, color=INK2, linewidth=1)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlabel(f"{name}  ({better})")
    ax.set_yticks(range(len(qs)), labels)
    ax.set_ylim(len(qs) - 0.5, -0.8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
ax0 = axes[0]
ax0.set_title(
    "Feature-engineering questions: each variant against the baseline, same folds",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
    pad=22,
)
ax0.text(
    0,
    1.03,
    "dot = mean of 5 CV repeats, line = range.  green: better in every repeat"
    "  ·  orange: worse in every repeat",
    transform=ax0.transAxes,
    fontsize=9,
    color=INK2,
)
save(fig, "feature_questions")

# -- 3. Model choice -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
gm = sel["grade"]["models"]
names = list(gm)
for j, n in enumerate(names):
    e = gm[n]["rmse"]
    chosen = n == sel["grade"]["chosen"]
    c = BLUE if chosen else GRAY
    axes[0].plot([e["low"], e["high"]], [j, j], color=c, linewidth=2)
    axes[0].plot(
        e["value"], j, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2
    )
    axes[0].annotate(
        f"{e['value']:.2f}" + ("  chosen" if chosen else ""),
        (e["value"], j),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
axes[0].set_yticks(range(len(names)), names)
axes[0].invert_yaxis()
axes[0].set_xlabel("grade: CV RMSE, points (lower is better)")
axes[0].set_xlim(5.8, 7.9)
dm = sel["dropout"]["models"]
dn = list(dm)
for j, n in enumerate(dn):
    e = dm[n]["auc"]
    chosen = n == sel["dropout"]["chosen"]
    c = BLUE if chosen else GRAY
    axes[1].plot(
        e["value"], j, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2
    )
    axes[1].annotate(
        f"AUC {e['value']:.3f} · calibration error {dm[n]['ece']['value']:.3f}"
        + ("  chosen" if chosen else ""),
        (e["value"], j),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
axes[1].set_yticks(range(len(dn)), dn)
axes[1].invert_yaxis()
axes[1].set_xlim(0.91, 0.975)
axes[1].set_xlabel("dropout: CV AUC (higher is better)")
axes[0].set_title(
    "Model choice, on the training set only (5 × 5-fold cross-validation)",
    loc="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
save(fig, "model_choice")

# -- 4. Caseload: the cohort ranked by risk --------------------------------------------
cohort = adm["cohort"]
k, n_all = adm["k"], adm["n"]
fig, ax = plt.subplots(figsize=(11, 3.4))
xs = np.arange(1, len(cohort) + 1)
ps = np.array([c["p"] for c in cohort])
cols = [SERVICE_COLOUR[c["service"]] if c["flagged_cost"] else GRID for c in cohort]
ax.bar(xs, ps, width=1.0, color=cols, linewidth=0)
n_flag = sum(c["flagged_cost"] for c in cohort)
ax.axvline(k + 0.5, color=INK, linewidth=1.2)
ax.annotate(
    f"capacity: the {k} students\nthe office can see this week",
    (k + 0.5, 0.95),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=9,
    color=INK,
    va="top",
)
ax.axhline(adm["threshold"], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.annotate(
    f"cost threshold {adm['threshold']:.3f}: {n_flag} students worth contacting",
    (n_all, adm["threshold"]),
    xytext=(0, 6),
    textcoords="offset points",
    fontsize=9,
    color=INK2,
    ha="right",
)
present = [
    s for s in SERVICE_COLOUR if any(c["service"] == s and c["flagged_cost"] for c in cohort)
]
handles = [Patch(color=SERVICE_COLOUR[s], label=SERVICE_NAME[s]) for s in present]
handles.append(Patch(color=GRID, label="below the threshold"))
ax.legend(
    handles=handles,
    frameon=False,
    loc="center right",
    fontsize=9,
    title="office (first reason)",
    title_fontsize=9,
    alignment="left",
)
ax.set_xlim(0, n_all + 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel(f"students ranked by predicted dropout risk (cohort of {n_all})")
ax.set_ylabel("P(dropout)")
ax.grid(axis="x", visible=False)
title(
    ax,
    "This term's caseload: who to see first, and which office should see them",
    "Held-out cohort scored by the frozen model. Colour = the office that owns the "
    "student's first reason.",
)
save(fig, "caseload")

# -- 5. Routing ------------------------------------------------------------------------
rt = [r for r in adm["routing"] if r["flagged_cost"] > 0]
rt.sort(key=lambda r: -r["flagged_cost"])
fig, ax = plt.subplots(figsize=(8, 0.55 * len(rt) + 1.2))
for j, r in enumerate(rt):
    ax.barh(j, r["flagged_cost"], color=SERVICE_COLOUR[r["service"]], height=0.6)
    word = "student" if r["flagged_cost"] == 1 else "students"
    ax.annotate(
        f"{r['flagged_cost']} {word} · {r['in_capacity']} in the top {k}",
        (r["flagged_cost"], j),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
ax.set_yticks(range(len(rt)), [r["label"] for r in rt])
ax.invert_yaxis()
ax.set_xlim(0, max(r["flagged_cost"] for r in rt) * 1.6)
ax.set_xlabel("students above the cost threshold")
ax.grid(axis="y", visible=False)
title(
    ax,
    "Where the referrals would go",
    "Most go to academic support: assignment completion is the model's strongest driver.",
)
save(fig, "routing")

# -- 6. Cluster check ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
ks = [r["k"] for r in kc]
axes[0].plot(
    ks,
    [r["silhouette"] for r in kc],
    color=BLUE,
    linewidth=2,
    marker="o",
    markersize=8,
    markeredgecolor=SURFACE,
    markeredgewidth=2,
)
axes[0].axhline(0.25, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
axes[0].annotate(
    "0.25: below this, no substantial structure\n(Kaufman & Rousseeuw)",
    (ks[-1], 0.25),
    xytext=(0, 6),
    textcoords="offset points",
    ha="right",
    fontsize=8.5,
    color=INK2,
)
axes[0].set_ylim(0, 0.6)
axes[0].set_xlabel("number of clusters k")
axes[0].set_ylabel("silhouette")
axes[1].plot(
    ks,
    [r["stability_ari"] for r in kc],
    color=ORANGE,
    linewidth=2,
    marker="o",
    markersize=8,
    markeredgecolor=SURFACE,
    markeredgewidth=2,
)
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("number of clusters k")
for a in axes:
    a.set_xticks(ks)
axes[1].set_ylabel("resample agreement (ARI)")
fig.text(
    0.01,
    0.97,
    "Are there dropout 'archetypes'? K-means on the risk drivers of "
    f"{adm['kmeans_check']['train_flagged_n']} flagged training students",
    fontsize=12,
    fontweight="bold",
    color=INK,
    va="top",
)
fig.text(
    0.01,
    0.9,
    "Weak separation, and partitions that change from sample to sample: "
    "named groups would be naming noise.",
    fontsize=9,
    color=INK2,
    va="top",
)
fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=4)
save(fig, "cluster_check")

# -- 7. Student card -------------------------------------------------------------------
card = adm["cards"][1]
contrib = {f: v for f, v in card["contributions"].items() if abs(v) > 1e-9}
top = sorted(contrib.items(), key=lambda t: -abs(t[1]))[:7]
fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_axes((0.2, 0.12, 0.33, 0.66))
names = [pretty(f) for f, _ in top][::-1]
vals = [v for _, v in top][::-1]
ax.barh(names, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.6)
ax.axvline(0, color=INK2, linewidth=1)
ax.set_xlabel("contribution to the risk score (log-odds)")
ax.grid(axis="y", visible=False)
ax.text(
    0,
    1.04,
    "orange raises the risk, blue lowers it",
    transform=ax.transAxes,
    fontsize=8.5,
    color=INK2,
)
fig.text(
    0.02,
    0.95,
    f"Student ranked {card['rank']} of {n_all}",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.02,
    0.89,
    f"dropout risk {card['p_dropout']:.0%} · expected grade "
    f"{card['expected_grade']:.0f}/100 · refer to: {SERVICE_NAME[card['service']]}",
    fontsize=10,
    color=INK2,
)
tx = fig.add_axes((0.58, 0.05, 0.4, 0.8))
tx.axis("off")
tx.grid(False)
tx.text(
    0,
    0.97,
    f"What would bring the risk under {adm['threshold']:.0%}?",
    fontsize=11,
    fontweight="bold",
    color=INK,
    va="top",
)
yy = 0.86
for n_, plan in enumerate(card["plans"]):
    tx.text(0, yy, f"Option {n_ + 1}", fontsize=9.5, fontweight="bold", color=BLUE, va="top")
    for st in plan["steps"]:
        unit = "" if st["lever"] in ("absences_last30d", "tutoring_sessions_month") else ""
        tx.text(
            0.2,
            yy,
            f"{pretty(st['lever'])}: {st['now']:.1f} → {st['planned']:.1f}{unit}   ({st['owner']})",
            fontsize=9.5,
            color=INK,
            va="top",
        )
        yy -= 0.075
    yy -= 0.04
if not card["plans"]:
    tx.text(
        0,
        yy,
        "No plan with at most two levers inside the observed ranges: this "
        "student needs a conversation, not a nudge.",
        fontsize=9.5,
        color=INK,
        va="top",
        wrap=True,
    )
tx.text(
    0,
    0.02,
    "\n".join(textwrap.wrap(adm["caveat"], 70)),
    fontsize=8.5,
    color=MUTED,
    va="bottom",
    style="italic",
)
save(fig, "student_card")
