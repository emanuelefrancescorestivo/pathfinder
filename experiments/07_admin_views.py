"""What an advising office would see for one cohort: routing, cluster check, what-ifs.

Uses the frozen models (results/selection.json) fitted on the training set and scores
the held-out cohort as if it were this term's students. No label is used to choose
which students to show: the example cards are the students at fixed risk ranks.
Labels appear only in the routing summary, to report how many flagged students in
each service actually dropped out.
"""

from __future__ import annotations

import numpy as np
from _common import load, save

from pathfinder import counterfactual as cf
from pathfinder.data import load_test, load_train
from pathfinder.decide import bayes_threshold, top_k
from pathfinder.explain import linear_contributions
from pathfinder.models import DROPOUT_MODELS, GRADE_MODELS
from pathfinder.segments import SERVICE_LABEL, kmeans_check, route

CARD_RANKS = [1, 10, 25]  # 1-based positions in the risk ranking

sel = load("selection")
tr, te = load_train(), load_test()
clf = DROPOUT_MODELS[sel["dropout"]["chosen"]]().fit(tr.X, tr.dropout)
reg = GRADE_MODELS[sel["grade"]["chosen"]]().fit(tr.X, tr.grade)
p = clf.predict_proba(te.X)[:, 1]
grade = reg.predict(te.X)
tau = bayes_threshold(sel["policy"]["cost_ratio_fn_to_fp"], 1.0)
k = int(round(sel["policy"]["capacity_share"] * len(p)))

contrib = linear_contributions(clf, te.X, tr.X)
service = route(contrib)
in_capacity = top_k(p, k)
flag_cost = p > tau
y = te.dropout.to_numpy()

order = np.argsort(-p, kind="stable")
cohort = [
    {
        "rank": int(r + 1),
        "p": float(p[i]),
        "service": service.iloc[i],
        "contact": bool(in_capacity[i]),
        "flagged_cost": bool(flag_cost[i]),
    }
    for r, i in enumerate(order)
]

summary = []
for s in [*SERVICE_LABEL, "none"]:
    m = (service == s).to_numpy()
    summary.append(
        {
            "service": s,
            "label": SERVICE_LABEL.get(s, "No dominant driver"),
            "in_capacity": int((m & in_capacity).sum()),
            "flagged_cost": int((m & flag_cost).sum()),
            "dropouts_among_flagged_cost": int((m & flag_cost & (y == 1)).sum()),
            "mean_risk_flagged_cost": float(p[m & flag_cost].mean())
            if (m & flag_cost).any()
            else None,
        }
    )

# Is there cluster structure among flagged students? (train cohort is larger)
p_tr = clf.predict_proba(tr.X)[:, 1]
c_tr = linear_contributions(clf, tr.X[p_tr > tau], tr.X)
check_train = kmeans_check(c_tr).to_dict(orient="records")
check_test = kmeans_check(contrib[flag_cost]).to_dict(orient="records")

view = cf.LinearView.of(clf, tr.X)
cards = []
for rank in CARD_RANKS:
    i = int(order[rank - 1])
    row = te.X.iloc[[i]]
    x = row.iloc[0]
    reasons = contrib.iloc[i].sort_values(ascending=False)
    single = cf.single_lever(view, x, float(p[i]), tau)
    plans = []
    for plan in cf.sparse_plans(view, x, float(p[i]), tau, max_levers=2, n=3):
        changes = dict(zip(plan["lever"], plan["change"], strict=True))
        plans.append(
            {
                "steps": plan.to_dict(orient="records"),
                "effort_sd": float((plan["effort_sd"] ** 2).sum() ** 0.5),
                "risk_after": float(clf.predict_proba(cf.apply(row, changes))[0, 1]),
            }
        )
    cards.append(
        {
            "rank": rank,
            "p_dropout": float(p[i]),
            "expected_grade": float(grade[i]),
            "service": service.iloc[i],
            "contributions": {k_: float(v) for k_, v in reasons.items()},
            "single_lever": single.to_dict(orient="records"),
            "plans": plans,
        }
    )

save(
    "admin_views",
    {
        "threshold": tau,
        "k": k,
        "n": len(p),
        "cohort": cohort,
        "routing": summary,
        "kmeans_check": {
            "train_flagged_n": int(len(c_tr)),
            "train": check_train,
            "test_flagged_n": int(flag_cost.sum()),
            "test": check_test,
        },
        "cards": cards,
        "caveat": cf.CAVEAT,
    },
)
for s in summary:
    print(f"{s['label']:26s} top-{k}: {s['in_capacity']:3d}  flagged: {s['flagged_cost']:3d}")
print(
    "k-means silhouette (train flagged):",
    [round(r["silhouette"], 3) for r in check_train],
    "stability:",
    [round(r["stability_ari"], 2) for r in check_train],
)
