"""Model selection on the training set only (5 x 5-fold cross-validation).

The selection rules are written here, before any model is fitted, and the result is
frozen in results/selection.json. 02_final_test.py refuses to run without that file,
and never changes the choice.

Rules
-----
Grade:   lowest mean CV RMSE. If a simpler model is within 0.05 points, take it.
Dropout: highest mean CV AUC. If a simpler model is within 0.005, take it.
Order of simplicity is the order of the model dictionaries in pathfinder.models.
Decision policy (stated, not tuned):
  * capacity: the office sees the top 10% of the cohort by risk;
  * cost: missing a dropout is taken to cost 5 times an unneeded meeting, so contact
    a student when P(dropout) > 1/6. The ratio 5 is a choice, not a finding; the
    sensitivity to it is reported.
"""

from __future__ import annotations

import numpy as np
from _common import save
from sklearn.metrics import average_precision_score

from pathfinder import evaluate as ev
from pathfinder.decide import bayes_threshold, expected_cost, precision_at_k, recall_at_k
from pathfinder.models import DROPOUT_MODELS, GRADE_MODELS

REPEATS, FOLDS, SEED = 5, 5, 0
CAPACITY_SHARE = 0.10
COST_RATIO = 5.0  # cost of a missed dropout / cost of an unneeded meeting
LOW_BAND = 65.0

from pathfinder.data import load_train  # noqa: E402

tr = load_train()
y_g, y_d = tr.grade.to_numpy(), tr.dropout.to_numpy()
low = y_g < LOW_BAND

# -- grade ---------------------------------------------------------------------------
grade: dict[str, dict] = {}
grade_oof: dict[str, np.ndarray] = {}
for name, make in GRADE_MODELS.items():
    oof = ev.repeated_oof(make, tr.X, tr.grade, repeats=REPEATS, folds=FOLDS, seed=SEED)
    grade_oof[name] = oof
    grade[name] = {
        "rmse": ev.over_repeats(ev.rmse, y_g, oof).as_dict(),
        "mae": ev.over_repeats(ev.mae, y_g, oof).as_dict(),
        "r2": ev.over_repeats(ev.r2, y_g, oof).as_dict(),
        "large_errors": ev.over_repeats(ev.large_errors, y_g, oof).as_dict(),
        "rmse_low_band": ev.over_repeats(ev.rmse, y_g[low], oof[:, low]).as_dict(),
        "mean_residual_low_band": float(np.mean(y_g[low] - oof[:, low])),
    }
    print(f"grade  {name:22s} RMSE {ev.over_repeats(ev.rmse, y_g, oof).fmt(2)}")

names = list(GRADE_MODELS)
best = min(names, key=lambda n: grade[n]["rmse"]["value"])
grade_choice = next(
    n for n in names if grade[n]["rmse"]["value"] <= grade[best]["rmse"]["value"] + 0.05
)
# Paired comparison on identical folds: Tobit minus the original notebook's OLS + clip.
diff = [
    ev.rmse(y_g, t) - ev.rmse(y_g, o)
    for t, o in zip(grade_oof["Tobit"], grade_oof["OLS + clip"], strict=True)
]

# -- dropout -------------------------------------------------------------------------
k = int(round(CAPACITY_SHARE * len(y_d)))
tau = bayes_threshold(cost_fn=COST_RATIO, cost_fp=1.0)
dropout: dict[str, dict] = {}
for name, make in DROPOUT_MODELS.items():
    oof = ev.repeated_oof(
        make, tr.X, tr.dropout, repeats=REPEATS, folds=FOLDS, seed=SEED, proba=True
    )
    dropout[name] = {
        "auc": ev.over_repeats(ev.auc, y_d, oof).as_dict(),
        "average_precision": ev.over_repeats(average_precision_score, y_d, oof).as_dict(),
        "brier": ev.over_repeats(ev.brier, y_d, oof).as_dict(),
        "ece": ev.over_repeats(ev.expected_calibration_error, y_d, oof).as_dict(),
        "mean_predicted": float(oof.mean()),
        "precision_at_k": ev.over_repeats(lambda y, p: precision_at_k(y, p, k), y_d, oof).as_dict(),
        "recall_at_k": ev.over_repeats(lambda y, p: recall_at_k(y, p, k), y_d, oof).as_dict(),
        "cost_per_student_at_tau": ev.over_repeats(
            lambda y, p: expected_cost(y, p > tau, COST_RATIO, 1.0), y_d, oof
        ).as_dict(),
    }
    print(
        f"dropout {name:26s} AUC {ev.over_repeats(ev.auc, y_d, oof).fmt(3)} "
        f"Brier {dropout[name]['brier']['value']:.3f} ECE {dropout[name]['ece']['value']:.3f}"
    )

dnames = list(DROPOUT_MODELS)
dbest = max(dnames, key=lambda n: dropout[n]["auc"]["value"])
dropout_choice = next(
    n for n in dnames if dropout[n]["auc"]["value"] >= dropout[dbest]["auc"]["value"] - 0.005
)

save(
    "selection",
    {
        "protocol": {"repeats": REPEATS, "folds": FOLDS, "seed": SEED, "n_train": len(y_d)},
        "grade": {
            "models": grade,
            "chosen": grade_choice,
            "reference": "OLS + clip",
            "tobit_minus_ols_clip_rmse_per_repeat": diff,
        },
        "dropout": {"models": dropout, "chosen": dropout_choice},
        "policy": {
            "capacity_share": CAPACITY_SHARE,
            "k_train": k,
            "cost_ratio_fn_to_fp": COST_RATIO,
            "threshold": tau,
        },
    },
)
print(f"chosen: grade = {grade_choice}, dropout = {dropout_choice}")
