"""The one contact with the held-out test set.

Reads the frozen choice from results/selection.json, fits it on the whole training set,
and reports test metrics with 95% bootstrap intervals. The reference model (the original
notebook's OLS + clip) is evaluated alongside for a paired comparison; it cannot change
the choice, which was made before this script ran.
"""

from __future__ import annotations

import numpy as np
from _common import load, save
from sklearn.metrics import average_precision_score

from pathfinder import evaluate as ev
from pathfinder.data import load_test, load_train
from pathfinder.decide import (
    bayes_threshold,
    capacity_table,
    confusion_at,
    precision,
    precision_at_k,
    recall,
    recall_at_k,
)
from pathfinder.models import DROPOUT_MODELS, GRADE_MODELS

N_BOOT, SEED = 2000, 0
sel = load("selection")
tr, te = load_train(), load_test()
y_g, y_d = te.grade.to_numpy(), te.dropout.to_numpy()

# -- grade ---------------------------------------------------------------------------
chosen, ref = sel["grade"]["chosen"], sel["grade"]["reference"]
p_c = GRADE_MODELS[chosen]().fit(tr.X, tr.grade).predict(te.X)
p_r = GRADE_MODELS[ref]().fit(tr.X, tr.grade).predict(te.X)
low = y_g < 65


def rmse_low(y: np.ndarray, p: np.ndarray) -> float:
    m = y < 65
    return ev.rmse(y[m], p[m])


grade = {
    "chosen": chosen,
    "reference": ref,
    "n_test": len(y_g),
    "n_low_band": int(low.sum()),
    chosen: {
        "rmse": ev.bootstrap(ev.rmse, y_g, p_c, n_boot=N_BOOT, seed=SEED).as_dict(),
        "mae": ev.bootstrap(ev.mae, y_g, p_c, n_boot=N_BOOT, seed=SEED).as_dict(),
        "r2": ev.bootstrap(ev.r2, y_g, p_c, n_boot=N_BOOT, seed=SEED).as_dict(),
        "large_errors": ev.large_errors(y_g, p_c),
        "rmse_low_band": ev.bootstrap(rmse_low, y_g, p_c, n_boot=N_BOOT, seed=SEED).as_dict(),
    },
    ref: {
        "rmse": ev.bootstrap(ev.rmse, y_g, p_r, n_boot=N_BOOT, seed=SEED).as_dict(),
        "mae": ev.bootstrap(ev.mae, y_g, p_r, n_boot=N_BOOT, seed=SEED).as_dict(),
        "r2": ev.bootstrap(ev.r2, y_g, p_r, n_boot=N_BOOT, seed=SEED).as_dict(),
        "large_errors": ev.large_errors(y_g, p_r),
        "rmse_low_band": ev.bootstrap(rmse_low, y_g, p_r, n_boot=N_BOOT, seed=SEED).as_dict(),
    },
    "rmse_difference_chosen_minus_reference": ev.bootstrap(
        lambda y, a, b: ev.rmse(y, a) - ev.rmse(y, b), y_g, p_c, p_r, n_boot=N_BOOT, seed=SEED
    ).as_dict(),
}

# -- dropout -------------------------------------------------------------------------
dchosen = sel["dropout"]["chosen"]
clf = DROPOUT_MODELS[dchosen]().fit(tr.X, tr.dropout)
p = clf.predict_proba(te.X)[:, 1]
tau = bayes_threshold(sel["policy"]["cost_ratio_fn_to_fp"], 1.0)
k = int(round(sel["policy"]["capacity_share"] * len(y_d)))
flags = p > tau


def boot(fn):  # stratified: every replicate keeps the 57 dropouts
    return ev.bootstrap(fn, y_d, p, n_boot=N_BOOT, seed=SEED, stratify=y_d).as_dict()


dropout = {
    "chosen": dchosen,
    "n_test": len(y_d),
    "n_dropouts": int(y_d.sum()),
    "dropout_rate_test": float(y_d.mean()),
    "dropout_rate_train": float(tr.dropout.mean()),
    "auc": boot(ev.auc),
    "average_precision": boot(average_precision_score),
    "brier": boot(ev.brier),
    "ece": float(ev.expected_calibration_error(y_d, p)),
    "mean_predicted": float(p.mean()),
    "capacity": {
        "k": k,
        "precision_at_k": boot(lambda y, q: precision_at_k(y, q, k)),
        "recall_at_k": boot(lambda y, q: recall_at_k(y, q, k)),
        "table": capacity_table(y_d, p, [15, 30, 45, 60, 90]).to_dict(orient="records"),
    },
    "cost": {
        "threshold": tau,
        "flagged": int(flags.sum()),
        **confusion_at(y_d, flags),
        "precision": boot(lambda y, q: precision(y, q > tau)),
        "recall": boot(lambda y, q: recall(y, q > tau)),
        "sensitivity": [
            {
                "cost_ratio": r,
                "threshold": bayes_threshold(r, 1.0),
                "flagged": int((p > bayes_threshold(r, 1.0)).sum()),
                "recall": recall(y_d, p > bayes_threshold(r, 1.0)),
                "precision": precision(y_d, p > bayes_threshold(r, 1.0)),
            }
            for r in (2.0, 5.0, 10.0, 20.0)
        ],
    },
}

save("final", {"grade": grade, "dropout": dropout})
g = grade
print(
    f"grade  {chosen}: RMSE {ev.Estimate(**g[chosen]['rmse']).fmt(2)}   "
    f"{ref}: RMSE {ev.Estimate(**g[ref]['rmse']).fmt(2)}   "
    f"diff {ev.Estimate(**g['rmse_difference_chosen_minus_reference']).fmt(2)}"
)
print(
    f"dropout {dchosen}: AUC {ev.Estimate(**dropout['auc']).fmt()}  "
    f"P@{k} {ev.Estimate(**dropout['capacity']['precision_at_k']).fmt()}  "
    f"R@{k} {ev.Estimate(**dropout['capacity']['recall_at_k']).fmt()}"
)
print(
    f"cost policy (tau={tau:.3f}): flagged {flags.sum()}, "
    f"recall {ev.Estimate(**dropout['cost']['recall']).fmt()}, "
    f"precision {ev.Estimate(**dropout['cost']['precision']).fmt()}"
)
