"""Reproduce the original notebook's dropout-pipeline defects (AUDIT.md items 3 to 5).

Runs after 01_select.py has frozen the choice, so nothing here can influence it. Uses
this repository's feature set (the notebook added log1p(absences) and a tutoring flag),
so the numbers are close to, not identical with, the notebook's printed ones.

Original procedure, step by step:
  1. scale, then SMOTE the whole training set to 50/50;
  2. 5-fold CV *on the resampled set*, pick the threshold that maximises F2;
  3. isotonic calibration (CalibratedClassifierCV, cv=5) on the resampled set;
  4. report the CV AUC of the resampled set as "5-fold CV AUC".
"""

from __future__ import annotations

import numpy as np
from _common import load, save
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pathfinder import evaluate as ev
from pathfinder.data import load_test, load_train
from pathfinder.decide import precision, recall
from pathfinder.models import DROPOUT_MODELS, RANDOM_STATE, preprocessor

load("selection")  # refuse to run before the choice is frozen

tr, te = load_train(), load_test()
y, yt = tr.dropout.to_numpy(), te.dropout.to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def f2_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    pr, rc, th = precision_recall_curve(y_true, p)
    f2 = 5 * pr * rc / np.maximum(4 * pr + rc, 1e-12)
    return float(th[int(np.nanargmax(f2[:-1]))])


# 1-2. SMOTE before CV (the original).
prep = preprocessor().fit(tr.X)
Xs, ys = SMOTE(random_state=RANDOM_STATE).fit_resample(prep.transform(tr.X), y)
lr = LogisticRegression(max_iter=2000)
oof_sm = cross_val_predict(lr, Xs, ys, cv=cv, method="predict_proba")[:, 1]
tau_sm = f2_threshold(ys, oof_sm)
fit_sm = LogisticRegression(max_iter=2000).fit(Xs, ys)
p_test_sm = fit_sm.predict_proba(prep.transform(te.X))[:, 1]

# The same procedure with SMOTE inside each fold (only real students are validated).
pipe = imb_pipeline(
    preprocessor(), SMOTE(random_state=RANDOM_STATE), LogisticRegression(max_iter=2000)
)
oof_in = cross_val_predict(pipe, tr.X, y, cv=cv, method="predict_proba")[:, 1]
tau_in = f2_threshold(y, oof_in)
p_test_in = pipe.fit(tr.X, y).predict_proba(te.X)[:, 1]

# 3. Isotonic calibration fitted on the resampled set.
cal = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="isotonic", cv=5)
p_cal = cal.fit(Xs, ys).predict_proba(prep.transform(te.X))[:, 1]
p_plain = DROPOUT_MODELS["Logistic"]().fit(tr.X, y).predict_proba(te.X)[:, 1]

audit = {
    "smote_before_cv": {
        "threshold_f2": tau_sm,
        "oof_precision": precision(ys, oof_sm >= tau_sm),
        "oof_recall": recall(ys, oof_sm >= tau_sm),
        "test_precision": precision(yt, p_test_sm >= tau_sm),
        "test_recall": recall(yt, p_test_sm >= tau_sm),
        "cv_auc_on_resampled": ev.auc(ys, oof_sm),
        "test_auc": ev.auc(yt, p_test_sm),
    },
    "smote_inside_folds": {
        "threshold_f2": tau_in,
        "oof_precision": precision(y, oof_in >= tau_in),
        "oof_recall": recall(y, oof_in >= tau_in),
        "test_precision": precision(yt, p_test_in >= tau_in),
        "test_recall": recall(yt, p_test_in >= tau_in),
        "cv_auc": ev.auc(y, oof_in),
    },
    "calibration": {
        "test_dropout_rate": float(yt.mean()),
        "train_dropout_rate": float(y.mean()),
        "isotonic_on_smote_mean_predicted": float(p_cal.mean()),
        "isotonic_on_smote_ece": ev.expected_calibration_error(yt, p_cal),
        "isotonic_on_smote_brier": ev.brier(yt, p_cal),
        "plain_logistic_mean_predicted": float(p_plain.mean()),
        "plain_logistic_ece": ev.expected_calibration_error(yt, p_plain),
        "plain_logistic_brier": ev.brier(yt, p_plain),
    },
    # Printed by the original notebook (cell numbers of the uploaded .ipynb), quoted
    # for comparison; they are not recomputed here.
    "notebook_printed": {
        "oof_precision_cell_118": 0.828,
        "oof_recall_cell_118": 0.984,
        "test_precision_cell_131": 0.540,
        "test_recall_cell_131": 0.947,
        "calibrated_mean_predicted_cell_133": 0.257,
        "cv_auc_resampled_cell_135": 0.9577,
        "test_auc_cell_135": 0.9166,
    },
}
save("audit", audit)
a = audit["smote_before_cv"]
print(
    f"SMOTE before CV : OOF precision {a['oof_precision']:.3f} recall {a['oof_recall']:.3f} "
    f"-> test precision {a['test_precision']:.3f} recall {a['test_recall']:.3f}"
)
b = audit["smote_inside_folds"]
print(
    f"SMOTE in folds  : OOF precision {b['oof_precision']:.3f} recall {b['oof_recall']:.3f} "
    f"-> test precision {b['test_precision']:.3f} recall {b['test_recall']:.3f}"
)
c = audit["calibration"]
print(
    f"calibration     : isotonic-on-SMOTE mean P {c['isotonic_on_smote_mean_predicted']:.3f}, "
    f"plain {c['plain_logistic_mean_predicted']:.3f}, observed {c['test_dropout_rate']:.3f}"
)
