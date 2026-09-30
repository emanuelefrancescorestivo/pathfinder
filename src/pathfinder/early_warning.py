"""How early can withdrawal be predicted? One model per cutoff week, evaluated forward in time.

Design choices, and why:

* Temporal split. Models are trained on earlier presentations and tested on a later one
  (by default 2013B, 2013J, 2014B -> 2014J). A random split would let the model learn a
  presentation's quirks from its own students, which a real deployment never can.
* One snapshot per week. At week w the cohort is the students still registered, so the
  task gets harder as the obvious early leavers drop out of it. The curve reports the
  cohort size and base rate next to every AUC for that reason.
* Two model families, both reported, neither chosen on the test presentation: the same
  logistic regression as the main project and a gradient-boosted tree model.
* Capacity: precision and recall in the top 10% of the test cohort by risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .decide import precision_at_k, recall_at_k
from .oulad import Oulad, presentation_of, snapshot

TRAIN = ("2013B", "2013J", "2014B")
TEST = ("2014J",)


def models() -> dict[str, object]:
    return {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, early_stopping=True, random_state=42
        ),
    }


def curve(
    d: Oulad,
    weeks: list[int],
    *,
    train: tuple[str, ...] = TRAIN,
    test: tuple[str, ...] = TEST,
    demographic: bool = False,
    capacity_share: float = 0.10,
    n_boot: int = 1000,
) -> pd.DataFrame:
    rows = []
    for w in weeks:
        X, y = snapshot(d, w, demographic=demographic)
        pres = presentation_of(X.index)
        tr, te = np.isin(pres, train), np.isin(pres, test)
        if y[tr].nunique() < 2 or y[te].nunique() < 2:
            raise ValueError(f"week {w}: a split has a single class")
        k = max(1, int(round(capacity_share * te.sum())))
        for name, model in models().items():
            p = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]  # type: ignore[attr-defined]
            yt = y[te].to_numpy()
            a = ev.bootstrap(ev.auc, yt, p, n_boot=n_boot, seed=w, stratify=yt)
            rows.append(
                {
                    "week": w,
                    "model": name,
                    "n_train": int(tr.sum()),
                    "n_test": int(te.sum()),
                    "withdrawal_rate_test": float(yt.mean()),
                    "auc": a.value,
                    "auc_low": a.low,
                    "auc_high": a.high,
                    "k": k,
                    "precision_at_k": precision_at_k(yt, p, k),
                    "recall_at_k": recall_at_k(yt, p, k),
                    "brier": ev.brier(yt, p),
                }
            )
    return pd.DataFrame(rows)
