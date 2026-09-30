"""Discrete-time survival: the weekly hazard of withdrawing, given the student is still here.

Each student-module contributes one row per week they are still registered at the start
of that week ("person-period" data). The row's label is 1 if they unregister during that
week. A student who never withdraws contributes rows until the last week observed and
no event: they are censored, not labelled "stayed". This is the same idea as the Tobit
model for grades: say what was not observed instead of pretending it was.

The hazard model is a logistic regression on the week's features plus a smooth function
of the week (cubic splines), so it can learn that withdrawal is more likely at some
points of the course than others. From weekly hazards h_w, the risk of withdrawing
within the next H weeks is 1 - prod(1 - h_{w+j}), j < H, with the student's current
features held fixed: at week w nothing later is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .oulad import Oulad, snapshot, withdrawal_day

WEEK = "week_of_course"


def person_period(
    d: Oulad, weeks: list[int], *, trajectory: bool = True, demographic: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack one snapshot per week; label = withdraws during that week."""
    frames, labels = [], []
    for w in weeks:
        X, _ = snapshot(d, w, trajectory=trajectory, demographic=demographic)
        day = withdrawal_day(d, X.index).to_numpy()
        event = (day >= 7 * w) & (day < 7 * (w + 1))
        X = X.assign(**{WEEK: float(w)})
        X.index = pd.MultiIndex.from_arrays(
            [*(X.index.get_level_values(i) for i in range(X.index.nlevels)), [w] * len(X)],
            names=[*X.index.names, "week"],
        )
        frames.append(X)
        labels.append(pd.Series(event.astype(int), index=X.index, name="event"))
    X = pd.concat(frames).fillna(0.0)
    return X, pd.concat(labels)


def horizon_label(d: Oulad, index: pd.Index, week: int, horizon: int) -> pd.Series:
    """1 if the student unregisters within `horizon` weeks of the start of `week`."""
    day = withdrawal_day(d, index).to_numpy()
    y = (day >= 7 * week) & (day < 7 * (week + horizon))
    return pd.Series(y.astype(int), index=index, name=f"withdraw_within_{horizon}w")


class HazardModel:
    """Logistic hazard on features + spline(week). Fit on person-period rows."""

    def __init__(self, n_knots: int = 6, C: float = 1.0) -> None:
        self.n_knots = n_knots
        self.C = C

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HazardModel:
        self.columns_ = [c for c in X.columns if c != WEEK]
        prep = ColumnTransformer(
            [
                ("x", StandardScaler(), self.columns_),
                (
                    "week",
                    SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="constant"),
                    [WEEK],
                ),
            ]
        )
        self.pipe_ = Pipeline(
            [("prep", prep), ("model", LogisticRegression(C=self.C, max_iter=5000))]
        ).fit(X[self.columns_ + [WEEK]], y)
        return self

    def hazard(self, X: pd.DataFrame, week: float) -> np.ndarray:
        Z = X.reindex(columns=self.columns_, fill_value=0.0).assign(**{WEEK: float(week)})
        return self.pipe_.predict_proba(Z)[:, 1]

    def risk_within(self, X: pd.DataFrame, week: int, horizon: int) -> np.ndarray:
        """P(withdraw in weeks week .. week + horizon - 1 | features at `week`)."""
        survive = np.ones(len(X))
        for j in range(horizon):
            survive *= 1.0 - self.hazard(X, week + j)
        return 1.0 - survive
