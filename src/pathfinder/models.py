"""The candidate models for both tasks, each a full pipeline (preprocessing inside).

Preprocessing lives inside the pipeline so that every cross-validation fold fits its
own scaler: nothing learned from a validation fold reaches training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .censored import TobitRegressor
from .data import CATEGORICAL, GRADE_CEILING, NUMERIC

RANDOM_STATE = 42


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Clip another regressor's output to [lower, upper]. The original notebook's fix."""

    def __init__(self, estimator: BaseEstimator, lower: float = 0.0, upper: float = 100.0):
        self.estimator = estimator
        self.lower = lower
        self.upper = upper

    def fit(self, X: ArrayLike, y: ArrayLike) -> ClippedRegressor:
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return np.clip(self.estimator_.predict(X), self.lower, self.upper)


def _pipe(model: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", preprocessor()), ("model", model)])


# Each entry builds a fresh, unfitted pipeline.
GRADE_MODELS: dict[str, Callable[[], Pipeline]] = {
    "OLS + clip": lambda: _pipe(ClippedRegressor(LinearRegression())),
    "Ridge + clip": lambda: _pipe(ClippedRegressor(RidgeCV(alphas=np.logspace(-2, 3, 30)))),
    "Tobit": lambda: _pipe(TobitRegressor(upper=GRADE_CEILING)),
    "Gradient boosting": lambda: _pipe(
        HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=3, early_stopping=True, random_state=RANDOM_STATE
        )
    ),
}

DROPOUT_MODELS: dict[str, Callable[[], Pipeline]] = {
    "Logistic": lambda: _pipe(LogisticRegression(max_iter=2000)),
    "Logistic, balanced weights": lambda: _pipe(
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
    "Gradient boosting": lambda: _pipe(
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, early_stopping=True, random_state=RANDOM_STATE
        )
    ),
}
