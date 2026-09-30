"""Metrics with uncertainty: repeated cross-validation and the bootstrap.

A single number from 300 test students (57 dropouts) is not a result; an interval is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

LARGE_ERROR = 15.0  # grade points; the original notebook's definition, kept for comparison


def rmse(y: ArrayLike, p: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y: ArrayLike, p: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y: ArrayLike, p: ArrayLike) -> float:
    y_ = np.asarray(y, float)
    return float(1 - np.sum((y_ - np.asarray(p)) ** 2) / np.sum((y_ - y_.mean()) ** 2))


def large_errors(y: ArrayLike, p: ArrayLike) -> int:
    return int(np.sum(np.abs(np.asarray(y) - np.asarray(p)) > LARGE_ERROR))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def auc(y: ArrayLike, p: ArrayLike) -> float:
    """ROC AUC by the rank-sum formula (ties get half credit)."""
    y_ = np.asarray(y).astype(bool)
    ranks = pd.Series(np.asarray(p)).rank(method="average").to_numpy()
    n1, n0 = y_.sum(), (~y_).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y_].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_table(y: ArrayLike, p: ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability against the observed rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))


def expected_calibration_error(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float(np.sum(t["n"] * np.abs(t["predicted"] - t["observed"])) / t["n"].sum())


# -- cross-validation --------------------------------------------------------------


def repeated_oof(
    make_model: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    repeats: int = 5,
    folds: int = 5,
    seed: int = 0,
    proba: bool = False,
) -> NDArray[np.float64]:
    """Out-of-fold predictions for `repeats` different fold assignments.

    Returns shape (repeats, n). Every prediction comes from a model that did not see
    that row. The spread across repeats is the part of the CV uncertainty that is due
    to the fold assignment.
    """
    out = np.empty((repeats, len(y)))
    for r in range(repeats):
        cv: KFold | StratifiedKFold
        if proba:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
            out[r] = cross_val_predict(clone(make_model()), X, y, cv=cv, method="predict_proba")[
                :, 1
            ]
        else:
            cv = KFold(folds, shuffle=True, random_state=seed + r)
            out[r] = cross_val_predict(clone(make_model()), X, y, cv=cv)
    return out


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def fmt(self, digits: int = 3) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "low": self.low, "high": self.high}


def over_repeats(metric: Callable[..., float], y: ArrayLike, oof: NDArray[np.float64]) -> Estimate:
    """Mean of a metric over CV repeats, with the min and max across repeats."""
    vals = np.array([metric(y, row) for row in oof])
    return Estimate(float(vals.mean()), float(vals.min()), float(vals.max()))


def bootstrap(
    metric: Callable[..., float],
    *arrays: ArrayLike,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    stratify: ArrayLike | None = None,
) -> Estimate:
    """Percentile bootstrap interval for metric(*arrays), resampling rows.

    With `stratify`, rows are resampled within each class so that every replicate keeps
    the observed number of positives (otherwise recall is undefined in some replicates).
    """
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    groups = (
        [np.arange(n)]
        if stratify is None
        else [np.flatnonzero(np.asarray(stratify) == v) for v in np.unique(stratify)]
    )
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        stats[i] = metric(*(a[idx] for a in arrs))
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return Estimate(float(metric(*arrs)), float(lo), float(hi))
