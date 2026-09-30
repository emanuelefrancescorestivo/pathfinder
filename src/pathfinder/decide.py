"""Turning probabilities into a list of students to contact.

The original notebook picked the threshold that maximised F2 on resampled data. F2
weights recall four times as much as precision, a number nobody chose on purpose.
An advising office faces one of two concrete constraints instead:

* capacity: it can see k students this week. Then no threshold is needed at all: rank
  by risk and take the top k. The question becomes "how many dropouts are in the top k?"
  (precision@k, recall@k).
* cost: missing a dropout costs c_fn, an unnecessary meeting costs c_fp. With
  calibrated probabilities, contacting a student is worth it exactly when
  p * c_fn > (1 - p) * c_fp, that is p > c_fp / (c_fp + c_fn). The threshold follows
  from the stated costs; it is not searched for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def top_k(p: ArrayLike, k: int) -> NDArray[np.bool_]:
    """Mask of the k highest-risk students (ties broken by position, deterministically)."""
    p_ = np.asarray(p)
    if not 0 <= k <= len(p_):
        raise ValueError(f"k={k} outside [0, {len(p_)}]")
    order = np.argsort(-p_, kind="stable")
    mask = np.zeros(len(p_), bool)
    mask[order[:k]] = True
    return mask


def precision_at_k(y: ArrayLike, p: ArrayLike, k: int) -> float:
    return float(np.asarray(y)[top_k(p, k)].mean()) if k else float("nan")


def recall_at_k(y: ArrayLike, p: ArrayLike, k: int) -> float:
    y_ = np.asarray(y)
    return float(y_[top_k(p, k)].sum() / y_.sum())


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """Probability above which contacting a student has lower expected cost."""
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("costs must be positive")
    return cost_fp / (cost_fp + cost_fn)


def expected_cost(y: ArrayLike, flagged: ArrayLike, cost_fn: float, cost_fp: float) -> float:
    """Average cost per student of a set of decisions."""
    y_, f = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    return float((cost_fn * np.sum(y_ & ~f) + cost_fp * np.sum(~y_ & f)) / len(y_))


def capacity_table(y: ArrayLike, p: ArrayLike, ks: list[int]) -> pd.DataFrame:
    y_ = np.asarray(y)
    rows = []
    for k in ks:
        m = top_k(p, k)
        rows.append(
            {
                "k": k,
                "dropouts_found": int(y_[m].sum()),
                "precision_at_k": float(y_[m].mean()),
                "recall_at_k": float(y_[m].sum() / y_.sum()),
            }
        )
    return pd.DataFrame(rows)


def confusion_at(y: ArrayLike, flagged: ArrayLike) -> dict[str, int]:
    y_, f = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    return {
        "tp": int(np.sum(y_ & f)),
        "fp": int(np.sum(~y_ & f)),
        "fn": int(np.sum(y_ & ~f)),
        "tn": int(np.sum(~y_ & ~f)),
    }


def precision(y: ArrayLike, flagged: ArrayLike) -> float:
    c = confusion_at(y, flagged)
    return c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")


def recall(y: ArrayLike, flagged: ArrayLike) -> float:
    c = confusion_at(y, flagged)
    return c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
