"""What-if analysis: the smallest changes that would bring a student under the threshold.

This is what DiCE (Mothilal, Sharma and Tan, 2020) computes by search. For a logistic
regression the answer has a closed form, so it is computed exactly here instead of
approximated: the log-odds are linear in the standardised features, and bringing the
risk under a threshold means lowering them by a known amount.

Two kinds of answer are produced for each student:

* single-lever what-ifs: for each lever alone, the value it would need to reach, and
  whether that value was ever observed in the training data (if not, the answer is an
  extrapolation and is marked infeasible);
* a joint plan: the combination of levers with the smallest total change, measured in
  standard deviations, subject to each lever moving only in its allowed direction and
  staying inside the observed range.

What these numbers are: the change *the model* would need to see. What they are not: a
promise that making the change would keep the student enrolled. The model is
correlational and was fitted on synthetic data. Every output carries that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Who can move each feature, and in which direction lowers risk if the model agrees.
# +1: the lever is raised, -1: lowered. Features not listed are fixed (prior GPA,
# commute, dormitory): no what-if is proposed on them.
STUDENT_LEVERS: dict[str, int] = {
    "attendance_rate_pct": +1,
    "assignment_completion_pct": +1,
    "quiz_average_pct": +1,
    "hours_self_study_week": +1,
    "absences_last30d": -1,
    "participation_score": +1,
}
INSTITUTION_LEVERS: dict[str, int] = {
    "tutoring_sessions_month": +1,  # offer tutoring
    "internet_reliability_score": +1,  # lend equipment, campus access
    "financial_stress_score": -1,  # financial aid
}
LEVERS = STUDENT_LEVERS | INSTITUTION_LEVERS
INTEGER = {"absences_last30d", "tutoring_sessions_month"}

CAVEAT = (
    "What the model would need to see, not a guarantee: the model is correlational and "
    "was trained on synthetic data."
)


@dataclass(frozen=True)
class LinearView:
    """The fitted logistic pipeline, unrolled into raw feature units."""

    intercept: float
    weight: pd.Series  # log-odds per standard deviation, numeric features only
    mean: pd.Series
    scale: pd.Series
    low: pd.Series  # observed training range
    high: pd.Series

    @classmethod
    def of(cls, model: Pipeline, train_X: pd.DataFrame) -> LinearView:
        prep, est = model.named_steps["prep"], model.named_steps["model"]
        names = list(prep.get_feature_names_out())
        coef = pd.Series(np.ravel(est.coef_), index=names)
        scaler = prep.named_transformers_["num"]
        num = list(scaler.feature_names_in_)
        return cls(
            intercept=float(np.ravel(est.intercept_)[0]),
            weight=coef[num],
            mean=pd.Series(scaler.mean_, index=num),
            scale=pd.Series(scaler.scale_, index=num),
            low=train_X[num].min(),
            high=train_X[num].max(),
        )


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def single_lever(view: LinearView, x: pd.Series, p: float, threshold: float) -> pd.DataFrame:
    """For each lever alone: the value that brings the risk to the threshold."""
    gap = _logit(p) - _logit(threshold)  # log-odds to remove (> 0 if flagged)
    rows = []
    for f, direction in LEVERS.items():
        w = view.weight[f]
        if w * direction >= 0:  # the model does not reward moving this lever
            continue
        needed = x[f] - gap / w * view.scale[f]
        if f in INTEGER:
            needed = np.floor(needed) if direction < 0 else np.ceil(needed)
        feasible = bool(view.low[f] <= needed <= view.high[f])
        rows.append(
            {
                "lever": f,
                "owner": "student" if f in STUDENT_LEVERS else "institution",
                "now": float(x[f]),
                "needed": float(needed),
                "change": float(needed - x[f]),
                "feasible": feasible,
            }
        )
    return pd.DataFrame(rows).sort_values("feasible", ascending=False, kind="stable")


def _solve(
    view: LinearView, x: pd.Series, gap: float, levers: dict[str, int]
) -> dict[str, float] | None:
    """Minimum-norm standardised changes that remove `gap` log-odds, or None.

    Water-filling: dz_j = min(room_j, lambda * |w_j|) with lambda raised until the gap
    closes. Levers that hit their bound are pinned; lambda only grows as levers are
    pinned, so a pinned lever never needs releasing.
    """
    useful = [f for f, d in levers.items() if view.weight[f] * d < 0]
    if not useful:
        return None
    w = np.abs(view.weight[useful].to_numpy())
    room = np.array(
        [
            ((view.high[f] - x[f]) if levers[f] > 0 else (x[f] - view.low[f])) / view.scale[f]
            for f in useful
        ]
    )
    dz = np.zeros(len(useful))
    free = room > 1e-12
    remaining = gap
    while remaining > 1e-9 and free.any():
        step = remaining * w[free] / np.sum(w[free] ** 2)
        over = step > room[free]
        if not over.any():
            dz[free] = step
            remaining = 0.0
            break
        for i in np.flatnonzero(free)[over]:
            dz[i] = room[i]
            remaining -= room[i] * w[i]
            free[i] = False
    if remaining > 1e-6:
        return None
    return {f: float(dz[i]) * levers[f] for i, f in enumerate(useful) if dz[i] > 1e-9}


def _rows(view: LinearView, x: pd.Series, dz: dict[str, float]) -> pd.DataFrame:
    rows = []
    for f, z in dz.items():
        change = z * view.scale[f]
        rows.append(
            {
                "lever": f,
                "owner": "student" if f in STUDENT_LEVERS else "institution",
                "now": float(x[f]),
                "planned": float(x[f] + change),
                "change": float(change),
                "effort_sd": abs(z),
            }
        )
    return pd.DataFrame(rows).sort_values("effort_sd", ascending=False, ignore_index=True)


def joint_plan(
    view: LinearView,
    x: pd.Series,
    p: float,
    threshold: float,
    levers: dict[str, int] | None = None,
) -> pd.DataFrame | None:
    """Smallest combined change (in standard deviations) that reaches the threshold.

    Minimises sum(dz_j^2) subject to sum(w_j dz_j) = -gap, each dz_j in its allowed
    direction and inside the observed range. Returns None when the levers together
    cannot close the gap. The answer usually moves every lever a little, which is
    exact but impractical; `sparse_plans` is what an office would read.
    """
    gap = _logit(p) - _logit(threshold)
    if gap <= 0:
        return pd.DataFrame(columns=["lever", "owner", "now", "planned", "change", "effort_sd"])
    dz = _solve(view, x, gap, levers or LEVERS)
    return None if dz is None else _rows(view, x, dz)


def sparse_plans(
    view: LinearView,
    x: pd.Series,
    p: float,
    threshold: float,
    max_levers: int = 2,
    n: int = 3,
) -> list[pd.DataFrame]:
    """Up to n diverse plans, each moving at most `max_levers` levers.

    This is DiCE's idea (sparse, diverse counterfactuals) solved exactly: every subset
    of levers of size <= max_levers is solved in closed form; integer levers (absences,
    tutoring sessions) are rounded to whole units in the helpful direction and the
    remaining gap is re-solved on the other levers. Plans are ranked by total effort in
    standard deviations, and each plan kept must use a lever no earlier plan used.
    """
    gap = _logit(p) - _logit(threshold)
    if gap <= 0:
        return []
    useful = [f for f, d in LEVERS.items() if view.weight[f] * d < 0]
    found: list[tuple[float, dict[str, float]]] = []
    for size in range(1, max_levers + 1):
        for subset in combinations(useful, size):
            levers = {f: LEVERS[f] for f in subset}
            dz = _solve(view, x, gap, levers)
            if dz is None:
                continue
            ints = {f: z for f, z in dz.items() if f in INTEGER}
            if ints:
                fixed = {
                    f: np.ceil(abs(z) * view.scale[f] - 1e-9) / view.scale[f] * LEVERS[f]
                    for f, z in ints.items()
                }
                if any(
                    abs(z)
                    > ((view.high[f] - x[f]) if LEVERS[f] > 0 else (x[f] - view.low[f]))
                    / view.scale[f]
                    + 1e-9
                    for f, z in fixed.items()
                ):
                    continue
                left = gap - sum(abs(z) * abs(view.weight[f]) for f, z in fixed.items())
                rest = {f: d for f, d in levers.items() if f not in INTEGER}
                extra = {} if left <= 1e-9 else _solve(view, x, left, rest) if rest else None
                if extra is None:
                    continue
                dz = fixed | extra
            if len(dz) != size:  # a lever went unused: the smaller subset covers it
                continue
            found.append((float(np.sqrt(sum(z * z for z in dz.values()))), dz))
    found.sort(key=lambda t: t[0])
    chosen: list[dict[str, float]] = []
    used: set[str] = set()
    for _, dz in found:
        if set(dz) - used:
            chosen.append(dz)
            used |= set(dz)
        if len(chosen) == n:
            break
    return [_rows(view, x, dz) for dz in chosen]


def apply(row: pd.DataFrame, changes: dict[str, float]) -> pd.DataFrame:
    """A one-row frame with the changes applied, ready for model.predict_proba."""
    out = row.copy()
    for f, c in changes.items():
        out[f] = out[f].astype(float) + c
    return out
