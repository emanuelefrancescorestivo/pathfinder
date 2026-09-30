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


# -- time-aware comparison ---------------------------------------------------------------
#
# The question an office asks is not "will this student ever withdraw?" but "who is
# likely to leave in the next few weeks?". Three scorers answer it at each landmark week:
#
#   landmark            one logistic model per week, snapshot features, label = withdraws
#                       within `horizon` weeks
#   landmark+trajectory the same with trend features (click slope, recent vs earlier
#                       activity, missed first assessment)
#   survival            ONE discrete-time hazard model for all weeks (survival.py), risk
#                       within `horizon` weeks from the weekly hazards
#
# All three are trained on the training presentations and scored on the test one.


def weekly_scores(
    d: Oulad,
    weeks: list[int],
    *,
    horizon: int = 4,
    train: tuple[str, ...] = TRAIN,
    test: tuple[str, ...] = TEST,
) -> tuple[dict[str, dict[int, pd.Series]], dict[int, pd.Series]]:
    """Scores of each scorer for the test students still registered at each week."""
    from .survival import HazardModel, horizon_label, person_period

    scores: dict[str, dict[int, pd.Series]] = {
        "landmark": {},
        "landmark+trajectory": {},
        "survival": {},
    }
    labels: dict[int, pd.Series] = {}

    Xp, yp = person_period(d, list(range(0, max(weeks) + horizon)), trajectory=True)
    in_train = np.isin(presentation_of(Xp.index), train)
    hazard = HazardModel().fit(Xp[in_train], yp[in_train])

    for w in weeks:
        for name, traj in (("landmark", False), ("landmark+trajectory", True)):
            X, _ = snapshot(d, w, trajectory=traj)
            y = horizon_label(d, X.index, w, horizon)
            pres = presentation_of(X.index)
            tr, te = np.isin(pres, train), np.isin(pres, test)
            labels[w] = y[te]
            if y[tr].nunique() < 2:
                continue  # no withdrawals in this window in the training data
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            p = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            scores[name][w] = pd.Series(p, index=X.index[te])
        Xs, _ = snapshot(d, w, trajectory=True)
        te = np.isin(presentation_of(Xs.index), test)
        scores["survival"][w] = pd.Series(
            hazard.risk_within(Xs[te], w, horizon), index=Xs.index[te]
        )
    return scores, labels


def horizon_table(
    scores: dict[str, dict[int, pd.Series]],
    labels: dict[int, pd.Series],
    *,
    capacity_share: float = 0.10,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """AUC (with interval), precision@k and calibration for each scorer and week."""
    rows = []
    for name, by_week in scores.items():
        for w, s in by_week.items():
            y = labels[w].reindex(s.index).to_numpy()
            p = s.to_numpy()
            if y.sum() == 0 or y.sum() == len(y):
                continue
            k = max(1, int(round(capacity_share * len(y))))
            a = ev.bootstrap(ev.auc, y, p, n_boot=n_boot, seed=w, stratify=y)
            rows.append(
                {
                    "week": w,
                    "scorer": name,
                    "n": len(y),
                    "events": int(y.sum()),
                    "auc": a.value,
                    "auc_low": a.low,
                    "auc_high": a.high,
                    "precision_at_k": precision_at_k(y, p, k),
                    "recall_at_k": recall_at_k(y, p, k),
                    "mean_predicted": float(p.mean()),
                    "observed_rate": float(y.mean()),
                    "brier": ev.brier(y, p),
                }
            )
    return pd.DataFrame(rows)


def lead_times(
    d: Oulad,
    by_week: dict[int, pd.Series],
    *,
    capacity_share: float = 0.10,
) -> pd.DataFrame:
    """For each test student who withdrew during the course: when were they first flagged?

    A student is flagged at week w if they are in the top `capacity_share` of that week's
    still-registered cohort. lead = withdrawal week - first flagged week (weeks before
    withdrawal that the office would have had). Students never flagged before
    withdrawing get NaN.
    """
    from .oulad import withdrawal_day

    first: dict[tuple, int] = {}
    for w in sorted(by_week):
        s = by_week[w]
        k = max(1, int(round(capacity_share * len(s))))
        for key in s.sort_values(ascending=False, kind="stable").index[:k]:
            first.setdefault(key, w)
    everyone = pd.MultiIndex.from_tuples(
        sorted({key for s in by_week.values() for key in s.index}),
        names=next(iter(by_week.values())).index.names,
    )
    day = withdrawal_day(d, everyone)
    wk = np.floor(day / 7)
    rows = []
    for key, left in zip(everyone, wk, strict=True):
        if np.isnan(left) or left > max(by_week):
            continue  # did not withdraw within the weeks scored
        f = first.get(key)
        lead = float(left - f) if f is not None and f <= left else np.nan
        rows.append(
            {"student": key, "withdrawal_week": int(left), "first_flag_week": f, "lead_weeks": lead}
        )
    return pd.DataFrame(rows)


def lead_summary(leads: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Share of withdrawals flagged at least 2 weeks ahead, with a bootstrap interval.

    "flagged_before_withdrawal" counts a flag in the withdrawal week itself (lead 0).

    One row per student, so resampling rows resamples students.
    """
    ahead = (leads["lead_weeks"] >= 2).to_numpy().astype(float)
    flagged = leads["lead_weeks"].notna().to_numpy().astype(float)
    e2 = ev.bootstrap(lambda a: float(a.mean()), ahead, n_boot=n_boot, seed=seed)
    e0 = ev.bootstrap(lambda a: float(a.mean()), flagged, n_boot=n_boot, seed=seed)
    return {
        "withdrawals": int(len(leads)),
        "flagged_before_withdrawal": e0.as_dict(),
        "flagged_2_weeks_ahead": e2.as_dict(),
        "median_lead_weeks": float(leads["lead_weeks"].median()) if flagged.any() else None,
    }
