"""The questions behind the feature set, each answered by cross-validation (training only).

Every question is a variant of the chosen pipelines (Tobit for the grade, logistic
regression for dropout), compared with the baseline on the same folds, 5 repeats.
A variant "helps" only if it improves the metric in every repeat, "hurts" if it
worsens it in every repeat; otherwise the effect is not consistent and the simpler
option is kept.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from _common import save
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from pathfinder import evaluate as ev
from pathfinder.censored import TobitRegressor
from pathfinder.data import CATEGORICAL, GRADE_CEILING, NUMERIC, load_train

REPEATS, FOLDS, SEED = 5, 5, 0


def build(
    model: Callable[[], object],
    numeric: list[str],
    categorical: list[str],
    add: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    interactions: bool = False,
) -> Callable[[], Pipeline]:
    def make() -> Pipeline:
        num: list = [("scale", StandardScaler())]
        if interactions:
            num.append(("pairs", PolynomialFeatures(2, interaction_only=True, include_bias=False)))
        parts = [("num", Pipeline(num), numeric)]
        if categorical:
            parts.append(
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    categorical,
                )
            )
        steps = [("prep", ColumnTransformer(parts)), ("model", model())]
        if add is not None:
            steps.insert(0, ("fe", FunctionTransformer(add)))
        return Pipeline(steps)

    return make


def log_absences(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(log1p_absences=np.log1p(X["absences_last30d"]))


def tutoring_flag(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(tutoring_any=(X["tutoring_sessions_month"] > 0).astype(float))


COMPACT = [
    "attendance_rate_pct",
    "assignment_completion_pct",
    "participation_score",
    "quiz_average_pct",
    "prior_gpa_20",
    "hours_self_study_week",
]
no_abs = [f for f in NUMERIC if f != "absences_last30d"]

QUESTIONS = [
    {
        "id": "log_absences",
        "question": "Absences have a long right tail. Does log(1 + absences) fit better?",
        "origin": "notebook cell 23",
        "numeric": no_abs + ["log1p_absences"],
        "categorical": CATEGORICAL,
        "add": log_absences,
    },
    {
        "id": "tutoring_flag",
        "question": "Many students never use tutoring. Does a 'uses tutoring' flag add "
        "anything to the session count?",
        "origin": "notebook cell 23",
        "numeric": NUMERIC + ["tutoring_any"],
        "categorical": CATEGORICAL,
        "add": tutoring_flag,
    },
    {
        "id": "drop_dormitory",
        "question": "The dormitory block failed a chi-square test. Is anything lost by "
        "dropping it?",
        "origin": "notebook cell 27",
        "numeric": NUMERIC,
        "categorical": [],
    },
    {
        "id": "compact_six",
        "question": "The notebook proposed six features as 'most of the explainable "
        "variance'. Is the compact set as good?",
        "origin": "notebook cell 32",
        "numeric": COMPACT,
        "categorical": [],
    },
    {
        "id": "interactions",
        "question": "Do pairwise interactions (e.g. attendance x assignments) capture "
        "something the additive model misses?",
        "origin": "notebook cells 70, 81",
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "interactions": True,
    },
]

tr = load_train()
yg, yd = tr.grade.to_numpy(), tr.dropout.to_numpy()


def tobit() -> TobitRegressor:
    return TobitRegressor(upper=GRADE_CEILING)


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=3000)


base_g = ev.repeated_oof(
    build(tobit, NUMERIC, CATEGORICAL), tr.X, tr.grade, repeats=REPEATS, folds=FOLDS, seed=SEED
)
base_d = ev.repeated_oof(
    build(logistic, NUMERIC, CATEGORICAL),
    tr.X,
    tr.dropout,
    repeats=REPEATS,
    folds=FOLDS,
    seed=SEED,
    proba=True,
)


def verdict(deltas: list[float], better_is_lower: bool) -> str:
    d = np.array(deltas) * (1 if better_is_lower else -1)
    if (d < 0).all():
        return "helps"
    if (d > 0).all():
        return "hurts"
    return "no consistent effect"


out = []
for q in QUESTIONS:
    kw = {"add": q.get("add"), "interactions": q.get("interactions", False)}
    g = ev.repeated_oof(
        build(tobit, q["numeric"], q["categorical"], **kw),
        tr.X,
        tr.grade,
        repeats=REPEATS,
        folds=FOLDS,
        seed=SEED,
    )
    d = ev.repeated_oof(
        build(logistic, q["numeric"], q["categorical"], **kw),
        tr.X,
        tr.dropout,
        repeats=REPEATS,
        folds=FOLDS,
        seed=SEED,
        proba=True,
    )
    d_rmse = [ev.rmse(yg, a) - ev.rmse(yg, b) for a, b in zip(g, base_g, strict=True)]
    d_auc = [ev.auc(yd, a) - ev.auc(yd, b) for a, b in zip(d, base_d, strict=True)]
    d_brier = [ev.brier(yd, a) - ev.brier(yd, b) for a, b in zip(d, base_d, strict=True)]
    row = {
        "id": q["id"],
        "question": q["question"],
        "origin": q["origin"],
        "grade_rmse_delta": d_rmse,
        "grade_verdict": verdict(d_rmse, True),
        "dropout_auc_delta": d_auc,
        "dropout_auc_verdict": verdict(d_auc, False),
        "dropout_brier_delta": d_brier,
        "dropout_brier_verdict": verdict(d_brier, True),
    }
    out.append(row)
    print(
        f"{q['id']:15s} dRMSE {np.mean(d_rmse):+.3f} ({row['grade_verdict']:20s}) "
        f"dAUC {np.mean(d_auc):+.4f} ({row['dropout_auc_verdict']})"
    )

save(
    "feature_questions",
    {"protocol": {"repeats": REPEATS, "folds": FOLDS, "seed": SEED}, "questions": out},
)
