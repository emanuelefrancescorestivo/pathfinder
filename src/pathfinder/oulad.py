"""Early warning on real data: the Open University Learning Analytics Dataset (OULAD).

OULAD (Kuzilek, Hlosta and Zdrahal, Scientific Data, 2017) records, for 22 module
presentations, each student's demographics, registration, assessment submissions and
daily clicks in the virtual learning environment. That makes the question the course
dataset could not answer testable: *how early* can withdrawal be predicted?

For a cutoff of `week` weeks after the module start (day 7 * week), a student-module
row is built from information dated strictly before the cutoff, for students still
registered at the cutoff. The label is final_result == "Withdrawn". Nothing dated on or
after the cutoff can reach a feature; tests/test_oulad.py checks this by appending
future events and asserting the features do not move.

The schema below follows the dataset's documentation. This module has been tested on
a small hand-made fixture with that schema, not yet on the real files: the download
host was blocked in the environment where it was written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA: dict[str, list[str]] = {
    "studentInfo": [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ],
    "studentRegistration": [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration",
    ],
    "studentVle": [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ],
    "assessments": [
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    ],
    "studentAssessment": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
}
KEY = ["code_module", "code_presentation", "id_student"]
PRESENTATION = ["code_module", "code_presentation"]

# Known when the module starts.
CONTEXT = ["num_of_prev_attempts", "studied_credits", "date_registration"]
# Sensitive attributes. Off by default: whether to use them is a policy decision,
# and experiments/05_early_warning_oulad.py reports what they add.
DEMOGRAPHIC = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
BEHAVIOUR = [
    "clicks_total",
    "active_days",
    "clicks_last_14d",
    "days_since_last_click",
    "no_clicks",
    "assessments_due",
    "assessments_missing",
    "assessments_late",
    "mean_score",
    "no_score",
]


class SchemaError(ValueError):
    """An OULAD file is missing or lacks a documented column."""


@dataclass(frozen=True)
class Oulad:
    info: pd.DataFrame
    registration: pd.DataFrame
    vle: pd.DataFrame
    assessments: pd.DataFrame
    student_assessment: pd.DataFrame


def load(directory: Path) -> Oulad:
    frames = {}
    for name, cols in SCHEMA.items():
        path = directory / f"{name}.csv"
        if not path.exists():
            raise SchemaError(f"{path} not found (download OULAD and unzip it there)")
        # The published files mark missing values with "?".
        df = pd.read_csv(path, na_values=["?", ""])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SchemaError(f"{name}.csv lacks columns {missing}")
        frames[name] = df
    return Oulad(
        frames["studentInfo"],
        frames["studentRegistration"],
        frames["studentVle"],
        frames["assessments"],
        frames["studentAssessment"],
    )


def snapshot(d: Oulad, week: int, *, demographic: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels for students still registered `week` weeks after the start."""
    cutoff = 7 * week
    reg = d.registration
    still = reg["date_unregistration"].isna() | (reg["date_unregistration"] >= cutoff)
    base = d.info.merge(reg.loc[still, KEY + ["date_registration"]], on=KEY, how="inner")

    # Clicks strictly before the cutoff.
    v = d.vle[d.vle["date"] < cutoff]
    clicks = v.groupby(KEY).agg(
        clicks_total=("sum_click", "sum"),
        active_days=("date", "nunique"),
        last_click=("date", "max"),
    )
    recent = v[v["date"] >= cutoff - 14].groupby(KEY)["sum_click"].sum().rename("clicks_last_14d")
    base = base.merge(clicks, on=KEY, how="left").merge(recent, on=KEY, how="left")
    base["no_clicks"] = base["clicks_total"].isna().astype(int)
    for c in ("clicks_total", "active_days", "clicks_last_14d"):
        base[c] = base[c].fillna(0)
    base["days_since_last_click"] = (cutoff - base["last_click"]).fillna(cutoff + 30)
    base = base.drop(columns="last_click")

    # Assessments due before the cutoff (exams excluded: they come at the end).
    due = d.assessments[
        (d.assessments["date"] < cutoff) & (d.assessments["assessment_type"] != "Exam")
    ]
    n_due = due.groupby(PRESENTATION).size().rename("assessments_due")
    sub = d.student_assessment.merge(
        due[PRESENTATION + ["id_assessment", "date"]], on="id_assessment", how="inner"
    )
    sub = sub[sub["date_submitted"] < cutoff].assign(
        late=lambda s: (s["date_submitted"] > s["date"]).astype(int)
    )
    per = sub.groupby(KEY).agg(
        submitted=("id_assessment", "nunique"),
        assessments_late=("late", "sum"),
        mean_score=("score", "mean"),
    )
    base = base.merge(n_due, on=PRESENTATION, how="left").merge(per, on=KEY, how="left")
    base["assessments_due"] = base["assessments_due"].fillna(0)
    base["submitted"] = base["submitted"].fillna(0)
    base["assessments_missing"] = base["assessments_due"] - base["submitted"]
    base["assessments_late"] = base["assessments_late"].fillna(0)
    base["no_score"] = base["mean_score"].isna().astype(int)
    base["mean_score"] = base["mean_score"].fillna(0)

    cols = ["code_module"] + CONTEXT + BEHAVIOUR + (DEMOGRAPHIC if demographic else [])
    X = base[cols].copy()
    X["date_registration"] = X["date_registration"].fillna(X["date_registration"].median())
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=categorical, dtype=float)
    X.index = pd.MultiIndex.from_frame(base[KEY])
    y = pd.Series(
        (base["final_result"] == "Withdrawn").astype(int).to_numpy(),
        index=X.index,
        name="withdrawn",
    )
    return X, y


def presentation_of(index: pd.MultiIndex) -> np.ndarray:
    return np.asarray(index.get_level_values("code_presentation"))
