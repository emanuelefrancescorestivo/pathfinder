"""Loading and validating the course dataset ("track F: student success").

The dataset is the one supplied with the ML course. Its provenance is not documented
and several features are uniformly distributed (see experiments/00_is_it_synthetic.py),
so it is treated as synthetic throughout: results describe the generator, not students.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_GRADE = "final_grade_100"
TARGET_DROPOUT = "dropped_out"

NUMERIC = [
    "attendance_rate_pct",
    "assignment_completion_pct",
    "quiz_average_pct",
    "hours_self_study_week",
    "prior_gpa_20",
    "absences_last30d",
    "tutoring_sessions_month",
    "internet_reliability_score",
    "commute_minutes",
    "financial_stress_score",
    "participation_score",
]
CATEGORICAL = ["dormitory_block"]
FEATURES = NUMERIC + CATEGORICAL

# Documented ranges, used to reject a malformed file before any model sees it.
RANGES: dict[str, tuple[float, float]] = {
    "attendance_rate_pct": (0, 100),
    "assignment_completion_pct": (0, 100),
    "quiz_average_pct": (0, 100),
    "hours_self_study_week": (0, 168),
    "prior_gpa_20": (0, 20),
    "absences_last30d": (0, 31),
    "tutoring_sessions_month": (0, 31),
    "internet_reliability_score": (0, 10),
    "commute_minutes": (0, 600),
    "financial_stress_score": (0, 10),
    "participation_score": (0, 10),
    TARGET_GRADE: (0, 100),
}

GRADE_CEILING = 100.0

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TRAIN_FILE = "track_f_student_success_train.csv"
TEST_FILE = "track_f_student_success_test.csv"


class DataError(ValueError):
    """The file does not have the columns or value ranges this project expects."""


def validate(df: pd.DataFrame, *, require_targets: bool = True) -> pd.DataFrame:
    """Check columns, missing values and ranges. Returns the frame unchanged."""
    needed = FEATURES + ([TARGET_GRADE, TARGET_DROPOUT] if require_targets else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    nulls = df[needed].isna().sum()
    if nulls.any():
        raise DataError(f"missing values: {nulls[nulls > 0].to_dict()}")
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        bad = ~df[col].between(lo, hi)
        if bad.any():
            raise DataError(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")
    if require_targets and not df[TARGET_DROPOUT].isin([0, 1]).all():
        raise DataError(f"{TARGET_DROPOUT} must be 0 or 1")
    return df


@dataclass(frozen=True)
class Split:
    X: pd.DataFrame
    grade: pd.Series
    dropout: pd.Series


def load(path: Path) -> Split:
    df = validate(pd.read_csv(path))
    return Split(df[FEATURES], df[TARGET_GRADE], df[TARGET_DROPOUT].astype(int))


def load_train(data_dir: Path = DATA_DIR) -> Split:
    return load(data_dir / TRAIN_FILE)


def load_test(data_dir: Path = DATA_DIR) -> Split:
    """The held-out set. Only experiments/03_final_test.py may call this."""
    return load(data_dir / TEST_FILE)
