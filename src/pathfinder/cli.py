"""`pathfinder rank`: the advisor's weekly list.

    pathfinder rank --train data/raw/track_f_student_success_train.csv \\
                    --students new_students.csv --capacity 30 --out contact_list.csv

Fits the selected models on the training file, then ranks the new students by
dropout risk and marks the top `capacity` for contact. Each row carries the risk,
the expected grade, and the three features that raised the risk most.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .data import FEATURES, DataError, load, validate
from .decide import top_k
from .explain import linear_contributions, top_reasons
from .models import DROPOUT_MODELS, GRADE_MODELS

DISCLOSURE = (
    "Risk scores come from a model trained on the course's synthetic dataset. They rank "
    "students for a conversation; they are not a verdict. The reasons list what the "
    "model reacts to, not what would change the outcome."
)

# The models chosen by experiments/02_select.py (cross-validation only).
DROPOUT_MODEL = "Logistic"
GRADE_MODEL = "Tobit"


def rank(train: Path, students: pd.DataFrame, capacity: int) -> pd.DataFrame:
    tr = load(train)
    validate(students, require_targets=False)
    X = students[FEATURES]
    clf = DROPOUT_MODELS[DROPOUT_MODEL]().fit(tr.X, tr.dropout)
    reg = GRADE_MODELS[GRADE_MODEL]().fit(tr.X, tr.grade)
    p = clf.predict_proba(X)[:, 1]
    out = students.copy()
    out["p_dropout"] = p.round(3)
    out["expected_grade"] = reg.predict(X).round(1)
    out["contact"] = top_k(p, min(capacity, len(X)))
    out["reasons"] = top_reasons(linear_contributions(clf, X, tr.X))
    out.insert(0, "rank", pd.Series(p, index=out.index).rank(ascending=False, method="first"))
    return out.sort_values("rank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathfinder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="rank students by dropout risk")
    r.add_argument("--train", type=Path, required=True)
    r.add_argument("--students", type=Path, required=True)
    r.add_argument("--capacity", type=int, default=30, help="students the office can see")
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        ranked = rank(args.train, pd.read_csv(args.students), args.capacity)
    except (DataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ranked.to_csv(args.out, index=False)
    n = int(ranked["contact"].sum())
    print(f"{len(ranked)} students ranked, {n} marked for contact -> {args.out}")
    print(DISCLOSURE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
