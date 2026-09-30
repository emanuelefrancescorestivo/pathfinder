"""`pathfinder rank`: the advisor's weekly list.

    pathfinder rank --train data/raw/track_f_student_success_train.csv \\
                    --students new_students.csv --capacity 30 --out contact_list.csv

Fits the selected models on the training file, then ranks the new students by
dropout risk and marks the top `capacity` for contact. Each row carries the risk,
the expected grade, the three features that raised the risk most, the office that
owns the first of them, and, for students marked for contact, the smallest change of
at most two levers that the model would need to see to drop the risk under the cost
threshold (a what-if, not a promise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import counterfactual as cf
from .data import FEATURES, DataError, load, validate
from .decide import bayes_threshold, top_k
from .explain import linear_contributions, top_reasons
from .models import DROPOUT_MODELS, GRADE_MODELS
from .segments import SERVICE_LABEL, route

DISCLOSURE = (
    "Risk scores come from a model trained on the course's synthetic dataset. They rank "
    "students for a conversation; they are not a verdict. The reasons list what the "
    "model reacts to, and the what-ifs what the model would need to see; neither says "
    "what would change the outcome."
)

# The models chosen by experiments/02_select.py (cross-validation only).
DROPOUT_MODEL = "Logistic"
GRADE_MODEL = "Tobit"


def _what_if(view: cf.LinearView, x: pd.Series, p: float, threshold: float) -> str:
    if p <= threshold:
        return "already under the threshold"
    plans = cf.sparse_plans(view, x, p, threshold, max_levers=2, n=1)
    if not plans:
        return "no plan with at most two levers: needs a conversation"
    return "; ".join(f"{r.lever} {r.now:g} -> {r.planned:.1f}" for r in plans[0].itertuples())


def rank(
    train: Path, students: pd.DataFrame, capacity: int, cost_ratio: float = 5.0
) -> pd.DataFrame:
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
    contrib = linear_contributions(clf, X, tr.X)
    out["reasons"] = top_reasons(contrib)
    out["office"] = route(contrib).map(lambda s: SERVICE_LABEL.get(s, "-"))
    tau = bayes_threshold(cost_ratio, 1.0)
    view = cf.LinearView.of(clf, tr.X)
    out["what_if"] = [
        _what_if(view, X.iloc[i], float(p[i]), tau) if out["contact"].iloc[i] else ""
        for i in range(len(X))
    ]
    out.insert(0, "rank", pd.Series(p, index=out.index).rank(ascending=False, method="first"))
    return out.sort_values("rank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathfinder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="rank students by dropout risk")
    r.add_argument("--train", type=Path, required=True)
    r.add_argument("--students", type=Path, required=True)
    r.add_argument("--capacity", type=int, default=30, help="students the office can see")
    r.add_argument(
        "--cost-ratio",
        type=float,
        default=5.0,
        help="cost of a missed dropout / cost of an unneeded meeting",
    )
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        ranked = rank(args.train, pd.read_csv(args.students), args.capacity, args.cost_ratio)
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
