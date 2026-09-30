import json
from pathlib import Path

import pandas as pd
import pytest

from pathfinder import cli
from pathfinder.data import DATA_DIR, FEATURES, TRAIN_FILE, DataError, validate

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (DATA_DIR / TRAIN_FILE).exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="course dataset not in data/raw")


def row(**over):
    base = {f: 1.0 for f in FEATURES} | {
        "dormitory_block": 2,
        "final_grade_100": 70.0,
        "dropped_out": 0,
    }
    return pd.DataFrame([base | over])


def test_validate_accepts_a_good_row():
    validate(row())


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"attendance_rate_pct": 140.0}, "outside"),
        ({"dropped_out": 2}, "0 or 1"),
        ({"prior_gpa_20": None}, "missing values"),
    ],
)
def test_validate_rejects(bad, message):
    with pytest.raises(DataError, match=message):
        validate(row(**bad))


def test_validate_reports_missing_columns():
    with pytest.raises(DataError, match="missing columns"):
        validate(row().drop(columns="commute_minutes"))


@needs_data
def test_cli_ranks_and_marks_capacity(tmp_path: Path, capsys):
    students = (
        pd.read_csv(DATA_DIR / TRAIN_FILE).head(40).drop(columns=["final_grade_100", "dropped_out"])
    )
    src, out = tmp_path / "s.csv", tmp_path / "out.csv"
    students.to_csv(src, index=False)
    code = cli.main(
        [
            "rank",
            "--train",
            str(DATA_DIR / TRAIN_FILE),
            "--students",
            str(src),
            "--capacity",
            "5",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    ranked = pd.read_csv(out)
    assert ranked["contact"].sum() == 5
    assert ranked["p_dropout"].is_monotonic_decreasing
    assert (
        ranked.loc[ranked["contact"], "p_dropout"].min()
        >= ranked.loc[~ranked["contact"], "p_dropout"].max()
    )
    assert ranked["expected_grade"].between(0, 100).all()
    assert "not a verdict" in capsys.readouterr().out


def test_cli_refuses_a_bad_file(tmp_path: Path, capsys):
    bad = tmp_path / "bad.csv"
    row(attendance_rate_pct=500.0).to_csv(bad, index=False)
    code = cli.main(
        ["rank", "--train", str(bad), "--students", str(bad), "--out", str(tmp_path / "o.csv")]
    )
    assert code == 2
    assert "outside" in capsys.readouterr().err


def test_cli_uses_the_models_the_selection_froze():
    path = ROOT / "results" / "selection.json"
    if not path.exists():
        pytest.skip("results/selection.json not generated")
    sel = json.loads(path.read_text(encoding="utf-8"))
    assert sel["dropout"]["chosen"] == cli.DROPOUT_MODEL
    assert sel["grade"]["chosen"] == cli.GRADE_MODEL
