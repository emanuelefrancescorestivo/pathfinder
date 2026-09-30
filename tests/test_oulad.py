"""The OULAD snapshot on a hand-made fixture with the documented schema."""

from pathlib import Path

import pandas as pd
import pytest

from pathfinder import oulad

M, P = "AAA", "2013J"


def fixture() -> dict[str, pd.DataFrame]:
    info = pd.DataFrame(
        {
            "code_module": M,
            "code_presentation": P,
            "id_student": [1, 2, 3],
            "gender": ["F", "M", "F"],
            "region": "R",
            "highest_education": "HE",
            "imd_band": "10-20%",
            "age_band": "0-35",
            "num_of_prev_attempts": [0, 1, 0],
            "studied_credits": [60, 120, 60],
            "disability": "N",
            "final_result": ["Pass", "Withdrawn", "Withdrawn"],
        }
    )
    reg = pd.DataFrame(
        {
            "code_module": M,
            "code_presentation": P,
            "id_student": [1, 2, 3],
            "date_registration": [-30, -10, -5],
            # student 3 left on day 10: gone before a week-2 cutoff (day 14)
            "date_unregistration": [None, 40, 10],
        }
    )
    vle = pd.DataFrame(
        {
            "code_module": M,
            "code_presentation": P,
            "id_student": [1, 1, 1, 2, 3],
            "id_site": 7,
            "date": [-3, 2, 13, 5, 1],
            "sum_click": [4, 10, 6, 3, 9],
        }
    )
    assessments = pd.DataFrame(
        {
            "code_module": M,
            "code_presentation": P,
            "id_assessment": [100, 101, 102],
            "assessment_type": ["TMA", "TMA", "Exam"],
            "date": [10, 30, 200],
            "weight": [10, 20, 100],
        }
    )
    sa = pd.DataFrame(
        {
            "id_assessment": [100, 100],
            "id_student": [1, 2],
            "date_submitted": [9, 12],
            "is_banked": 0,
            "score": [80.0, 50.0],
        }
    )
    return {
        "studentInfo": info,
        "studentRegistration": reg,
        "studentVle": vle,
        "assessments": assessments,
        "studentAssessment": sa,
    }


def as_oulad(f: dict[str, pd.DataFrame]) -> oulad.Oulad:
    return oulad.Oulad(
        f["studentInfo"],
        f["studentRegistration"],
        f["studentVle"],
        f["assessments"],
        f["studentAssessment"],
    )


def test_snapshot_features_and_labels():
    X, y = oulad.snapshot(as_oulad(fixture()), week=2)
    ids = list(X.index.get_level_values("id_student"))
    assert ids == [1, 2]  # student 3 unregistered on day 10 < 14
    assert list(y) == [0, 1]
    one, two = X.iloc[0], X.iloc[1]
    assert one["clicks_total"] == 20 and one["active_days"] == 3
    assert one["clicks_last_14d"] == 16  # days 2 and 13; day -3 is outside [0, 14)
    assert one["days_since_last_click"] == 1
    assert one["assessments_due"] == 1 and one["assessments_missing"] == 0
    assert one["assessments_late"] == 0 and one["mean_score"] == 80
    assert two["assessments_late"] == 1  # submitted day 12, due day 10
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert "code_module_AAA" in X.columns


def test_nothing_after_the_cutoff_reaches_a_feature():
    f = fixture()
    before, _ = oulad.snapshot(as_oulad(f), week=2)
    g = {k: v.copy() for k, v in f.items()}
    g["studentVle"] = pd.concat(
        [
            g["studentVle"],
            pd.DataFrame(
                {
                    "code_module": M,
                    "code_presentation": P,
                    "id_student": [1, 2],
                    "id_site": 7,
                    "date": [14, 90],
                    "sum_click": [1000, 1000],
                }
            ),
        ]
    )
    g["studentAssessment"] = pd.concat(
        [
            g["studentAssessment"],
            pd.DataFrame(
                {
                    "id_assessment": [101],
                    "id_student": [1],
                    "date_submitted": [14],
                    "is_banked": 0,
                    "score": [0.0],
                }
            ),
        ]
    )
    after, _ = oulad.snapshot(as_oulad(g), week=2)
    pd.testing.assert_frame_equal(before, after)


def test_demographics_are_opt_in():
    X, _ = oulad.snapshot(as_oulad(fixture()), week=2)
    assert not any(c.startswith("gender") for c in X.columns)
    Xd, _ = oulad.snapshot(as_oulad(fixture()), week=2, demographic=True)
    assert any(c.startswith("gender") for c in Xd.columns)


def test_load_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(oulad.SchemaError, match="not found"):
        oulad.load(tmp_path)


def test_load_reads_question_marks_as_missing(tmp_path: Path):
    f = fixture()
    for name, df in f.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    text = (tmp_path / "studentAssessment.csv").read_text().replace("50.0", "?")
    (tmp_path / "studentAssessment.csv").write_text(text)
    d = oulad.load(tmp_path)
    assert d.student_assessment["score"].isna().sum() == 1
