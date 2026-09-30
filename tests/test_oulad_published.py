"""The local OULAD copy matches the counts published with the dataset (skipped if absent).

Kuzilek, Hlosta and Zdrahal (2017), Scientific Data 4, 170171: 22 module presentations,
32,593 students (registrations), 10,655,280 daily click summaries.
"""

from pathlib import Path

import pytest

from pathfinder import oulad

DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "oulad"
pytestmark = pytest.mark.skipif(
    not (DIR / "studentVle.csv").exists(), reason="OULAD not in data/raw/oulad"
)


def test_counts_match_the_publication():
    d = oulad.load(DIR)
    assert len(d.info) == 32_593
    assert d.info.groupby(["code_module", "code_presentation"]).ngroups == 22
    assert len(d.vle) == 10_655_280
    assert len(d.registration) == len(d.info)
