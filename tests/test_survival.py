"""Person-period data, the hazard model and lead times, on fixtures with OULAD's schema."""

import numpy as np
import pandas as pd
import pytest
from test_early_warning import random_oulad
from test_oulad import as_oulad, fixture

from pathfinder import early_warning as ew
from pathfinder.survival import HazardModel, horizon_label, person_period


def test_person_period_rows_stop_at_the_event_and_censor_the_rest():
    X, y = person_period(as_oulad(fixture()), list(range(8)), trajectory=True)
    per = pd.DataFrame(
        {
            "id": X.index.get_level_values("id_student"),
            "week": X.index.get_level_values("week"),
            "event": y.to_numpy(),
        }
    )
    s3 = per[per["id"] == 3]  # unregistered on day 10: week 1
    assert list(s3["week"]) == [0, 1] and list(s3["event"]) == [0, 1]
    s2 = per[per["id"] == 2]  # unregistered on day 40: week 5
    assert list(s2["week"]) == [0, 1, 2, 3, 4, 5] and s2["event"].sum() == 1
    assert s2.loc[s2["event"] == 1, "week"].item() == 5
    s1 = per[per["id"] == 1]  # never withdrew: observed every week, no event
    assert list(s1["week"]) == list(range(8)) and s1["event"].sum() == 0


def test_horizon_label():
    d = as_oulad(fixture())
    X, _ = person_period(d, [0], trajectory=False)
    idx = X.index.droplevel("week")
    assert list(horizon_label(d, idx, 0, 1)) == [0, 0, 0]  # nobody leaves in days 0-6
    assert list(horizon_label(d, idx, 0, 2)) == [0, 0, 1]  # student 3 leaves on day 10
    assert list(horizon_label(d, idx, 0, 6)) == [0, 1, 1]


@pytest.fixture(scope="module")
def synthetic():
    return random_oulad(n_per_pres=120, seed=3)


def test_hazard_model_risk_is_a_product_of_weekly_hazards(synthetic):
    X, y = person_period(synthetic, list(range(10)), trajectory=True)
    m = HazardModel().fit(X, y)
    X4 = X.xs(4, level="week")
    h = [m.hazard(X4, 4 + j) for j in range(3)]
    expected = 1 - (1 - h[0]) * (1 - h[1]) * (1 - h[2])
    np.testing.assert_allclose(m.risk_within(X4, 4, 3), expected)
    np.testing.assert_allclose(m.risk_within(X4, 4, 1), h[0])
    r = [m.risk_within(X4, 4, H) for H in (1, 2, 4, 8)]
    assert all((a <= b + 1e-12).all() for a, b in zip(r, r[1:], strict=False))
    assert ((r[-1] >= 0) & (r[-1] <= 1)).all()


def test_weekly_scores_and_tables_end_to_end(synthetic):
    scores, labels = ew.weekly_scores(synthetic, [2, 6, 10], horizon=4)
    assert set(scores) == {"landmark", "landmark+trajectory", "survival"}
    table = ew.horizon_table(scores, labels, n_boot=50)
    assert {"auc", "auc_low", "auc_high", "precision_at_k"} <= set(table.columns)
    surv = table[table["scorer"] == "survival"]
    assert (surv["auc"] > 0.55).all()  # the generator ties withdrawal to low activity
    leads = ew.lead_times(synthetic, scores["survival"])
    assert (leads["lead_weeks"].dropna() >= 0).all()
    assert (
        leads["first_flag_week"].dropna()
        <= leads["withdrawal_week"][leads["first_flag_week"].notna()]
    ).all()
    summary = ew.lead_summary(leads, n_boot=50)
    assert 0 <= summary["flagged_2_weeks_ahead"]["value"] <= 1


def test_lead_times_by_hand():
    d = as_oulad(fixture())  # student 3 leaves in week 1, student 2 in week 5
    idx = pd.MultiIndex.from_tuples(
        [("AAA", "2013J", 1), ("AAA", "2013J", 2), ("AAA", "2013J", 3)],
        names=["code_module", "code_presentation", "id_student"],
    )
    by_week = {
        0: pd.Series([0.1, 0.2, 0.9], index=idx),  # top 10% of 3 = 1 student: student 3
        1: pd.Series([0.1, 0.8, 0.3], index=idx),  # student 2 flagged in week 1
        2: pd.Series([0.1, 0.8], index=idx[:2]),
        5: pd.Series([0.1, 0.8], index=idx[:2]),
    }
    t = ew.lead_times(d, by_week)
    lead = dict(zip(t["student"], t["lead_weeks"], strict=True))
    assert lead[("AAA", "2013J", 3)] == 1  # flagged week 0, left week 1
    assert lead[("AAA", "2013J", 2)] == 4  # flagged week 1, left week 5
    assert ("AAA", "2013J", 1) not in lead  # never withdrew
