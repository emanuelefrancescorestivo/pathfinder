import numpy as np
import pandas as pd
import pytest

from pathfinder import counterfactual as cf
from pathfinder.data import FEATURES, RANGES
from pathfinder.explain import linear_contributions
from pathfinder.models import DROPOUT_MODELS
from pathfinder.segments import SERVICES, kmeans_check, route


def cohort(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.uniform(*RANGES.get(f, (1, 5)), n) for f in FEATURES})
    X["absences_last30d"] = rng.integers(0, 13, n)
    X["tutoring_sessions_month"] = rng.integers(0, 5, n)
    X["dormitory_block"] = rng.integers(1, 6, n)
    z = (
        -0.06 * (X["attendance_rate_pct"] - 70)
        - 0.05 * (X["assignment_completion_pct"] - 60)
        + 0.3 * (X["financial_stress_score"] - 5)
        + 0.25 * (X["absences_last30d"] - 6)
        - 0.2 * (X["internet_reliability_score"] - 5)
        - 1.8
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    model = DROPOUT_MODELS["Logistic"]().fit(X, y)
    return X, y, model


@pytest.fixture(scope="module")
def fitted():
    X, y, model = cohort()
    p = model.predict_proba(X)[:, 1]
    return X, model, cf.LinearView.of(model, X), p


def flagged_row(X, p, threshold, rank=5):
    i = np.argsort(-p)[rank]
    return X.iloc[[i]], float(p[i])


def test_single_lever_reaches_the_threshold(fitted):
    X, model, view, p = fitted
    row, pi = flagged_row(X, p, 0.2)
    table = cf.single_lever(view, row.iloc[0], pi, 0.2)
    cont = table[~table["lever"].isin(cf.INTEGER)]
    assert len(cont) > 0
    for _, r in cont.iterrows():
        after = model.predict_proba(cf.apply(row, {r["lever"]: r["change"]}))[0, 1]
        assert after == pytest.approx(0.2, abs=1e-6)
        assert np.sign(r["change"]) == cf.LEVERS[r["lever"]]


def test_integer_levers_are_rounded_past_the_threshold(fitted):
    X, model, view, p = fitted
    row, pi = flagged_row(X, p, 0.2)
    table = cf.single_lever(view, row.iloc[0], pi, 0.2)
    for _, r in table[table["lever"].isin(cf.INTEGER)].iterrows():
        assert float(r["needed"]).is_integer()
        after = model.predict_proba(cf.apply(row, {r["lever"]: r["change"]}))[0, 1]
        assert after <= 0.2 + 1e-9


def test_joint_plan_reaches_threshold_within_bounds_and_is_cheapest(fitted):
    X, model, view, p = fitted
    row, pi = flagged_row(X, p, 0.2)
    plan = cf.joint_plan(view, row.iloc[0], pi, 0.2)
    assert plan is not None and len(plan) > 0
    changes = dict(zip(plan["lever"], plan["change"], strict=True))
    after = model.predict_proba(cf.apply(row, changes))[0, 1]
    assert after == pytest.approx(0.2, abs=1e-6)
    for _, r in plan.iterrows():
        assert view.low[r["lever"]] - 1e-9 <= r["planned"] <= view.high[r["lever"]] + 1e-9
        assert np.sign(r["change"]) == cf.LEVERS[r["lever"]]
    # minimum norm: no feasible single-lever answer is cheaper in standard deviations
    single = cf.single_lever(view, row.iloc[0], pi, 0.2)
    effort = float(np.sqrt((plan["effort_sd"] ** 2).sum()))
    for _, r in single[single["feasible"] & ~single["lever"].isin(cf.INTEGER)].iterrows():
        assert effort <= abs(r["change"]) / view.scale[r["lever"]] + 1e-9


def test_joint_plan_respects_bounds_when_one_lever_saturates(fitted):
    X, model, view, p = fitted
    row, pi = flagged_row(X, p, 0.05, rank=0)
    plan = cf.joint_plan(view, row.iloc[0], pi, 0.05)
    if plan is None:
        pytest.skip("gap too large for this synthetic student")
    changes = dict(zip(plan["lever"], plan["change"], strict=True))
    assert model.predict_proba(cf.apply(row, changes))[0, 1] == pytest.approx(0.05, abs=1e-6)
    for _, r in plan.iterrows():
        assert view.low[r["lever"]] - 1e-9 <= r["planned"] <= view.high[r["lever"]] + 1e-9


def test_joint_plan_reports_impossible(fitted):
    X, model, view, p = fitted
    row, pi = flagged_row(X, p, 1e-9, rank=0)
    assert cf.joint_plan(view, row.iloc[0], pi, 1e-9) is None


def test_route_follows_the_first_reason_not_the_service_total():
    cols = [f for fs in SERVICES.values() for f in fs] + ["dormitory_block_2"]
    c = pd.DataFrame(0.0, index=[0, 1, 2, 3], columns=cols)
    c.loc[0, "financial_stress_score"] = 1.0
    c.loc[1, "attendance_rate_pct"] = 0.6
    c.loc[1, "absences_last30d"] = 0.6  # engagement totals 1.2 ...
    c.loc[1, "quiz_average_pct"] = 1.0  # ... but the first reason is academic
    c.loc[2, :] = -0.1
    c.loc[3, "dormitory_block_2"] = 5.0  # not owned by any service
    c.loc[3, "commute_minutes"] = 0.2
    assert list(route(c)) == ["financial", "academic", "none", "access"]


def test_kmeans_check_separates_real_clusters_from_noise():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(m, 0.1, (60, 3)) for m in (0, 3, 6)])
    noise = rng.uniform(size=(180, 3))
    good = kmeans_check(pd.DataFrame(blobs), ks=range(3, 4), n_resamples=10)
    bad = kmeans_check(pd.DataFrame(noise), ks=range(3, 4), n_resamples=10)
    assert good["silhouette"].iloc[0] > 0.8 and good["stability_ari"].iloc[0] > 0.9
    assert bad["silhouette"].iloc[0] < 0.5


def test_contributions_feed_routing(fitted):
    X, model, view, p = fitted
    c = linear_contributions(model, X.iloc[:50], X)
    r = route(c)
    assert set(r) <= set(SERVICES) | {"none"}


def test_sparse_plans_are_small_whole_diverse_and_reach_the_threshold(fitted):
    X, model, view, p = fitted
    checked = 0
    for rank in range(0, 40, 4):
        row, pi = flagged_row(X, p, 0.2, rank=rank)
        plans = cf.sparse_plans(view, row.iloc[0], pi, 0.2, max_levers=2, n=3)
        used: set[str] = set()
        for plan in plans:
            assert 1 <= len(plan) <= 2
            levers = set(plan["lever"])
            assert levers - used  # each plan brings a lever the earlier ones did not
            used |= levers
            for _, r in plan.iterrows():
                assert np.sign(r["change"]) == cf.LEVERS[r["lever"]]
                assert view.low[r["lever"]] - 1e-9 <= r["planned"] <= view.high[r["lever"]] + 1e-9
                if r["lever"] in cf.INTEGER:
                    assert abs(r["change"] - round(r["change"])) < 1e-9
            changes = dict(zip(plan["lever"], plan["change"], strict=True))
            assert model.predict_proba(cf.apply(row, changes))[0, 1] <= 0.2 + 1e-6
            checked += 1
        efforts = [float(np.sqrt((pl["effort_sd"] ** 2).sum())) for pl in plans]
        assert efforts == sorted(efforts)
    assert checked > 5
