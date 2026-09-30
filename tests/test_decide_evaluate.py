import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pathfinder import decide
from pathfinder import evaluate as ev


def test_top_k_and_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    assert decide.top_k(p, 2).tolist() == [True, True, False, False, False, False]
    assert decide.precision_at_k(y, p, 3) == pytest.approx(2 / 3)
    assert decide.recall_at_k(y, p, 3) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        decide.top_k(p, 7)


def test_bayes_threshold_minimises_expected_cost_on_calibrated_probabilities():
    """With outcomes drawn from the stated probabilities, flagging above the Bayes
    threshold costs less than any other threshold on a grid."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 200_000)
    y = rng.uniform(size=p.size) < p
    tau = decide.bayes_threshold(cost_fn=5, cost_fp=1)
    assert tau == pytest.approx(1 / 6)
    best = decide.expected_cost(y, p > tau, 5, 1)
    for t in np.linspace(0, 0.6, 25):
        assert best <= decide.expected_cost(y, p > t, 5, 1) + 1e-3


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.uniform(size=500), 1)  # many ties
    assert ev.auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_bootstrap_interval_contains_the_point_and_covers_the_truth():
    rng = np.random.default_rng(2)
    covered = 0
    for s in range(100):
        x = rng.normal(10, 2, 200)
        e = ev.bootstrap(lambda a: float(np.mean(a)), x, n_boot=400, seed=s)
        assert e.low <= e.value <= e.high
        covered += e.low <= 10 <= e.high
    assert covered >= 88  # nominal 95; loose bound for 100 trials


def test_stratified_bootstrap_keeps_class_counts():
    y = np.array([1] * 5 + [0] * 95)
    e = ev.bootstrap(lambda a: float(a.sum()), y, n_boot=200, stratify=y)
    assert e.low == e.high == 5


def test_calibration_error_is_zero_for_a_perfect_forecaster():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    y = np.concatenate(
        [
            np.r_[np.ones(int(q * 1000)), np.zeros(1000 - int(q * 1000))]
            for q in np.linspace(0.05, 0.95, 10)
        ]
    )
    assert ev.expected_calibration_error(y, p) == pytest.approx(0, abs=1e-3)
