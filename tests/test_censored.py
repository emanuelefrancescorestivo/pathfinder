import numpy as np
import pytest
from scipy.optimize import check_grad
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

from pathfinder.censored import TobitRegressor


def simulate(n=4000, upper=100.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta = np.array([6.0, -3.0, 2.0])
    y_star = 85 + X @ beta + rng.normal(0, 7, n)
    return X, np.minimum(y_star, upper), beta


def test_without_censoring_it_is_least_squares():
    """Analytic reference: with no censored rows the Tobit MLE is OLS, and sigma is
    the maximum-likelihood residual scale sqrt(RSS / n)."""
    X, y, _ = simulate(upper=1e9)
    tob = TobitRegressor(upper=1e9).fit(X, y)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(tob.coef_, ols.coef_, atol=1e-4)
    assert tob.intercept_ == pytest.approx(ols.intercept_, abs=1e-4)
    rss = np.sum((y - ols.predict(X)) ** 2)
    assert tob.sigma_ == pytest.approx(np.sqrt(rss / len(y)), rel=1e-4)


def test_recovers_parameters_that_least_squares_attenuates():
    X, y, beta = simulate(upper=92.0, seed=1)
    assert (y == 92).mean() > 0.2  # about as censored as the real grades
    tob = TobitRegressor(upper=92).fit(X, y)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(tob.coef_, beta, atol=0.35)
    assert tob.sigma_ == pytest.approx(7, abs=0.3)
    # Least squares on the censored target shrinks every slope toward zero, by roughly
    # the same factor (here about 0.76 to 0.88).
    assert np.all(ols.coef_ / beta < 0.9)


def test_gradient_matches_finite_differences():
    X, y, _ = simulate(n=300, seed=2)
    y = np.maximum(y, 70.0)  # add a floor so all three branches are exercised
    m = TobitRegressor(upper=100, lower=70, alpha=0.5)
    m._up_mask, m._lo_mask = y >= 100, y <= 70
    assert m._up_mask.any() and m._lo_mask.any()
    w = np.r_[80.0, 1.0, -1.0, 0.5, np.log(6.0)]
    err = check_grad(
        lambda w: m._nll_and_grad(w, X, y)[0], lambda w: m._nll_and_grad(w, X, y)[1], w
    )
    assert err < 1e-5


def test_predictions_are_inside_the_bounds_and_expected_value_is_right():
    X, y, _ = simulate(seed=3)
    m = TobitRegressor(upper=100).fit(X, y)
    p = m.predict(X)
    assert p.max() <= 100 and p.min() > 0
    # E[min(Y*,100)] by Monte Carlo for one student
    mu = m.predict_latent(X[:1])[0]
    draws = np.minimum(np.random.default_rng(0).normal(mu, m.sigma_, 400_000), 100)
    assert p[0] == pytest.approx(draws.mean(), abs=0.05)
    assert m.prob_at_ceiling(X[:1])[0] == pytest.approx((draws == 100).mean(), abs=0.005)


def test_lower_bound_prediction_matches_monte_carlo():
    X, y, _ = simulate(seed=4)
    y = np.maximum(y, 75.0)
    m = TobitRegressor(upper=100, lower=75).fit(X, y)
    mu = m.predict_latent(X[:1])[0]
    draws = np.clip(np.random.default_rng(1).normal(mu, m.sigma_, 400_000), 75, 100)
    assert m.predict(X[:1])[0] == pytest.approx(draws.mean(), abs=0.05)


def test_works_with_sklearn_tools():
    X, y, _ = simulate(n=500, seed=5)
    p = cross_val_predict(clone(TobitRegressor()), X, y, cv=3)
    assert p.shape == y.shape


def test_rejects_fully_censored_target():
    X = np.ones((5, 1))
    with pytest.raises(ValueError, match="identified"):
        TobitRegressor(upper=1).fit(X, np.ones(5))
