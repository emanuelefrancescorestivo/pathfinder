"""Tobit (censored normal) regression as a scikit-learn estimator.

Why: 18.9% of training grades are exactly 100. A grade of 100 does not mean the
student's "latent" performance was 100, only that it was at least 100. Least squares
treats those rows as exact, which flattens every slope (attenuation bias), and then
predicts above 100, which the original notebook patched by clipping.

The Tobit model states the censoring instead:

    Y* = b0 + X b + e,   e ~ N(0, s^2)
    Y  = min(Y*, upper)            (and max(., lower) if a floor is given)

and fits (b0, b, log s) by maximum likelihood. A censored row contributes
log P(Y* >= upper) instead of a density. `predict` returns E[Y], the expected
*observed* grade, which lies in [lower, upper] by construction; `predict_latent`
returns E[Y*].
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _logpdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return -0.5 * z * z - _LOG_SQRT_2PI


class TobitRegressor(RegressorMixin, BaseEstimator):
    """Linear regression with a normal error, censored above and/or below.

    Parameters
    ----------
    upper, lower : censoring points. A target value at or beyond one is censored.
    alpha : L2 penalty on the slopes (not the intercept or the scale). 0 = plain MLE.
    max_iter : L-BFGS iterations.
    """

    def __init__(
        self,
        upper: float | None = 100.0,
        lower: float | None = None,
        alpha: float = 0.0,
        max_iter: int = 1000,
    ) -> None:
        self.upper = upper
        self.lower = lower
        self.alpha = alpha
        self.max_iter = max_iter

    # -- likelihood ---------------------------------------------------------------

    def _nll_and_grad(
        self, w: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        b0, b, log_s = w[0], w[1:-1], w[-1]
        s = np.exp(log_s)
        mu = b0 + X @ b
        up = self._up_mask
        lo = self._lo_mask
        mid = ~(up | lo)

        ll = 0.0
        d_mu = np.zeros_like(y)  # d loglik / d mu, per row
        d_logs = 0.0

        z = (y[mid] - mu[mid]) / s
        ll += float(np.sum(_logpdf(z) - log_s))
        d_mu[mid] = z / s
        d_logs += float(np.sum(z * z - 1.0))

        if up.any():
            a = (self.upper - mu[up]) / s
            log_sf = log_ndtr(-a)
            lam = np.exp(_logpdf(a) - log_sf)  # inverse Mills ratio
            ll += float(np.sum(log_sf))
            d_mu[up] = lam / s
            d_logs += float(np.sum(lam * a))

        if lo.any():
            c = (self.lower - mu[lo]) / s
            log_cdf = log_ndtr(c)
            kap = np.exp(_logpdf(c) - log_cdf)
            ll += float(np.sum(log_cdf))
            d_mu[lo] = -kap / s
            d_logs += float(np.sum(-kap * c))

        n = len(y)
        nll = -ll / n + 0.5 * self.alpha * float(b @ b) / n
        grad = np.empty_like(w)
        grad[0] = -np.sum(d_mu) / n
        grad[1:-1] = -(X.T @ d_mu) / n + self.alpha * b / n
        grad[-1] = -d_logs / n
        return nll, grad

    # -- estimator API ------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> TobitRegressor:
        X_arr, y_arr = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        if self.upper is not None and self.lower is not None and self.lower >= self.upper:
            raise ValueError("lower must be below upper")
        self._up_mask = (
            y_arr >= self.upper if self.upper is not None else np.zeros(len(y_arr), bool)
        )
        self._lo_mask = (
            y_arr <= self.lower if self.lower is not None else np.zeros(len(y_arr), bool)
        )
        if (self._up_mask | self._lo_mask).all():
            raise ValueError("every target is censored; the model is not identified")

        # Start from least squares on the uncensored rows.
        keep = ~(self._up_mask | self._lo_mask)
        A = np.column_stack([np.ones(keep.sum()), X_arr[keep]])
        coef, *_ = np.linalg.lstsq(A, y_arr[keep], rcond=None)
        resid = y_arr[keep] - A @ coef
        w0 = np.r_[coef, np.log(max(float(np.std(resid)), 1e-6))]

        res = minimize(
            self._nll_and_grad,
            w0,
            args=(X_arr, y_arr),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": 1e-8},
        )
        if not res.success:
            raise RuntimeError(f"Tobit fit did not converge: {res.message}")
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:-1].copy()
        self.sigma_ = float(np.exp(res.x[-1]))
        self.n_iter_ = int(res.nit)
        self.censored_fraction_ = float((self._up_mask | self._lo_mask).mean())
        del self._up_mask, self._lo_mask
        return self

    def predict_latent(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "coef_")
        X_arr = validate_data(self, X, reset=False, dtype=np.float64)
        return self.intercept_ + X_arr @ self.coef_

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """E[min(max(Y*, lower), upper)], the expected observed value."""
        mu = self.predict_latent(X)
        s = self.sigma_
        # Contributions: below the floor, between the bounds, above the ceiling.
        if self.upper is not None:
            a = (self.upper - mu) / s
            p_below_up, phi_up = ndtr(a), np.exp(_logpdf(a))
        else:
            p_below_up, phi_up = np.ones_like(mu), np.zeros_like(mu)
        if self.lower is not None:
            c = (self.lower - mu) / s
            p_below_lo, phi_lo = ndtr(c), np.exp(_logpdf(c))
        else:
            p_below_lo, phi_lo = np.zeros_like(mu), np.zeros_like(mu)
        out = mu * (p_below_up - p_below_lo) + s * (phi_lo - phi_up)
        if self.upper is not None:
            out = out + self.upper * (1 - p_below_up)
        if self.lower is not None:
            out = out + self.lower * p_below_lo
        return out

    def prob_at_ceiling(self, X: ArrayLike) -> NDArray[np.float64]:
        """P(Y* >= upper): the chance the observed grade is the maximum."""
        if self.upper is None:
            raise ValueError("no upper censoring point")
        mu = self.predict_latent(X)
        return 1.0 - ndtr((self.upper - mu) / self.sigma_)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = True
        return tags
