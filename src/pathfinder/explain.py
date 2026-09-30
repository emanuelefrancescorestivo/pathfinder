"""Why a student was ranked high: exact contributions of a linear model.

For a logistic regression the log-odds are a sum, so each feature's contribution
relative to an average student is coef_j * (x_j - mean_j), exactly. These are the
SHAP values of a linear model when features are treated as independent, computed
without the shap package.

They explain the model, not the student. "Low attendance raised the score" says what
the model reacts to; it does not say that raising attendance would prevent dropout.
The original notebook's counterfactuals (DiCE) made that causal leap; this module
does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def linear_contributions(
    model: Pipeline, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame:
    """Per-student, per-feature contribution to the log-odds (or to the prediction).

    `background` defines the "average student" (normally the training set).
    """
    prep, est = model.named_steps["prep"], model.named_steps["model"]
    coef = np.ravel(est.coef_)
    names = list(prep.get_feature_names_out())
    Z = prep.transform(X)
    z0 = prep.transform(background).mean(axis=0)
    return pd.DataFrame((Z - z0) * coef, columns=names, index=X.index)


def top_reasons(contrib: pd.DataFrame, n: int = 3) -> pd.Series:
    """The n features that pushed each student's risk up the most, as text."""

    def one(row: pd.Series) -> str:
        up = row[row > 0].sort_values(ascending=False).head(n)
        return "; ".join(f"{name} (+{val:.2f})" for name, val in up.items()) or "-"

    return contrib.apply(one, axis=1)
