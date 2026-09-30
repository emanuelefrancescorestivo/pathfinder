"""Which office should see a flagged student? Routing by the dominant risk driver.

The original notebook clustered flagged students with K-means and named the clusters
("The Disconnected", ...). Its own silhouette score was 0.155, which is no cluster
structure at all: the named archetypes were artefacts of asking K-means for four groups.

An advising office does not need natural clusters; it needs a routing rule it can
read, audit and argue with. So each flagged student is sent to the service that owns
the feature contributing most to that student's risk score (the student's first
listed reason). Summing contributions per service was tried first and rejected: it
favours services that own more features.

    engagement  attendance, absences, participation
    academic    assignments, quizzes, self-study, prior GPA, tutoring
    financial   financial stress
    access      internet reliability, commute

`kmeans_check` keeps the question honest: it measures whether the same contribution
vectors have cluster structure (silhouette over k, and stability across resamples),
so the README can show why a rule was chosen over clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

SERVICES: dict[str, list[str]] = {
    "engagement": ["attendance_rate_pct", "absences_last30d", "participation_score"],
    "academic": [
        "assignment_completion_pct",
        "quiz_average_pct",
        "hours_self_study_week",
        "prior_gpa_20",
        "tutoring_sessions_month",
    ],
    "financial": ["financial_stress_score"],
    "access": ["internet_reliability_score", "commute_minutes"],
}
SERVICE_LABEL = {
    "engagement": "Engagement follow-up",
    "academic": "Academic support",
    "financial": "Financial aid",
    "access": "Digital & travel access",
}


def service_scores(contrib: pd.DataFrame) -> pd.DataFrame:
    """Summed contribution to the log-odds of each service's features."""
    return pd.DataFrame(
        {
            s: contrib[[f for f in fs if f in contrib.columns]].sum(axis=1)
            for s, fs in SERVICES.items()
        },
        index=contrib.index,
    )


FEATURE_SERVICE = {f: s for s, fs in SERVICES.items() for f in fs}


def route(contrib: pd.DataFrame) -> pd.Series:
    """The service owning each student's largest positive contribution."""
    own = contrib[[c for c in contrib.columns if c in FEATURE_SERVICE]]
    best = own.idxmax(axis=1).map(FEATURE_SERVICE)
    return best.where(own.max(axis=1) > 0, "none")


def kmeans_check(
    contrib: pd.DataFrame, ks: range = range(2, 7), n_resamples: int = 50, seed: int = 0
) -> pd.DataFrame:
    """Silhouette and resampling stability of K-means on the contribution vectors.

    Stability: fit K-means on bootstrap resamples, label the full set with each fit,
    and report the mean adjusted Rand index between pairs of labelings (1 = the same
    partition every time; near 0 = the partition is an accident of the sample).
    """
    Z = contrib.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        fits = []
        for _ in range(n_resamples):
            idx = rng.integers(0, len(Z), len(Z))
            km = KMeans(k, n_init=5, random_state=int(rng.integers(1 << 30))).fit(Z[idx])
            fits.append(km.predict(Z))
        ari = [adjusted_rand_score(fits[i], fits[i + 1]) for i in range(0, len(fits) - 1, 2)]
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(Z, labels)),
                "stability_ari": float(np.mean(ari)),
            }
        )
    return pd.DataFrame(rows)
