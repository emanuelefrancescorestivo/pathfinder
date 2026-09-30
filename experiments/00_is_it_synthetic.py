"""Evidence that the course dataset is synthetic. Training set only.

Real student data does not have attendance, quiz scores and prior GPA spread
uniformly between two round numbers. Nor do students who dropped out have a final
grade. Both hold here.
"""

from __future__ import annotations

from _common import save
from scipy import stats

from pathfinder.data import NUMERIC, TARGET_DROPOUT, TARGET_GRADE, load_train

tr = load_train()
df = tr.X.assign(**{TARGET_GRADE: tr.grade, TARGET_DROPOUT: tr.dropout})

uniform = {}
for col in NUMERIC:
    x = df[col].to_numpy(float)
    if df[col].nunique() < 20:  # counts: not a continuous uniform candidate
        continue
    u = (x - x.min()) / (x.max() - x.min())
    uniform[col] = {
        "min": float(x.min()),
        "max": float(x.max()),
        "ks_p_uniform": float(stats.kstest(u, "uniform").pvalue),
    }

by_status = df.groupby(TARGET_DROPOUT)[TARGET_GRADE].mean()
save(
    "synthetic",
    {
        "n_train": len(df),
        "uniform_check": uniform,
        "n_features_consistent_with_uniform_p_gt_0_5": int(
            sum(v["ks_p_uniform"] > 0.5 for v in uniform.values())
        ),
        "share_grade_at_100": float((df[TARGET_GRADE] == 100).mean()),
        "dropout_rate": float(df[TARGET_DROPOUT].mean()),
        "mean_grade_dropouts": float(by_status.loc[1]),
        "mean_grade_stayers": float(by_status.loc[0]),
    },
)
for k, v in uniform.items():
    print(f"{k:28s} KS p(uniform) = {v['ks_p_uniform']:.3f}")
print(f"grades exactly 100: {(df[TARGET_GRADE] == 100).mean():.1%}")
