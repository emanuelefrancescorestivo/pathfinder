"""Early-warning curve on OULAD: AUC and precision@10% by week, forward in time.

Needs the OULAD CSV files in data/raw/oulad/ (see data/README.md). Writes
results/early_warning.json with and without demographic attributes.
"""

from __future__ import annotations

from _common import ROOT, save

from pathfinder import oulad
from pathfinder.early_warning import TEST, TRAIN, curve

WEEKS = [0, 2, 4, 6, 8, 10, 12, 16, 20]

d = oulad.load(ROOT / "data" / "raw" / "oulad")
behaviour = curve(d, WEEKS)
with_demo = curve(d, WEEKS, demographic=True)
print(behaviour.round(3).to_string(index=False))
save(
    "early_warning",
    {
        "train_presentations": list(TRAIN),
        "test_presentations": list(TEST),
        "weeks": WEEKS,
        "behaviour_only": behaviour.to_dict(orient="records"),
        "with_demographics": with_demo.to_dict(orient="records"),
    },
)
