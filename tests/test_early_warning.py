"""The early-warning pipeline runs end to end on a random dataset with the OULAD schema.

The generated students withdraw more often when they click less, so the models should
beat chance; that is all this test claims. It says nothing about the real data.
"""

import numpy as np
import pandas as pd

from pathfinder.early_warning import curve
from pathfinder.oulad import Oulad


def random_oulad(n_per_pres: int = 150, seed: int = 0) -> Oulad:
    rng = np.random.default_rng(seed)
    info, reg, vle, asm, sa = [], [], [], [], []
    aid = 0
    for pres in ("2013B", "2013J", "2014B", "2014J"):
        for mod in ("AAA", "BBB"):
            for a_date in (20, 50):
                aid += 1
                asm.append((mod, pres, aid, "TMA", a_date, 20))
            for s in range(n_per_pres):
                sid = hash((pres, mod, s)) % 10**7
                engaged = rng.uniform()
                withdrawn = rng.uniform() > engaged
                unreg = rng.integers(30, 200) if withdrawn else None
                info.append(
                    (
                        mod,
                        pres,
                        sid,
                        "F",
                        "R",
                        "HE",
                        "10-20%",
                        "0-35",
                        int(rng.integers(0, 2)),
                        60,
                        "N",
                        "Withdrawn" if withdrawn else "Pass",
                    )
                )
                reg.append((mod, pres, sid, -int(rng.integers(1, 60)), unreg))
                for day in range(-5, 150):
                    if rng.uniform() < 0.3 * engaged:
                        vle.append((mod, pres, sid, 1, day, int(rng.integers(1, 20))))
                for a in (aid - 1, aid):
                    if rng.uniform() < engaged:
                        sa.append(
                            (a, sid, int(rng.integers(0, 60)), 0, float(rng.uniform(40, 100)))
                        )
    return Oulad(
        pd.DataFrame(
            info,
            columns=[
                "code_module",
                "code_presentation",
                "id_student",
                "gender",
                "region",
                "highest_education",
                "imd_band",
                "age_band",
                "num_of_prev_attempts",
                "studied_credits",
                "disability",
                "final_result",
            ],
        ),
        pd.DataFrame(
            reg,
            columns=[
                "code_module",
                "code_presentation",
                "id_student",
                "date_registration",
                "date_unregistration",
            ],
        ),
        pd.DataFrame(
            vle,
            columns=[
                "code_module",
                "code_presentation",
                "id_student",
                "id_site",
                "date",
                "sum_click",
            ],
        ),
        pd.DataFrame(
            asm,
            columns=[
                "code_module",
                "code_presentation",
                "id_assessment",
                "assessment_type",
                "date",
                "weight",
            ],
        ),
        pd.DataFrame(
            sa, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
        ),
    )


def test_curve_runs_and_beats_chance():
    out = curve(random_oulad(), [2, 6], n_boot=50)
    assert set(out["model"]) == {"Logistic", "Gradient boosting"}
    assert list(out["week"].unique()) == [2, 6]
    assert (out["n_test"] > 0).all()
    assert (out["auc"] > 0.6).all()
    assert ((out["auc_low"] <= out["auc"]) & (out["auc"] <= out["auc_high"])).all()
