"""Fill the result blocks of README.md and AUDIT.md from results/*.json.

A block is the text between `<!-- BEGIN:name -->` and `<!-- END:name -->`. Everything
outside the blocks is prose and is left alone. `--check` exits non-zero if a document
is out of date (the test suite runs this).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = [ROOT / "README.md", ROOT / "AUDIT.md"]


def j(name: str) -> dict[str, Any] | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(e: dict[str, float], d: int = 3) -> str:
    return f"{e['value']:.{d}f} [{e['low']:.{d}f}, {e['high']:.{d}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def synthetic() -> str:
    s = j("synthetic")
    if s is None:
        return "_Not generated: run experiments/00_is_it_synthetic.py._"
    rows = [
        [f"`{k}`", f"{v['min']:g} to {v['max']:g}", f"{v['ks_p_uniform']:.3f}"]
        for k, v in s["uniform_check"].items()
    ]
    return (
        table(["feature", "range", "KS test p-value vs uniform"], rows)
        + f"\n\nGrades exactly 100: **{s['share_grade_at_100']:.1%}** of the training set. "
        f"Students who dropped out still have a final grade (mean "
        f"{s['mean_grade_dropouts']:.1f}, against {s['mean_grade_stayers']:.1f} for the others)."
    )


def grade_cv() -> str:
    s = j("selection")
    if s is None:
        return "_Not generated: run experiments/01_select.py._"
    g = s["grade"]
    rows = []
    for name, m in g["models"].items():
        mark = " (chosen)" if name == g["chosen"] else ""
        rows.append(
            [
                name + mark,
                ci(m["rmse"], 2),
                f"{m['mae']['value']:.2f}",
                f"{m['r2']['value']:.3f}",
                f"{m['large_errors']['value']:.1f}",
                f"{m['rmse_low_band']['value']:.2f}",
                f"{m['mean_residual_low_band']:+.2f}",
            ]
        )
    d = g["tobit_minus_ols_clip_rmse_per_repeat"]
    p = s["protocol"]
    return (
        table(
            [
                "model",
                "RMSE [min, max over repeats]",
                "MAE",
                "R²",
                "errors > 15 pts",
                "RMSE, grade < 65",
                "mean residual, grade < 65",
            ],
            rows,
        )
        + f"\n\n{p['repeats']} repeats of {p['folds']}-fold CV on the {p['n_train']} training "
        f"students. Tobit minus OLS + clip, RMSE, same folds: "
        + ", ".join(f"{x:+.3f}" for x in d)
        + f" (better in {sum(x < 0 for x in d)} of {len(d)} repeats)."
    )


def grade_test() -> str:
    f = j("final")
    if f is None:
        return "_Not generated: run experiments/02_final_test.py._"
    g = f["grade"]
    rows = [
        [
            name,
            ci(g[name]["rmse"], 2),
            ci(g[name]["mae"], 2),
            ci(g[name]["r2"]),
            str(g[name]["large_errors"]),
            ci(g[name]["rmse_low_band"], 2),
        ]
        for name in (g["chosen"], g["reference"])
    ]
    return (
        table(
            [
                "model",
                "RMSE",
                "MAE",
                "R²",
                "errors > 15 pts",
                f"RMSE, grade < 65 (n = {g['n_low_band']})",
            ],
            rows,
        )
        + f"\n\nPaired difference in RMSE ({g['chosen']} minus {g['reference']}): "
        f"**{ci(g['rmse_difference_chosen_minus_reference'], 2)}**. "
        f"{g['n_test']} test students, 95% bootstrap intervals."
    )


def dropout_cv() -> str:
    s = j("selection")
    if s is None:
        return "_Not generated: run experiments/01_select.py._"
    d = s["dropout"]
    k = s["policy"]["k_train"]
    rows = []
    for name, m in d["models"].items():
        mark = " (chosen)" if name == d["chosen"] else ""
        rows.append(
            [
                name + mark,
                ci(m["auc"]),
                f"{m['average_precision']['value']:.3f}",
                f"{m['brier']['value']:.3f}",
                f"{m['ece']['value']:.3f}",
                f"{m['mean_predicted']:.3f}",
                f"{m['precision_at_k']['value']:.3f}",
            ]
        )
    return table(
        [
            "model",
            "AUC [min, max]",
            "average precision",
            "Brier",
            "calibration error",
            "mean predicted P (true rate 0.150)",
            f"precision@{k}",
        ],
        rows,
    )


def dropout_test() -> str:
    f = j("final")
    if f is None:
        return "_Not generated: run experiments/02_final_test.py._"
    d = f["dropout"]
    cap, cost = d["capacity"], d["cost"]
    head = (
        f"Model: {d['chosen']}. {d['n_test']} test students, "
        f"{d['n_dropouts']} dropouts.\n\n"
        + table(
            [
                "AUC",
                "average precision",
                "Brier",
                "calibration error",
                "mean predicted P",
                "observed rate (train rate)",
            ],
            [
                [
                    ci(d["auc"]),
                    ci(d["average_precision"]),
                    ci(d["brier"]),
                    f"{d['ece']:.3f}",
                    f"{d['mean_predicted']:.3f}",
                    f"{d['dropout_rate_test']:.3f} ({d['dropout_rate_train']:.3f})",
                ]
            ],
        )
    )
    captab = table(
        ["students contacted (k)", "dropouts found", "precision@k", "recall@k"],
        [
            [
                str(r["k"]),
                f"{r['dropouts_found']}/{d['n_dropouts']}",
                f"{r['precision_at_k']:.2f}",
                f"{r['recall_at_k']:.2f}",
            ]
            for r in cap["table"]
        ],
    )
    sens = table(
        [
            "cost of a missed dropout / cost of a meeting",
            "threshold",
            "flagged",
            "recall",
            "precision",
        ],
        [
            [
                f"{r['cost_ratio']:g}",
                f"{r['threshold']:.3f}",
                str(r["flagged"]),
                f"{r['recall']:.2f}",
                f"{r['precision']:.2f}",
            ]
            for r in cost["sensitivity"]
        ],
    )
    return (
        head + "\n\n**Capacity policy** (contact the top k by risk). With k = "
        f"{cap['k']} (10% of the cohort): precision {ci(cap['precision_at_k'], 2)}, "
        f"recall {ci(cap['recall_at_k'], 2)}.\n\n"
        + captab
        + "\n\n**Cost policy** (contact when P > c_fp / (c_fp + c_fn)). At the stated ratio 5, "
        f"threshold {cost['threshold']:.3f}: {cost['flagged']} flagged, recall "
        f"{ci(cost['recall'], 2)}, precision {ci(cost['precision'], 2)}.\n\n" + sens
    )


def audit_smote() -> str:
    a = j("audit")
    if a is None:
        return "_Not generated: run experiments/03_audit_original.py._"
    b, i, n = a["smote_before_cv"], a["smote_inside_folds"], a["notebook_printed"]
    return table(
        ["procedure", "threshold", "validation precision / recall", "test precision / recall"],
        [
            [
                "original notebook, as printed",
                "0.258",
                f"{n['oof_precision_cell_118']:.3f} / {n['oof_recall_cell_118']:.3f}",
                f"{n['test_precision_cell_136']:.3f} / {n['test_recall_cell_136']:.3f}",
            ],
            [
                "SMOTE before CV (reproduced)",
                f"{b['threshold_f2']:.3f}",
                f"{b['oof_precision']:.3f} / {b['oof_recall']:.3f}",
                f"{b['test_precision']:.3f} / {b['test_recall']:.3f}",
            ],
            [
                "SMOTE inside each fold",
                f"{i['threshold_f2']:.3f}",
                f"{i['oof_precision']:.3f} / {i['oof_recall']:.3f}",
                f"{i['test_precision']:.3f} / {i['test_recall']:.3f}",
            ],
        ],
    ) + (
        f"\n\nCV AUC reported on the resampled set: {b['cv_auc_on_resampled']:.3f} "
        f"(notebook: {n['cv_auc_resampled_cell_135']}); honest CV AUC: {i['cv_auc']:.3f}; "
        f"test AUC: {b['test_auc']:.3f} (notebook: {n['test_auc_cell_135']})."
    )


def audit_calibration() -> str:
    a = j("audit")
    if a is None:
        return "_Not generated: run experiments/03_audit_original.py._"
    c = a["calibration"]
    return table(
        ["model", "mean predicted P", "calibration error", "Brier"],
        [
            [
                "isotonic calibration fitted on SMOTE data (original)",
                f"{c['isotonic_on_smote_mean_predicted']:.3f}",
                f"{c['isotonic_on_smote_ece']:.3f}",
                f"{c['isotonic_on_smote_brier']:.3f}",
            ],
            [
                "plain logistic regression (this repository)",
                f"{c['plain_logistic_mean_predicted']:.3f}",
                f"{c['plain_logistic_ece']:.3f}",
                f"{c['plain_logistic_brier']:.3f}",
            ],
        ],
    ) + (
        f"\n\nObserved dropout rate: {c['test_dropout_rate']:.3f} in the test set, "
        f"{c['train_dropout_rate']:.3f} in the training set."
    )


def early_warning() -> str:
    e = j("early_warning")
    if e is None:
        return (
            "_Not run yet. The OULAD files were not reachable where this was written; "
            "see data/README.md, then run experiments/05_early_warning_oulad.py._"
        )
    rows = [
        [
            str(r["week"]),
            r["model"],
            str(r["n_test"]),
            f"{r['withdrawal_rate_test']:.3f}",
            f"{r['auc']:.3f} [{r['auc_low']:.3f}, {r['auc_high']:.3f}]",
            f"{r['precision_at_k']:.2f}",
            f"{r['recall_at_k']:.2f}",
        ]
        for r in e["behaviour_only"]
    ]
    return (
        f"Trained on {', '.join(e['train_presentations'])}; tested on "
        f"{', '.join(e['test_presentations'])}. Behaviour and course context only.\n\n"
        + table(
            [
                "week",
                "model",
                "students still registered",
                "withdrawal rate",
                "AUC",
                "precision@10%",
                "recall@10%",
            ],
            rows,
        )
    )


def headline() -> str:
    s, f, a = j("selection"), j("final"), j("audit")
    if s is None or f is None or a is None:
        return "_Run experiments/run_all.py to fill this in._"
    g, d = s["grade"]["models"], f["dropout"]
    ft = f["grade"]
    cap = d["capacity"]
    b = a["smote_before_cv"]
    diff = ft["rmse_difference_chosen_minus_reference"]
    verdict = (
        "the same direction, but the interval includes zero"
        if diff["low"] < 0 < diff["high"]
        else "the interval excludes zero"
    )
    ols, tob = g["OLS + clip"], g["Tobit"]
    found = next(r for r in cap["table"] if r["k"] == cap["k"])["dropouts_found"]
    return "\n".join(
        [
            "- **Modelling the grade ceiling helps where it matters.** A Tobit model cuts "
            "the cross-validated RMSE on students below 65 from "
            f"{ols['rmse_low_band']['value']:.2f} to {tob['rmse_low_band']['value']:.2f} "
            "points and their mean over-prediction from "
            f"{-ols['mean_residual_low_band']:.1f} to {-tob['mean_residual_low_band']:.1f}. "
            f"On the {ft['n_test']}-student test set the overall RMSE difference is "
            f"{ci(diff, 2)}: {verdict}.",
            "- **The original validation over-promised.** Oversampling before "
            f"cross-validation reported precision {b['oof_precision']:.2f}; the test set "
            f"delivered {b['test_precision']:.2f}.",
            "- **A usable answer for an advising office:** contacting the "
            f"{cap['k']} highest-risk of {d['n_test']} test students finds {found} of the "
            f"{d['n_dropouts']} dropouts (precision {ci(cap['precision_at_k'], 2)}).",
            "- **All of it on synthetic data.** Several features are uniformly distributed "
            "and dropouts have final grades. The early-warning question needs real data: "
            "see *Early warning on real data* below.",
        ]
    )


def feature_questions() -> str:
    f = j("feature_questions")
    if f is None:
        return "_Not generated: run experiments/06_feature_questions.py._"
    rows = []
    for q in f["questions"]:
        g, a = q["grade_rmse_delta"], q["dropout_auc_delta"]
        rows.append(
            [
                q["question"],
                q["origin"],
                f"{sum(g) / len(g):+.3f} ({q['grade_verdict']})",
                f"{sum(a) / len(a):+.4f} ({q['dropout_auc_verdict']})",
            ]
        )
    p = f["protocol"]
    return table(
        ["question", "from", "grade: change in CV RMSE", "dropout: change in CV AUC"], rows
    ) + (
        f"\n\nMean over {p['repeats']} repeats of {p['folds']}-fold CV, each variant on the "
        'same folds as the baseline. "helps" or "hurts" means the sign held in every '
        "repeat."
    )


def routing() -> str:
    a = j("admin_views")
    if a is None:
        return "_Not generated: run experiments/07_admin_views.py._"
    rows = [
        [
            r["label"],
            str(r["in_capacity"]),
            str(r["flagged_cost"]),
            str(r["dropouts_among_flagged_cost"]),
        ]
        for r in a["routing"]
        if r["flagged_cost"] or r["in_capacity"]
    ]
    kc = a["kmeans_check"]
    sil = max(r["silhouette"] for r in kc["train"])
    ari = [r["stability_ari"] for r in kc["train"]]
    return table(
        [
            "office",
            f"in the top {a['k']}",
            f"above the cost threshold ({a['threshold']:.3f})",
            "of those, actually dropped out",
        ],
        rows,
    ) + (
        f"\n\nK-means on the risk drivers of the {kc['train_flagged_n']} flagged training "
        f"students: best silhouette {sil:.3f} over k = 2 to 6; agreement between bootstrap "
        f"refits (adjusted Rand index) falls from {max(ari):.2f} to {min(ari):.2f} as k grows."
    )


def student_card() -> str:
    a = j("admin_views")
    if a is None:
        return "_Not generated: run experiments/07_admin_views.py._"
    lines = []
    for c in a["cards"]:
        head = (
            f"- **Rank {c['rank']}**: risk {c['p_dropout']:.2f}, expected grade "
            f"{c['expected_grade']:.0f}, office: {c['service']}."
        )
        if c["plans"]:
            opts = []
            for pl in c["plans"]:
                opts.append(
                    " and ".join(
                        f"{st['lever']} {st['now']:.1f} → {st['planned']:.1f}" for st in pl["steps"]
                    )
                    + f" (effort {pl['effort_sd']:.2f} SD, risk after {pl['risk_after']:.3f})"
                )
            head += " What-ifs: " + "; or ".join(opts) + "."
        else:
            head += " No what-if with at most two levers inside the observed ranges."
        lines.append(head)
    return "\n".join(lines)


BLOCKS: dict[str, Callable[[], str]] = {
    "headline": headline,
    "synthetic": synthetic,
    "grade_cv": grade_cv,
    "grade_test": grade_test,
    "dropout_cv": dropout_cv,
    "dropout_test": dropout_test,
    "audit_smote": audit_smote,
    "audit_calibration": audit_calibration,
    "early_warning": early_warning,
    "feature_questions": feature_questions,
    "routing": routing,
    "student_card": student_card,
}
PATTERN = re.compile(r"(<!-- BEGIN:(\w+) -->\n)(.*?)(<!-- END:\2 -->)", re.S)


def render(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group(2)
        if name not in BLOCKS:
            raise SystemExit(f"unknown block {name}")
        return f"{m.group(1)}{BLOCKS[name]()}\n{m.group(4)}"

    return PATTERN.sub(sub, text)


def main(check: bool = False) -> int:
    stale = []
    for doc in DOCS:
        old = doc.read_text(encoding="utf-8")
        new = render(old)
        if new != old:
            stale.append(doc.name)
            if not check:
                doc.write_text(new, encoding="utf-8")
    if check and stale:
        print(f"out of date: {', '.join(stale)} (run experiments/render_readme.py)")
        return 1
    print("documents up to date" if check else f"rendered {', '.join(d.name for d in DOCS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check="--check" in sys.argv))
