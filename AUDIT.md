# Audit of the original notebook

The original is `ML2_FINAL_PATHFINDERCOMPLETE_2.ipynb` (first-year ML course, PSL).
Cell numbers below refer to it. Each item gives what was wrong, how I found it, and its
status in this repository. Where a number is reproduced here, it comes from
`experiments/03_audit_original.py`. Numbers quoted from the notebook are marked "as
printed".

## 1. The dataset is synthetic, and the notebook does not say so

- **What.** Five features are uniformly distributed, and students who dropped out
  still have a final grade. Findings such as "linear beats trees" and the named
  dropout archetypes were presented as facts about students.
- **How found.** Kolmogorov–Smirnov tests against the uniform distribution
  (`experiments/00_is_it_synthetic.py`), plus the grades of dropouts.
- **Status.** Disclosed at the top of the README. No claim about students is made.

## 2. The grade is censored at 100 and was modelled as if it were not

- **What.** About one training grade in five is exactly 100 (README, synthetic-data
  table). Least squares on a censored
  target flattens every slope. That produced predictions above 100 (patched by
  clipping, cell 65) and a systematic over-prediction of weak students. The notebook
  attributed the latter to "a structural limitation of the current feature set"
  (cell 89) after three unrelated fixes failed.
- **How found.** Counting target values equal to the maximum, then comparing a
  censored-likelihood model.
- **Status.** Fixed: `TobitRegressor`. Cross-validation favours it and shrinks the
  over-prediction of students below 65 (README, Task 1, with per-repeat differences).
  On the test set alone the RMSE difference is not significant.

## 3. Oversampling before cross-validation

- **What.** SMOTE was fitted on the whole training set (cell 115). The threshold was
  then chosen by cross-validation on the resampled data (cell 118). Synthetic points
  built from a validation student's neighbours were in the training folds, so the
  validation precision was far above what new students delivered.
- **How found.** Comparing the validation precision with the test precision printed
  by the notebook itself, then reproducing both procedures.
- **Status.** Reproduced below. This repository does not resample.

<!-- BEGIN:audit_smote -->
| procedure | threshold | validation precision / recall | test precision / recall |
|---|---|---|---|
| original notebook, as printed | 0.258 | 0.828 / 0.984 | 0.540 / 0.947 |
| SMOTE before CV (reproduced) | 0.225 | 0.813 / 0.985 | 0.514 / 0.947 |
| SMOTE inside each fold | 0.251 | 0.454 / 0.964 | 0.545 / 0.947 |

CV AUC reported on the resampled set: 0.956 (notebook: 0.9577); honest CV AUC: 0.941; test AUC: 0.924 (notebook: 0.9166).
<!-- END:audit_smote -->

The model was not the problem: with SMOTE inside each fold the test precision is about
the same. The *estimate* was, and the notebook reported that estimate as the expected
performance.

## 4. Probabilities calibrated on resampled data

- **What.** Isotonic calibration was fitted on the 50/50 resampled set (cell 133), so
  the calibrated probabilities reflect a 50% base rate, not 15%. The notebook called
  them calibrated.
- **How found.** Mean predicted probability against the observed rate.
- **Status.** Fixed: plain logistic regression, whose probabilities are calibrated to
  the training base rate.

<!-- BEGIN:audit_calibration -->
| model | mean predicted P | calibration error | Brier |
|---|---|---|---|
| isotonic calibration fitted on SMOTE data (original) | 0.260 | 0.084 | 0.099 |
| plain logistic regression (this repository) | 0.159 | 0.057 | 0.087 |

Observed dropout rate: 0.190 in the test set, 0.150 in the training set.
<!-- END:audit_calibration -->

- **Open.** The test cohort's dropout rate is higher than the training cohort's. No
  model trained on one cohort can know another's base rate; this is disclosed and not
  corrected.

## 5. The test set was used to choose models

- **What.** The classifier leaderboard (cell 129) is computed on the held-out test set,
  and the champion is the model with the best *test* recall. For the grade, Lasso was
  evaluated on the test set (cell 90), then a stacked model was built and evaluated on
  it again (cell 101) and declared final. The text says the test set was "sealed
  throughout all modelling decisions" (cell 92).
- **How found.** Reading which data each leaderboard used.
- **Status.** Fixed. `experiments/01_select.py` chooses on the training set with rules
  written before fitting, and freezes the choice in `results/selection.json`.
  `experiments/02_final_test.py` is the only script that loads the test set to evaluate
  the choice, and it refuses to run without that file.

## 6. A CV score on resampled data reported as the CV score

- **What.** "5-fold CV AUC 0.9577 ± 0.0055" (cell 135, as printed) was measured on the
  SMOTE-resampled set, where synthetic minority points are easy to separate.
- **Status.** Fixed: every CV score here is measured on real students only (the
  honest figure is in the table under item 3).

## 7. An arbitrary decision threshold

- **What.** The threshold maximised F2, which weights recall four times as much as
  precision. Nobody chose that weighting.
- **Status.** Replaced by two stated policies: capacity (top k) and cost (the Bayes
  threshold for a stated cost ratio, with its sensitivity). See README, Task 2.

## 8. Causal claims from a correlational model

- **What.** DiCE counterfactuals (chapter 4) were presented as "what they should
  change", and the recommended interventions as if they would work.
- **Status.** Replaced in v0.3. What-if plans are back (`counterfactual.py`), solved
  exactly for the logistic model: sparse (at most two levers), diverse, bounded to the
  observed range, and checked against the model by a test. Every card and every file
  written by `pathfinder rank` says they describe the model, not the student.

## 9. Clusters that are not there

- **What.** K-means on SHAP values of the flagged students, with silhouette 0.155
  (cell 176, as printed): the clusters are hardly separated. The clusters were named as
  archetypes ("The Disconnected") with intervention plans. The accompanying text cites
  silhouette 0.225 and "three profiles" for k = 4 (cell 175).
- **Status.** Replaced in v0.3. `segments.kmeans_check` re-measures the question on
  this repository's model (silhouette and bootstrap stability for k = 2 to 6, shown in
  the README), and finds no structure either. Students are routed to an office by
  their first reason instead: a rule an office can read and dispute.

## 10. Hand-typed results

- **What.** The final regression leaderboard (cell 88) is a typed list and disagrees
  with computed values elsewhere (Linear Regression MAE 5.19 there, 4.68 in cells 67
  and 75). The pipeline diagram (cell 180) names LightGBM as the calibrated model,
  while the selected model was logistic regression.
- **Status.** Fixed: `experiments/render_readme.py` renders every table from
  `results/*.json`, and a test fails if a document is out of date.

## 11. Stacking on a proxy label

- **What.** The final grade model appended a predicted P(grade < 65) as a feature (cell
  99), for a cross-validated R² of 0.814 against 0.812 without it (cell 104, as
  printed).
- **Status.** Dropped. The gain is within noise, and the extra stage is one more place
  for leakage.

## Found while rebuilding

### 12. Categorical columns missed under pandas 3

- **What.** `oulad.snapshot` selected columns to one-hot encode with
  `dtype == object`. Under pandas 3, text columns have a dedicated string dtype, so
  module codes reached the model as text.
- **How found.** `tests/test_early_warning.py` failed with "could not convert string
  to float".
- **Status.** Fixed (`is_numeric_dtype`), with an assertion that every feature is
  numeric.

### 13. Routing by summed contributions favoured the largest service

- **What.** The first routing rule summed each service's feature contributions. The
  academic service owns five features, the financial one owns one, so the sum favoured
  the academic service by construction.
- **How found.** Looking at the share of students routed to each service before
  drawing the chart.
- **Status.** Fixed: route by the single largest contribution (the first reason
  shown). Academic support still receives most referrals, because assignment completion
  has the largest coefficient; that is a property of the model and is stated in the
  README.

### 14. A feature change found after the test evaluation

- **What.** `experiments/06_feature_questions.py` finds that dropping the dormitory
  block improves both models in every CV repeat, by a small margin.
- **How found.** The feature-question experiment, written after the single test
  evaluation.
- **Status.** Not applied. Applying it and scoring the same test set again would repeat
  item 5. It is recorded for the next version, which will need new data to evaluate.

### 15. Minimum-norm what-ifs are exact but impractical

- **What.** The first what-if solver returned the smallest change measured in
  standard deviations. That change moves every lever a little ("+0.15 tutoring
  sessions").
- **How found.** Reading the plans for the first three example students.
- **Status.** Fixed: `sparse_plans` solves every subset of at most two levers exactly,
  rounds integer levers to whole units, and keeps up to three plans, each using a lever
  the others do not. The minimum-norm solver remains as `joint_plan`.
