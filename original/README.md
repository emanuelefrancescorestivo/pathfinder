# The original course notebook

`ML2_FINAL_PATHFINDERCOMPLETE_2.ipynb` is the first-year Machine Learning course project
(PSL) that this repository rebuilds, written by Emanuele Restivo, Marcos Almodovar and
Yani Boukrif. It is kept **unmodified**, outputs included, so
that every claim in [AUDIT.md](../AUDIT.md) can be checked against it.

    SHA-256 d24f5e032cc9d2e987856072eb75978d639eac45e29eaff1b4a40ff2b6999fc5

- **Cell numbers** in AUDIT.md count the notebook's cells from 0, in file order
  (markdown and code alike), not Jupyter's execution counters.
- **Numbers marked "as printed"** in AUDIT.md are the outputs stored in this file.
- **To re-run it** you need the course CSVs (not included, see
  [data/README.md](../data/README.md)) next to the notebook, and the libraries it
  imports (among them `lightgbm`, `xgboost`, `imbalanced-learn`, `shap`, `dice-ml` and
  `umap-learn`), none of which the rebuilt package needs.
- It is **not** linted, type-checked or tested by CI: it is a historical record, and
  fixing it in place would erase the evidence.
