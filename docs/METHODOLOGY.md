# Methodology
**Cleaning**: strip whitespace; `''`, `?`, `na`, `n/a`, `nan`, `null` (case-insens.) → missing; a column is numeric only if every non-missing value parses. Rows with missing target are dropped at training.
**Splits**: test (`test_size`) first, then validation/calibration from the remainder; stratified for classification when every class has ≥2 rows and fits the split, else random with a warning. Seeds are user-set.
**Preprocessing**: median / constant-`missing` imputation, one-hot (max 30 categories, unseen → infrequent/zeros), scaling only for linear models; all fitted on the fit split only (tested).
**Metrics**: reported on the test split. Binary F1/precision/recall use `classes[1]` as positive; multiclass uses macro. ROC-AUC/Brier are `n/a` if the evaluation set lacks a class. Baselines (majority accuracy, mean-prediction RMSE) are shown.
**Model selection**: "Compare all models" scores the validation split with default hyperparameters; the test set is not used.
**Explanations**: global = permutation importance (test split, ≤2000 rows); local = baseline substitution vs. training median/mode. Neither is SHAP; neither is causal; correlated features distort both.
**Calibration** (binary only): sigmoid (Platt on logit) or isotonic fit on the validation split; needs ≥50 rows and ≥10 per class. Evaluated on test with Brier and ECE (10 bins).
**Regression intervals**: split conformal, q = ⌈(n+1)(1−α)⌉-th smallest |residual| on the validation split; withheld if that rank exceeds n. Marginal coverage under exchangeability; constant width.
**OOD**: Isolation Forest on preprocessed fit data; threshold = flag-rate quantile of validation-split scores. A flag is not an error proof; no flag is not a reliability proof.
**Drift**: numeric KS; categorical chi-square (categories with <10 total pooled); min 30 non-missing per side; p-values adjusted by BH (default) or Bonferroni; "drift" requires adjusted p<α **and** KS ≥ 0.1 / Cramér's V ≥ 0.1 (thresholds are conventions, adjustable in code). Drift does not imply accuracy loss.
