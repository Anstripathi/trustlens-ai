# Methodology

**Cleaning**: strip whitespace; `''`, `?`, `na`, `n/a`, `nan`, and `null` (case-insensitive) become missing. A column is numeric only when every non-missing value parses. Rows with a missing target are dropped at training.

**Splits**: the test split is created first, then calibration/validation from the remainder. Classification splits are stratified when class counts and split sizes permit; otherwise training records a warning and uses a random split. Seeds are user-set.

**Preprocessing**: median/constant-`missing` imputation, one-hot encoding (up to 30 categories), and scaling for linear models. Preprocessing is fit on fit rows only, including during cross-validation.

**Evaluation**: metrics are reported on the untouched test split. Classification includes accuracy, precision, recall, F1, ROC-AUC when defined, and confusion matrices. Regression includes MAE, MSE, RMSE, and R-squared. Majority-accuracy and mean-prediction RMSE baselines are shown.

**Model comparison and optimization**: model-family comparison reports validation metrics, fit/prediction time, and tree-node or coefficient counts where measurable. Bounded randomized search uses fit rows only and refits preprocessing inside each CV fold (at most 20 parameter settings and 5 folds). Baseline and optimized models are compared on validation data. Applying an optimized pipeline requires an explicit user action; the test set is never used for tuning or selection.

**SHAP**: permutation SHAP wraps the complete fitted pipeline and explains original input columns. Categorical values are encoded only for masking and decoded before prediction. Background is capped at 10 fit rows; explanations at 25 rows and 25,000 masked model rows, with three forward/reverse permutations. Global importance is mean absolute contribution. Local waterfall values explain one selected scalar output (regression prediction or a chosen class probability). Permutation SHAP is approximate, dependent features can affect attribution, and contributions are not causal.

**LIME**: tabular LIME uses a reproducible seed, at most 300 fit rows for feature distributions, and at most 5,000 perturbations. Categorical values are perturbed as categories; generated inputs pass through the fitted preprocessing pipeline. Local R-squared measures surrogate fit around the selected observation, not overall model quality. LIME is a local approximation and not a causal effect.

**Counterfactual scenarios**: bounded search uses numeric quantiles and observed categorical values. User numeric ranges and immutable columns are enforced; at most 200 candidates and 3 changed features are considered. A result is a model-input scenario, not a causal intervention or real-world guarantee. A no-candidate result is reported explicitly.

**Calibration**: sigmoid (Platt on logits) or isotonic calibration is fit on the calibration split and evaluated on test rows. Binary reliability/ECE track a selected class probability. Multiclass calibration uses one-vs-rest models with row renormalization; reliability/ECE use top-label confidence versus accuracy, and Brier is the multiclass squared-probability score. Calibration requires at least 50 calibration rows and 10 rows per class. ECE is noisy when test bins are small.

**Prediction ambiguity**: normalized entropy summarizes the spread of raw class probabilities. It is not epistemic/model uncertainty or data uncertainty. Raw probabilities are not guarantees of correctness.

**Regression intervals**: split conformal intervals use the calibration-split residual quantile and are withheld when the finite-sample rank is unavailable. Marginal coverage relies on exchangeability; intervals have constant width and the guarantee does not apply under drift or out-of-distribution inputs.

**OOD**: Isolation Forest uses deterministic samples of at most 10,000 fit rows and 10,000 calibration rows. Its threshold is the requested flag-rate quantile of calibration scores. A flag does not prove the prediction is wrong; no flag does not prove reliability.

**Drift**: numeric KS; categorical chi-square with rare categories pooled; at least 30 non-missing rows per side; p-values adjusted by Benjamini-Hochberg or Bonferroni. Drift requires both adjusted significance and a practical effect-size threshold. Drift does not imply accuracy loss.
