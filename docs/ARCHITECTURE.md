# Architecture

TrustLens AI is a modular Streamlit application. `pages/*.py` render the interface and call reusable logic in `src/`; `src/ui.py` owns Streamlit session-state helpers.

Data flow: uploaded bytes pass through strict CSV validation into separate raw-string and cleaned typed frames. The model pipeline splits data into fit, calibration/validation, and test sets, then fits preprocessing only on fit rows. Evaluation, explanations, optimization, calibration, and uncertainty functions consume the fitted pipeline and its split data. Dataset drift compares two cleaned frames independently. The Student Marks Demo uses synthetic data and does not accept an interactive marks prediction.

Core modules:

- `src/train.py`: task validation, supported estimators, leakage-aware splits, pipelines, and validation-split model comparison.
- `src/optimization.py`: bounded randomized search with preprocessing refit inside cross-validation folds on fit data only.
- `src/explain.py`: permutation importance and baseline-substitution attribution.
- `src/shap_explain.py`: bounded model-agnostic SHAP over original columns, plus comparable model-family importance.
- `src/lime_explain.py`: deterministic, bounded tabular LIME using the complete fitted pipeline.
- `src/counterfactual.py`: bounded search over empirical feature alternatives and user constraints.
- `src/calibration.py`: binary and multiclass calibration, reliability tables, Brier scores, and ECE.
- `src/uncertainty.py`: split-conformal regression intervals and sampled Isolation Forest OOD flags.

Session state holds the active dataset, its content fingerprint, the trained result, and session-local analysis outputs. Loading a new dataset clears dependent results. Explanation and optimization results are tied to the current model identity and dataset fingerprint. `require_model` refuses a model trained on a different dataset. Nothing is cached across sessions and uploaded data is not written to disk.

The fit/calibration/test split keeps calibration and OOD threshold selection away from test labels. Model comparison and hyperparameter selection use validation data; the test split remains for final reporting. Expensive operations have bounded sample, row, fold, or candidate counts.
