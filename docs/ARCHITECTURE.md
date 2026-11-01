# Architecture

TrustLens AI is a modular Streamlit application. `pages/*.py` render the interface and call reusable logic in `src/`; `src/ui.py` owns Streamlit session-state helpers.

Data flow: uploaded bytes pass through strict CSV validation into separate raw-string and cleaned typed frames. The model pipeline splits data into fit, calibration/validation, and test sets, then fits preprocessing only on fit rows. Evaluation, explanations, optimization, calibration, and uncertainty functions consume the fitted pipeline and its split data. Dataset drift compares two cleaned frames independently.

Core modules:

- `src/train.py`: task validation, supported estimators, leakage-aware splits, pipelines, and validation-split model comparison.
- `src/optimization.py`: bounded randomized search with preprocessing refit inside cross-validation folds on fit data only.
- `src/experiments.py`: bounded metadata-only experiment registry and dataset-fingerprinted training recipes; no estimator serialization.
- `src/governance.py`: deterministic one-pass audit routing and session-local human decision queue.
- `src/llm_eval.py`: offline trace evaluation with lexical/citation proxies and existing local pattern scanners.
- `src/governance_records.py`: versioned local policy snapshots and a hash-chained metadata audit trail.
- `src/explain.py`: permutation importance and baseline-substitution attribution.
- `src/shap_explain.py`: bounded model-agnostic SHAP over original columns, plus comparable model-family importance.
- `src/lime_explain.py`: deterministic, bounded tabular LIME using the complete fitted pipeline.
- `src/counterfactual.py`: bounded search over empirical feature alternatives and user constraints.
- `src/calibration.py`: binary and multiclass calibration, reliability tables, Brier scores, and ECE.
- `src/uncertainty.py`: split-conformal regression intervals and sampled Isolation Forest OOD flags.
- `src/security/`: independent local security scanners, bounded assessment runner, risk scoring, and redacted report serialization.
- `pages/6_AI_Security.py`: session-only UI for selected prompt/output checks and optional selected-column PII scanning of the loaded dataset.
- `src/auditor/`: structured assessment contracts, bounded adapters over existing modules, orchestration, risk scoring, recommendations, and report serialization.
- `pages/7_Model_Auditor.py`: current-session audit configuration, results, optional drift uploads, and in-memory report downloads.
- `src/firewall/`: local input/output enforcement policies, metadata-only approval/event records, and optional bounded in-memory rate limiting over existing security scanners.
- `pages/8_AI_Firewall.py`: session-scoped policy controls, literal custom rules, enforcement test console, approval review, event history, and privacy-conscious exports.
- `src/monitoring/`: bounded metrics and drift/OOD analysis for supplied prediction batches, threshold alerts, acknowledged history, report serialization, and session-scoped aggregate SQLite storage.
- `pages/9_Model_Monitoring.py`: validated reference/current prediction CSV workflow. It does not represent live production telemetry and does not write source rows to storage.
- `pages/10_Advanced_Model_Intelligence.py`: experiment history, measured prediction timing, and safe metadata version recipes.
- `pages/11_Governance_Center.py`: bounded fixed-stage assessment and human review queue.
- `pages/12_LLM_RAG_Evaluation.py`: offline evaluation of user-supplied traces without a model endpoint.
- `pages/13_Governance_Control_Center.py`: policy versioning and local audit event review.

Session state holds the active dataset, its content fingerprint, the trained result, and session-local analysis outputs. Loading a new dataset clears dependent results. Explanation and optimization results are tied to the current model identity and dataset fingerprint. `require_model` refuses a model trained on a different dataset. Nothing is cached across sessions and uploaded data is not written to disk.

The fit/calibration/test split keeps calibration and OOD threshold selection away from test labels. Model comparison and hyperparameter selection use validation data; the test split remains for final reporting. Expensive operations have bounded sample, row, fold, or candidate counts.

The security package does not call or evaluate an LLM. It uses deterministic pattern checks and explicitly labelled heuristics, keeps full user text out of assessment objects and exports, and masks bounded evidence excerpts. Jailbreak fixtures are stored in the active session only to show the local scanner's detection results; they do not test a target model. Dataset scanning is limited to selected columns, a cell cap, and a total character budget. Reports are generated in memory and do not write to the artifacts directory.

The Model Auditor coordinates the existing profiler, active `TrainResult`, explanation functions, calibration/uncertainty/drift functions, and local security scanners. It does not reimplement those algorithms. Metrics and samples are bounded; comparison and fit-only cross-validation refit candidate/cloned pipelines only after explicit opt-in and do not alter the active session model. Audit results are invalidated when the dataset or model is replaced. Reports serialize selected metadata and sanitized findings, omit raw prompts/outputs and dataset rows, and apply CSV formula protection. Full auditor methodology and thresholds are documented in `docs/AUDITOR.md`.
