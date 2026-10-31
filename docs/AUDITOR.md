# Autonomous AI Model Auditor

The Model Auditor is an orchestration layer over existing TrustLens validation, training, evaluation, explanation, reliability, drift, and local security modules. It operates on the active Streamlit session dataset and, when its dataset fingerprint matches, the currently selected trained model. It does not automatically change or deploy a model.

## Evidence and execution

- Dataset profiling reuses `src.data_profile`. Datasets larger than 10,000 rows use a deterministic sample for estimated quality, correlation, target, and outlier observations; dimensions and observed data types remain exact. Without an expected schema, data-type validity is **Not evaluated**.
- Model metrics use the held-out test split, capped at a deterministic sample of 10,000 rows. Fit metrics are explicitly in-sample diagnostics. Stored split indices are checked for overlap, with a bounded feature-duplicate screen.
- Model comparison is opt-in and reuses `compare_models` on at most 10,000 deterministic rows. It selects using validation rows and does not replace the active model. Optional three-fold cross-validation clones the active pipeline, uses only fit rows, and caps the sample at 5,000 rows.
- Explainability reuses permutation importance, SHAP, LIME, and counterfactual routines. Budgets are fixed at 2,000 permutation rows, one SHAP/LIME observation, 300 LIME samples, and 20 counterfactual candidates. Explanation stability is **Not evaluated** because the current application has no validated stability test.
- Reliability checks reuse calibration, conformal interval, OOD, and drift functions. Split-derived checks use at most 10,000 deterministic rows. Drift is **Not evaluated** unless the user supplies both reference and current CSVs.
- Security checks reuse Milestone 3 local scanners. Prompt and output checks require supplied samples; dataset scanning requires explicit selected columns. Built-in jailbreak fixtures test scanner patterns only and are not target-model tests.

Each check reports **Passed**, **Finding**, **Failed**, **Not applicable**, or **Not evaluated**. A failed stage does not discard successful results. Scanned raw prompts and model outputs are not copied into audit results; source excerpts are omitted. Reports contain aggregate statistics and sanitized findings, not dataset rows or class labels.

## Risk method

The risk engine assigns points per unique finding: informational 0, low 10, medium 25, high 50, and critical 75. Category totals are capped at 100 and combined using these weights, renormalized across evaluated categories: dataset 20%, model 30%, explainability 15%, reliability 20%, and security 15%.

The overall score is withheld as **Insufficient Evidence** unless dataset and model checks, at least three categories, and at least six checks were evaluated. Failed, skipped, not-applicable, and not-evaluated checks do not count as passes. Duplicate findings are removed before scoring. Levels are Low (<15), Moderate (<35), High (<60), and Critical (60–100). This deterministic ordinal prioritization is not a probability of failure and never certifies safety, fairness, trustworthiness, or production readiness.

## Reports and limitations

JSON and CSV reports are generated in memory. They include assessment metadata, executed/skipped/failed checks, findings, category risk, scoring methodology, and finding-linked remediation guidance. They exclude dataset rows, raw prompts, raw outputs, and unmasked PII; CSV cells receive formula-injection protection. Human review is required. Recommendations suggest review actions and do not claim a problem has been fixed.

The audit is not a fairness assessment, target-LLM penetration test, causal analysis, or production deployment approval. Heuristic thresholds can produce false positives or miss issues. Results depend on the active dataset, split, model, selected checks, and sample limits.

## Local use

Start the app with `streamlit run app.py`, load data in Data Explorer, and train a model in Model Lab if model checks are needed. Open **Model Auditor**, choose the target and optional scopes, then run the audit. Candidate benchmarking and cross-validation explicitly refit model clones; both are off by default. Provide both optional drift CSVs to enable drift checks.
