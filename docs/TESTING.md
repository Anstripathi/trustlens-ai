# Testing

Run the full suite with `python -m pytest -v`. Tests use seeded synthetic data and do not require network access.

**Milestone 3 local verification**: `.venv\\Scripts\\python.exe -m pytest -v` completed with **201 passed, 4 warnings** on Python 3.11.5, Streamlit 1.66.1, pandas 3.1.0, NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1, matplotlib 3.11.2, SHAP 0.49.1, and LIME 0.2.0.1. `.venv\\Scripts\\python.exe -m pip check` reported no broken requirements. Three warnings come from SHAP's matplotlib colormap deprecations; one is the existing scikit-learn class-missing metric warning.

**Milestone 4 local verification**: `.venv\\Scripts\\python.exe -m pytest -v` completed with **238 passed, 4 warnings** on Python 3.11.5. The warning sources are unchanged: three SHAP matplotlib colormap pending deprecations and one existing scikit-learn class-missing metric warning. `.venv\\Scripts\\python.exe -m pip check` reported no broken requirements. Streamlit was started locally and the homepage plus all seven registered pages (including Model Auditor and AI Security) were opened in the browser without application exceptions. AppTest covers empty-state and populated flows, all audit stages, opt-in bounded validation/CV/benchmarking, report privacy, and active-model preservation. Community Cloud deployment was not run.

**Milestone 5 local verification**: the full suite completed with **252 passed, 4 warnings** on Python 3.11.5. Warning sources remain the existing three SHAP matplotlib pending deprecations and one scikit-learn missing-class metric warning. `pip check` reported no broken requirements. Firewall tests cover policy decisions, literal-rule matching, output redaction, approval metadata, fail-closed behavior, session rate limits, report privacy, CSV formula escaping, and Streamlit console rendering. The public app was deployed before this local-only change; this M5 work has not been pushed to Community Cloud.

**Milestones 6–10 local verification**: `python -m pytest -q` completed with **283 passed, 4 warnings** on Python 3.11.5. `python -m pip check` reported no broken requirements. Warnings remain the three SHAP matplotlib colormap pending deprecations and the existing scikit-learn missing-class metric warning. `git diff --check` reported no whitespace errors (only Git line-ending notices). A local Streamlit server started on port 8504; the homepage and all 13 registered module routes rendered without Python exceptions. The public deployment was not updated.

Streamlit AppTest covered the homepage and all existing pages, including the AI Security empty state and an assessment submission with export controls. Model Lab, Explainability, and Reliability Monitor were also rendered with a trained current model. Engine tests cover classification/regression, numerical/categorical inputs, explanation bounds and reproducibility, counterfactual constraints, search budgets, test-split preservation, multiclass calibration, security-scanner limits, masking, score determinism, report exports, and failure isolation.

These checks do not establish behavior on every browser/OS, on large real-world datasets, or on Streamlit Community Cloud. GitHub Actions and a cloud deployment of this milestone have not been run.

## Local smoke checklist

- Home loads and empty states appear without data/model.
- Load demo data and train classification and regression models.
- Run model comparison and bounded optimization; explicitly choose whether to adopt the optimized model.
- Explain a held-out row with SHAP and LIME, compare contribution directions, and search a constrained counterfactual.
- Check calibration/reliability, OOD flags, conformal intervals, and dataset drift.
- Run selected AI Security checks on local text; verify masked findings, explicit skipped/failed checks, and redacted JSON/CSV downloads.
- Optionally scan selected loaded-dataset columns; verify the cell and character budgets.
- Exercise AI Firewall input/output directions, add literal rules with block/redact/approval-required actions, and inspect session-only decisions and metadata reports.
- Upload invalid CSV input and confirm a safe error appears.
