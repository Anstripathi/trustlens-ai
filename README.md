# TrustLens AI

TrustLens AI is a Streamlit platform for tabular model evaluation, explainability, and reliability analysis. It supports classification and regression while keeping datasets in the app session. It does not certify models as trustworthy, fair, causal, or production-ready.

## Capabilities

- **Data Explorer**: strict CSV validation, profiling, missingness, duplicates, correlations, and target/leakage hints.
- **Model Lab**: Logistic/Linear Regression, Decision Tree, and Random Forest with fit/calibration/test separation, validation model comparison, performance timings, complexity indicators, and bounded cross-validated hyperparameter search.
- **Explainability**: global and local permutation SHAP, LIME local approximations, bounded counterfactual scenarios, and retained permutation-importance/baseline-substitution methods.
- **Reliability Monitor**: binary and multiclass calibration, reliability diagrams, Brier/ECE, raw probability distributions, entropy-based prediction ambiguity, conformal regression intervals, Isolation Forest OOD flags, and dataset drift.
- **AI Security**: local prompt-injection and jailbreak-pattern checks, masked PII/credential scanning, limited unsafe-output phrase rules, bounded selected-column dataset scanning, transparent heuristic risk prioritization, and redacted JSON/CSV reports. No target LLM is called.
- **Model Auditor**: coordinates evidence-backed dataset, active-model, explanation, reliability, and optional local security checks; applies documented risk scoring and finding-linked recommendations; exports sanitized JSON/CSV reports.
- **Student Marks Demo**: synthetic educational dataset and held-out model metrics; no interactive marks prediction.

## Verification

The verified Milestone 3 baseline was **201 passed, 4 warnings**. Milestone 4 verification is recorded in [docs/TESTING.md](./docs/TESTING.md). Auditor scoring, bounds, limitations and report privacy are documented in [docs/AUDITOR.md](./docs/AUDITOR.md). SHAP/LIME/counterfactual/optimization assumptions are in [docs/METHODOLOGY.md](./docs/METHODOLOGY.md); security-screening assumptions are in [docs/SECURITY.md](./docs/SECURITY.md).

## Run locally (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -v
python -m pip check
streamlit run app.py
```

The app uses local Python packages only. It does not transmit datasets or security scan text to external APIs and does not require Docker, a paid API, or an external database. The security checks are deterministic rules and heuristics, not semantic moderation or a security guarantee. SHAP and LIME add dependency weight; LIME's package declares scikit-image even though this project uses tabular explainability only.

## Deployment

To deploy or redeploy, select `app.py` from the intended branch in Streamlit Community Cloud, use Python 3.11, and install dependencies from `requirements.txt`. A cloud deployment is not verified as part of local Milestone 4 work; exercise all seven pages after deployment.

## Project layout

- `app.py`: executive overview.
- `pages/`: Streamlit modules, including AI Security and Model Auditor.
- `src/`: validation, training, evaluation, explanation, optimization, calibration, uncertainty, security checks, auditor orchestration, and UI-state logic.
- `tests/`: seeded unit and Streamlit AppTest coverage.
- `docs/`: architecture, methodology, security, and testing notes.

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for module boundaries, [docs/SECURITY.md](./docs/SECURITY.md) for upload protections, and [docs/AUDITOR.md](./docs/AUDITOR.md) for auditor methodology and limits.
