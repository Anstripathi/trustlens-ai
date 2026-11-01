# TrustLens AI

TrustLens AI is a Streamlit platform for tabular model evaluation, explainability, and reliability analysis. It supports classification and regression while keeping datasets in the app session. It does not certify models as trustworthy, fair, causal, or production-ready.

## Capabilities

- **Data Explorer**: strict CSV validation, profiling, missingness, duplicates, correlations, and target/leakage hints.
- **Model Lab**: Logistic/Linear Regression, Decision Tree, and Random Forest with fit/calibration/test separation, validation model comparison, performance timings, complexity indicators, and bounded cross-validated hyperparameter search.
- **Explainability**: global and local permutation SHAP, LIME local approximations, bounded counterfactual scenarios, and retained permutation-importance/baseline-substitution methods.
- **Reliability Monitor**: binary and multiclass calibration, reliability diagrams, Brier/ECE, raw probability distributions, entropy-based prediction ambiguity, conformal regression intervals, Isolation Forest OOD flags, and dataset drift.
- **AI Security**: local prompt-injection and jailbreak-pattern checks, masked PII/credential scanning, limited unsafe-output phrase rules, bounded selected-column dataset scanning, transparent heuristic risk prioritization, and redacted JSON/CSV reports. No target LLM is called.
- **Model Auditor**: coordinates evidence-backed dataset, active-model, explanation, reliability, and optional local security checks; applies documented risk scoring and finding-linked recommendations; exports sanitized JSON/CSV reports.
- **AI Firewall**: reusable local input/output policy enforcement with allow, warn, block, redact, and human-review decisions; literal custom rules, bounded session rate limiting, metadata-only event history, and privacy-conscious reports.
- **Model Monitoring**: manually supplied prediction batches, reference/current feature drift, labeled performance and confidence/latency summaries, optional model OOD checks, threshold alerts, and local aggregate-only SQLite history.
- **Advanced Model Intelligence**: validation-safe experiment comparison, measured runtime indicators, dataset-fingerprinted metadata versions, and reproducible training recipes without executable model serialization.
- **Governance Center**: bounded local audit orchestration, policy-based review routing, and explicit human decision records.
- **LLM and RAG Evaluation**: offline trace metrics, lexical grounding and citation proxies, and local injection/PII checks without an LLM endpoint.
- **Governance Control Center**: versioned policy metadata and a verifiable local hash-chained audit log, with no authentication claim.

## Verification

The verified Milestone 3 baseline was **201 passed, 4 warnings**. Milestone 4 verification is recorded in [docs/TESTING.md](./docs/TESTING.md). Auditor scoring, bounds, limitations and report privacy are documented in [docs/AUDITOR.md](./docs/AUDITOR.md). Firewall policies and callable interfaces are described in [docs/FIREWALL.md](./docs/FIREWALL.md). Monitoring inputs, local aggregate retention and limitations are documented in [docs/MONITORING.md](./docs/MONITORING.md). SHAP/LIME/counterfactual/optimization assumptions are in [docs/METHODOLOGY.md](./docs/METHODOLOGY.md); security-screening assumptions are in [docs/SECURITY.md](./docs/SECURITY.md).

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

To deploy or redeploy, select `app.py` from the intended branch in Streamlit Community Cloud, use Python 3.11, and install dependencies from `requirements.txt`. The current local milestones have not been pushed or deployed. Exercise all registered pages after any deployment; the governance/monitoring stores use local ephemeral storage and reviewer labels are not authenticated identities.

## Project layout

- `app.py`: executive overview.
- `pages/`: Streamlit modules for data exploration, modeling, security, monitoring, evaluation, and governance.
- `src/`: validation, training, evaluation, explanation, optimization, calibration, uncertainty, security checks, firewall policy enforcement, monitoring, auditor orchestration, and UI-state logic.
- `tests/`: seeded unit and Streamlit AppTest coverage.
- `docs/`: architecture, methodology, security, and testing notes.

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for module boundaries, [docs/SECURITY.md](./docs/SECURITY.md) for upload protections, and [docs/AUDITOR.md](./docs/AUDITOR.md) for auditor methodology and limits.
