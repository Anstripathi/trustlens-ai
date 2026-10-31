# TrustLens AI

TrustLens AI is a Streamlit platform for tabular model evaluation, explainability, and reliability analysis. It supports classification and regression while keeping datasets in the app session. It does not certify models as trustworthy, fair, causal, or production-ready.

## Capabilities

- **Data Explorer**: strict CSV validation, profiling, missingness, duplicates, correlations, and target/leakage hints.
- **Model Lab**: Logistic/Linear Regression, Decision Tree, and Random Forest with fit/calibration/test separation, validation model comparison, performance timings, complexity indicators, and bounded cross-validated hyperparameter search.
- **Explainability**: global and local permutation SHAP, LIME local approximations, bounded counterfactual scenarios, and retained permutation-importance/baseline-substitution methods.
- **Reliability Monitor**: binary and multiclass calibration, reliability diagrams, Brier/ECE, raw probability distributions, entropy-based prediction ambiguity, conformal regression intervals, Isolation Forest OOD flags, and dataset drift.
- **Student Marks Demo**: synthetic educational dataset and held-out model metrics; no interactive marks prediction.

## Verification

The Milestone 2 local suite completed with **144 passed, 4 warnings**. `pip check` found no broken requirements. Versions and limitations are recorded in [docs/TESTING.md](./docs/TESTING.md). SHAP/LIME/counterfactual/optimization methods and their assumptions are documented in [docs/METHODOLOGY.md](./docs/METHODOLOGY.md).

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

The app uses local Python packages only. It does not transmit datasets to external APIs and does not require Docker, a paid API, or an external database. SHAP and LIME add dependency weight; LIME's package declares scikit-image even though this project uses tabular explainability only.

## Deployment

The repository is [Anstripathi/trustlens-ai](https://github.com/Anstripathi/trustlens-ai). After committing and pushing the verified changes, deploy `app.py` from the `main` branch through Streamlit Community Cloud using Python 3.11 and the repository's `requirements.txt`. This workspace's Milestone 2 changes have not been pushed or deployed.

## Project layout

- `app.py`: executive overview.
- `pages/`: five Streamlit modules.
- `src/`: validation, training, evaluation, explanation, optimization, calibration, uncertainty, and UI-state logic.
- `tests/`: seeded unit and Streamlit AppTest coverage.
- `docs/`: architecture, methodology, security, and testing notes.

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for module boundaries and [docs/SECURITY.md](./docs/SECURITY.md) for upload protections and known limits.
