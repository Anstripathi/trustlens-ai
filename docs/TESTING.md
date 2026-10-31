# Testing

Run the full suite with `python -m pytest -v`. Tests use seeded synthetic data and do not require network access.

**Milestone 2 local verification**: `python -m pytest -v` completed with **144 passed, 4 warnings** on Python 3.11.5, Streamlit 1.66.1, pandas 3.1.0, NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1, matplotlib 3.11.2, SHAP 0.49.1, and LIME 0.2.0.1. `python -m pip check` reported no broken requirements. Three warnings come from SHAP's matplotlib colormap deprecations; one is the existing scikit-learn class-missing metric warning.

Streamlit AppTest covered the homepage and all five existing pages. Model Lab, Explainability, and Reliability Monitor were also rendered with a trained current model. Engine tests cover classification/regression, numerical/categorical inputs, explanation bounds and reproducibility, counterfactual constraints, search budgets, test-split preservation, and multiclass calibration.

These checks do not establish behavior on every browser/OS, on large real-world datasets, or on Streamlit Community Cloud. GitHub Actions and a cloud deployment of this milestone have not been run.

## Local smoke checklist

- Home loads and empty states appear without data/model.
- Load demo data and train classification and regression models.
- Run model comparison and bounded optimization; explicitly choose whether to adopt the optimized model.
- Explain a held-out row with SHAP and LIME, compare contribution directions, and search a constrained counterfactual.
- Check calibration/reliability, OOD flags, conformal intervals, and dataset drift.
- Confirm the Student Marks Demo contains no interactive marks prediction.
- Upload invalid CSV input and confirm a safe error appears.
