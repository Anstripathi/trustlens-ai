# TrustLens AI — Explainable and Reliable ML Platform

A Streamlit app for tabular data: inspect data quality, train classification/regression models with
leakage-safe evaluation, explain predictions, assess reliability, and compare datasets for drift.
It does **not** certify any model as trustworthy, unbiased, causal or production-ready.

## Status (honest summary)
| Area | State |
|---|---|
| Core logic in `src/` (validation, profiling, training, metrics, explanations, calibration, conformal intervals, OOD, drift, student demo) | Implemented; **100 test checks pass** locally (see docs/TESTING.md) |
| Streamlit pages | Implemented; logic exercised against a stub of `streamlit`; **never run in a real Streamlit server by the author of this build** — run it yourself and report problems |
| SHAP | **Not implemented.** Permutation importance (global) + baseline-substitution attribution (local) are used |
| UCI Adult download | Implemented, **untested** (no network in build env) |
| GitHub Actions workflow | Written, **has not run** |
| Deployment to Streamlit Community Cloud | **Not done** — needs your GitHub account (steps below). No live URL exists |
| Screenshots / benchmark numbers | None included |

## Features
- **Data Explorer**: strict CSV parsing, size/type limits, missing tokens (`?`, empty…), duplicates, constant & identifier columns, leakage heuristics, correlation, class distribution.
- **Model Lab**: Logistic/Linear Regression, Decision Tree, Random Forest; fit / validation / test split (stratified when possible); preprocessing fit on the fit split only; metrics on the test split; model comparison on the validation split only.
- **Explainability**: permutation importance on original columns; local attribution by baseline substitution; causal caveat everywhere.
- **Reliability Monitor**: confidence warnings, binary calibration (sigmoid/isotonic fit on the validation split), reliability diagram, Brier/ECE, split-conformal regression intervals, Isolation-Forest OOD flag, and **data drift** (KS / chi-square, BH or Bonferroni, effect-size thresholds).
- **Student Marks Demo**: linear regression on a *synthetic* dataset with held-out metrics and extrapolation warnings.

## Setup (Windows / PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest
streamlit run app.py
```
No Docker, no paid services, no API keys.

## Deploy to Streamlit Community Cloud (manual steps for you)
1. Create a GitHub repo and push this folder (`git init; git add .; git commit -m "TrustLens AI"; git remote add origin <url>; git push -u origin main`).
2. Go to https://share.streamlit.io → *New app* → pick the repo, branch `main`, main file `app.py`; choose Python 3.11 in *Advanced settings*. No secrets needed.
3. Read the deploy logs; open the URL; run the smoke list in docs/TESTING.md. Only then record the URL here.

## Layout
`app.py`, `pages/` (UI only) · `src/` (all logic, no UI imports except `src/ui.py`) · `tests/` · `docs/`.
See docs/ARCHITECTURE.md, METHODOLOGY.md, SECURITY.md, TESTING.md.

## Limitations
Small data → noisy metrics; calibration/intervals are withheld when data is too small; attributions are not causal; drift ≠ accuracy loss; no fairness metrics; uploads are processed in memory only; free-tier hosting has memory/CPU limits (unmeasured).
