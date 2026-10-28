# Testing
Run: `python -m pytest -v`. Fixtures are small, seeded, synthetic; no network needed (the two network tests expect failure).

**What was actually run in the build environment**: 100 test checks via a minimal local pytest stand-in (pytest itself was not installable offline), Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1: all passed. Pages were additionally executed against a stub `streamlit` module for classification and regression flows (no exceptions).
**Not run**: real pytest, Python 3.11, pandas 2.x, real Streamlit (`tests/test_app_smoke.py` uses `streamlit.testing.AppTest` and auto-skips without Streamlit), GitHub Actions, deployment.
**Bug found by tests during the build**: pandas silently absorbed extra CSV fields into an index → replaced with a strict csv-module parser. Identifier heuristic false positive on integer-valued incomes → tightened.

**What tests establish**: exact metric values on hand-computed fixtures; split disjointness/reproducibility; preprocessing statistics come from the fit split; conformal coverage near nominal on 3,000 rows; calibrator reduces ECE on synthetic over-confidence; drift detects known shifts and stays quiet on same-distribution data; security rejects traversal, model files, ragged/binary/oversized CSVs. **They do not establish** UI correctness in a browser, security completeness, or model quality on real data.

**Live smoke checklist (after deploy)**: home loads; load demo data; upload bad file → friendly error; train classification (target `churned`) and regression (`income`); explain; calibration/OOD tabs; drift with two CSVs; Student demo; no tracebacks visible.
