# Data

Nothing is bundled. Datasets are never committed (see `.gitignore`).

## UCI Adult Income (primary benchmark)
- Source: https://archive.ics.uci.edu/dataset/2/adult (Becker & Kohavi, 1996). Check the page for the current licence/terms; at the time of writing it is listed as CC BY 4.0 - **verify yourself**.
- The in-app button downloads **`adult.data` only (32,561 rows)**, not the full 48,842 instances (`adult.test` is not used). Column names are added by `src/datasets.py`.
- Target: `income` (`>50K` / `<=50K`). Missing values are written as `?` and are mapped to missing by the cleaning step. No rows are dropped except those with a missing target.
- Not verified in the build environment (no network): the download URL and the resulting schema. If the download fails the app shows a message; upload the CSV manually instead.
- Limitations: 1994 census data, not suitable for modern decisions. Contains sex, race and native-country attributes and proxies; models can reproduce historical bias. Do **not** use for employment or eligibility decisions. Check fairness per group before drawing conclusions - TrustLens AI does not compute fairness metrics.
- `fnlwgt` is a census sampling weight and `education`/`education-num` are redundant encodings of each other; consider excluding them.

## Synthetic data
- "Demo classification" (`src/datasets.py`) and "Student marks" (`src/student_demo.py`) are generated with fixed seeds. They are **synthetic**, labelled as such in the UI, and not real records.
