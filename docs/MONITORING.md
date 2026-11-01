# Model Monitoring

The Model Monitoring page consumes prediction batches explicitly supplied as CSV. TrustLens has no deployed prediction API and does not manufacture production events. The page is at [pages/9_Model_Monitoring.py](../pages/9_Model_Monitoring.py); bounded analysis is in `src/monitoring/` and reuses the existing drift and OOD modules.

## Input format and available signals

Upload a current batch and, optionally, a reference batch. Select the predicted value column and optionally true labels, confidence in `[0, 1]`, measured latency in milliseconds, and feature columns.

- Every run records the number of rows and usable predictions, plus an ordinal-label prediction distribution for classification or aggregate quantiles for regression.
- When true labels are supplied, classification accuracy, macro precision/recall/F1 and top-label ECE are computed. Regression MAE, RMSE and R² are computed. The reference comparison is only available when the corresponding labeled columns exist in both batches.
- Confidence summaries and latency percentiles are computed only when their columns are supplied and validate. Latency is not inferred from file-processing time.
- Feature missingness and numeric summary statistics are reported for selected columns. Reference/current feature drift reuses the existing corrected significance/effect-size checks with deterministic samples capped at 10,000 rows per batch.
- Optional OOD comparison uses the active current model and its fit/calibration splits. It is unavailable when the model task or feature schema does not match.
- Configurable thresholds create warning/critical alerts. Each alert has a recommendation and a session history acknowledgement with a reviewer label.

No raw uploaded row or original prediction label is placed in a snapshot. Class labels are represented as `Class 1`, `Class 2`, etc. Feature names, aggregate statistics, alerts, batch fingerprints, model-version fingerprints, and reviewer labels may still be sensitive; review exports before sharing. CSV formulas are escaped.

## Local persistence and privacy

Monitoring snapshots are saved to `artifacts/monitoring.sqlite3` using Python's standard-library SQLite. The database contains aggregate snapshots only, never uploaded CSV records. Records are scoped by a random UUID; the application stores only its SHA-256 scope hash. It retains at most 100 snapshots per key and removes records older than 30 days when a new snapshot is added.

The page offers a recovery-key download. Treat that key as a bearer secret: anyone with it and access to this installation can read the associated aggregate history. It is not an account credential. The DB is local/ephemeral on Community Cloud and does not provide guaranteed durable storage. If SQLite is unavailable, the current session continues with in-memory history and reports the persistence limitation.

## Limitations

Monitoring is batch/manual, not continuous service telemetry. It cannot compute model performance without labels, infer real request counts, measure serving latency unless supplied by the source, or establish prediction correctness from drift/OOD signals. Thresholds are operational heuristics, not regulatory limits. No model is automatically replaced or redeployed. See [TESTING.md](./TESTING.md) for the current local verification result.
