# Experiment tracking and model versions

Model Lab performs model comparisons on the calibration split and bounded randomized search with cross-validation confined to fit rows. The held-out test split is reserved for final reporting. Fit duration, prediction duration and model complexity are reported only where measured.

Both randomized and deterministic budget-sampled grid search are available. The current estimator wrappers do not expose a native early-stopping interface, so no early-stopping behavior is claimed.

Advanced Model Intelligence stores reproducible configuration metadata, dataset fingerprints and aggregate metrics in a bounded local JSON registry. Uploaded data rows and fitted estimator objects are never written. A model version is a recipe, not a deployable serialized model; recreate it with the matching dataset in Model Lab and verify the result before using it. A changed fingerprint excludes a version from the current comparison.

The registry is local and has no authentication or cross-device synchronization. On ephemeral hosting, local records may disappear. Fingerprints are dataset identifiers, not anonymization; users should treat registry files as local project data. Timing is machine- and workload-dependent, and is not a production capacity guarantee.
