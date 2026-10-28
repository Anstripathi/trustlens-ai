# Artifacts

The app does not persist uploads or trained models. This folder is only a target for optional,
app-generated JSON reports via `src/artifact_manager.save_json_report` (not used by the UI).
Streamlit Community Cloud storage is ephemeral. Serialized models (pickle/joblib) are never loaded.
