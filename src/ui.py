"""Streamlit session-state helpers (UI glue only; no ML logic here)."""
from __future__ import annotations

import streamlit as st

from .artifact_manager import dataset_fingerprint, safe_error_message


def set_dataset(df, raw, name: str, notes: list[str]) -> None:
    """Store a dataset and drop any results derived from a previous dataset."""
    for k in ("result", "model_data_fp", "compare", "calibration", "ood", "conformal"):
        st.session_state.pop(k, None)
    st.session_state.update(df=df, raw=raw, dataset_name=name, load_notes=notes,
                            df_fp=dataset_fingerprint(df))


def require_dataset():
    if "df" not in st.session_state:
        st.info("No dataset loaded. Go to **Data Explorer** to upload a CSV or load a demo dataset.")
        st.stop()
    return st.session_state["df"]


def require_model():
    res = st.session_state.get("result")
    if res is None:
        st.info("No trained model yet. Train one in the **Model Lab** first.")
        st.stop()
    if st.session_state.get("model_data_fp") != st.session_state.get("df_fp"):
        st.warning("The loaded dataset changed since this model was trained. Retrain in the Model Lab.")
        st.stop()
    return res


def show_error(exc: Exception) -> None:
    st.error(safe_error_message(exc))
