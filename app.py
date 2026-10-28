"""TrustLens AI - home page."""
import streamlit as st

st.set_page_config(page_title="TrustLens AI", page_icon="🔍", layout="wide")
st.title("🔍 TrustLens AI")
st.caption("An explainable and reliable machine learning platform")

st.markdown("""
TrustLens AI helps you ask five questions about a tabular dataset and model:

1. **Can I trust the data quality?** → *Data Explorer*
2. **How well does the model perform on unseen data?** → *Model Lab*
3. **Why did the model make this prediction?** → *Explainability*
4. **How reliable is the prediction?** → *Reliability Monitor* (confidence, calibration, intervals, OOD)
5. **Has the incoming data changed?** → *Reliability Monitor → Drift*

There is also an educational **Student Marks Demo** (synthetic data).
""")
st.warning("TrustLens AI does not certify a model as trustworthy, unbiased, causal or production-ready. "
           "Metrics describe behaviour on your data only. Uploaded data is processed in memory for your "
           "session and is not saved by the app.")
if "df" in st.session_state:
    st.success(f"Loaded dataset: **{st.session_state['dataset_name']}** "
               f"({st.session_state['df'].shape[0]:,} rows × {st.session_state['df'].shape[1]} columns)")
