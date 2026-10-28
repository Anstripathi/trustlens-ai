import matplotlib.pyplot as plt
import streamlit as st

from src import config
from src.data_profile import correlation_matrix, profile
from src.data_validation import ValidationError, load_csv_bytes
from src.artifact_manager import dataset_fingerprint, reject_model_artifact
from src.datasets import load_adult, make_demo_classification
from src.data_validation import validate_loaded
from src.ui import set_dataset, show_error

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("📊 Data Explorer")

up = st.file_uploader(f"Upload a CSV (max {config.MAX_UPLOAD_MB} MB)", type=None)
c1, c2 = st.columns(2)
if c1.button("Load synthetic demo dataset"):
    lr = validate_loaded(make_demo_classification().astype(str).replace("nan", ""))
    set_dataset(lr.clean, lr.raw, "synthetic demo (not real data)", lr.notes)
if c2.button("Download UCI Adult (needs internet)"):
    with st.spinner("Downloading..."):
        lr, msg = load_adult()
    if lr is None:
        st.error(msg)
    else:
        set_dataset(lr.clean, lr.raw, "UCI Adult (adult.data, 32,561 rows)", lr.notes)
        st.success(msg)
if up is not None:
    try:
        reject_model_artifact(up.name)
        lr = load_csv_bytes(up.getvalue(), up.name)
        if st.session_state.get("df_fp") != dataset_fingerprint(lr.clean):
            set_dataset(lr.clean, lr.raw, up.name, lr.notes)
    except ValidationError as e:
        show_error(e)

if "df" not in st.session_state:
    st.info("Upload a CSV or load a demo dataset to begin.")
    st.stop()

df = st.session_state["df"]
st.subheader(st.session_state["dataset_name"])
for n in st.session_state.get("load_notes", []):
    st.caption("• " + n)

target = st.selectbox("Target column (optional)", ["(none)"] + list(df.columns))
target = None if target == "(none)" else target
p = profile(df, target)

a, b, c, d = st.columns(4)
a.metric("Rows", f"{p['n_rows']:,}")
b.metric("Columns", p["n_cols"])
c.metric("Duplicate rows", p["duplicate_rows"])
d.metric("Cells missing", f"{int(p['missing']['missing'].sum()):,}")

if p["constant_columns"]:
    st.warning(f"Constant/empty columns: {p['constant_columns']}")
if p["identifier_columns"]:
    st.warning(f"Possible identifier columns (exclude from modelling): {p['identifier_columns']}")
for r in p["leakage_risks"]:
    st.error("Possible leakage: " + r)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Missing & types", "Numeric", "Categorical", "Correlation / target"])
with tab1:
    st.dataframe(df.head(50), use_container_width=True)
with tab2:
    st.dataframe(p["missing"], use_container_width=True)
with tab3:
    st.dataframe(p["numeric_stats"], use_container_width=True)
with tab4:
    st.dataframe(p["categorical_summary"], use_container_width=True)
with tab5:
    corr = correlation_matrix(df)
    if corr is None:
        st.caption("Need at least two varying numeric columns for a correlation plot.")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
        fig.colorbar(im); st.pyplot(fig); plt.close(fig)
        st.caption("Pearson correlation measures linear association, not causation.")
    if p["class_distribution"] is not None:
        st.write("Target distribution")
        st.dataframe(p["class_distribution"])
