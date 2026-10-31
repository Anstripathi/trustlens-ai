import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.student_demo import PROVENANCE, fit_student_model
from src.theme import render_page_header, render_sidebar

st.set_page_config(page_title="Student Marks Demo", page_icon=":material/school:", layout="wide")
render_sidebar()
render_page_header("Student Marks Demo", "An educational example using a synthetic dataset.")
st.warning("Synthetic educational dataset - not real student records. " + PROVENANCE)

sm = fit_student_model()
st.info("This page is a static example of regression on synthetic data. It does not predict marks for user-entered values.")
st.caption("Study time alone cannot explain student outcomes; the fitted relationship is illustrative only.")

fig, ax = plt.subplots()
ax.scatter(sm.train.study_hours, sm.train.marks, label="train", alpha=0.6)
ax.scatter(sm.test.study_hours, sm.test.marks, label="test (held-out)", alpha=0.8, marker="x")
xs = np.linspace(*sm.hours_range, 50)
ax.plot(xs, sm.model.predict(pd.DataFrame({"study_hours": xs})), "k-", label="fitted line")
ax.set_xlabel("Study hours"); ax.set_ylabel("Marks"); ax.legend()
st.pyplot(fig); plt.close(fig)

m = sm.metrics
c = st.columns(3)
c[0].metric("MAE", f"{m['mae']:.2f}"); c[1].metric("RMSE", f"{m['rmse']:.2f}"); c[2].metric("R²", f"{m['r2']:.3f}")
st.caption(f"Evaluation: random 75/25 split ({m['n_train']} train / {m['n_test']} test rows); metrics are on the test rows.")
st.caption("The fitted line summarizes an association in this synthetic dataset, not a guarantee about any individual student.")
