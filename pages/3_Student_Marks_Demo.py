import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.student_demo import PROVENANCE, fit_student_model

st.set_page_config(page_title="Student Marks Demo", layout="wide")
st.title("🎓 Student Marks Demo")
st.warning("Synthetic educational dataset - not real student records. " + PROVENANCE)

sm = fit_student_model()
hours = st.number_input("Study hours", 0.0, 24.0, 5.0, 0.5)
pred, warn = sm.predict(float(hours))
st.metric("Predicted marks", f"{pred:.1f}")
if warn:
    st.warning(warn)
st.caption("Marks depend on many factors besides study time; this single-feature model is illustrative only.")

fig, ax = plt.subplots()
ax.scatter(sm.train.study_hours, sm.train.marks, label="train", alpha=0.6)
ax.scatter(sm.test.study_hours, sm.test.marks, label="test (held-out)", alpha=0.8, marker="x")
xs = np.linspace(*sm.hours_range, 50)
ax.plot(xs, sm.model.predict(pd.DataFrame({"study_hours": xs})), "k-", label="fitted line")
ax.scatter([hours], [pred], color="red", zorder=5, label="your input")
ax.set_xlabel("Study hours"); ax.set_ylabel("Marks"); ax.legend()
st.pyplot(fig); plt.close(fig)

m = sm.metrics
c = st.columns(3)
c[0].metric("MAE", f"{m['mae']:.2f}"); c[1].metric("RMSE", f"{m['rmse']:.2f}"); c[2].metric("R²", f"{m['r2']:.3f}")
st.caption(f"Evaluation: random 75/25 split ({m['n_train']} train / {m['n_test']} test rows); metrics are on the test rows.")
st.latex(rf"\text{{marks}} \approx {m['intercept']:.2f} + {m['slope']:.2f}\times\text{{hours}}")
st.caption("Each extra study hour changes predicted marks by the slope - an association in this data, not a guarantee.")
