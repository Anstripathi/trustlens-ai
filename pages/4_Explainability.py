import matplotlib.pyplot as plt
import streamlit as st

from src.explain import CAUSAL_NOTE, global_importance, local_explanation
from src.ui import require_dataset, require_model, show_error

st.set_page_config(page_title="Explainability", layout="wide")
st.title("💡 Explainability Center")
require_dataset()
res = require_model()
st.info(CAUSAL_NOTE + " SHAP is not used in this build; model-agnostic methods are used instead.")

st.subheader("Global: permutation importance")
if st.button("Compute global importance"):
    try:
        with st.spinner("Permuting features on the test split..."):
            g = global_importance(res)
        if not g["available"]:
            st.warning(g["reason"])
        else:
            st.session_state["global_imp"] = g
    except Exception as e:  # noqa: BLE001
        show_error(e)
g = st.session_state.get("global_imp")
if g is not None:
    t = g["table"].head(15)[::-1]
    fig, ax = plt.subplots()
    ax.barh(t.feature, t.importance_mean, xerr=t.importance_std)
    ax.set_xlabel(f"Score drop ({g['scoring']})")
    st.pyplot(fig); plt.close(fig)
    st.caption(g["method"])

st.subheader("Local: explain one test-set prediction")
i = st.number_input("Test row position", 0, len(res.X_test) - 1, 0)
row = res.X_test.iloc[[int(i)]]
try:
    ex = local_explanation(res, row)
    st.write(f"Model prediction: **{ex['prediction']}** (actual: {res.y_test.iloc[int(i)]})")
    st.write(f"Explained quantity: {ex['output_explained']} = {ex['model_output']:.3f}")
    t = ex["contributions"].head(10)[::-1]
    fig, ax = plt.subplots()
    ax.barh(t.feature, t.contribution, color=["tab:green" if v > 0 else "tab:red" for v in t.contribution])
    st.pyplot(fig); plt.close(fig)
    st.dataframe(ex["contributions"])
    st.caption(ex["method"])
except Exception as e:  # noqa: BLE001
    show_error(e)
