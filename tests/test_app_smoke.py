"""Streamlit smoke tests. Skipped automatically if streamlit is not installed."""
import pytest
from pathlib import Path

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["app.py", "pages/1_Data_Explorer.py", "pages/3_Student_Marks_Demo.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page):
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / page, default_timeout=60).run()
    assert not at.exception


def test_gated_pages_show_info_without_data():
    for page in ["pages/2_Model_Lab.py", "pages/4_Explainability.py"]:
        at = AppTest.from_file(Path(__file__).resolve().parents[1] / page, default_timeout=60).run()
        assert not at.exception and len(at.info) >= 1


def test_demo_dataset_then_model_lab_trains():
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "pages/1_Data_Explorer.py", default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    ml = AppTest.from_file(Path(__file__).resolve().parents[1] / "pages/2_Model_Lab.py", default_timeout=120)
    ml.session_state["df"] = at.session_state["df"]
    ml.session_state["df_fp"] = at.session_state["df_fp"]
    ml.run()
    ml.selectbox[0].select("churned").run()
    ml.button[0].click().run()
    assert not ml.exception and "result" in ml.session_state


@pytest.mark.parametrize("page", ["pages/2_Model_Lab.py", "pages/4_Explainability.py", "pages/5_Reliability_Monitor.py"])
def test_model_pages_render_with_current_model(page):
    from helpers import clf_df
    from src.artifact_manager import dataset_fingerprint
    from src.train import TrainConfig, train_model

    frame = clf_df(n=500)
    result = train_model(frame, TrainConfig("y", "classification", "Logistic Regression"))
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / page, default_timeout=120)
    at.session_state["df"] = frame
    at.session_state["df_fp"] = dataset_fingerprint(frame)
    at.session_state["model_data_fp"] = at.session_state["df_fp"]
    at.session_state["result"] = result
    at.run()
    assert not at.exception
