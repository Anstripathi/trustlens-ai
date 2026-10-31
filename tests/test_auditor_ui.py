from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_auditor_page_runs_without_data_and_shows_empty_state():
    page = Path(__file__).resolve().parents[1] / "pages/7_Model_Auditor.py"
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception
    assert any("No dataset loaded" in element.value for element in at.markdown)
    assert len(at.form_submit_button) == 1


def test_auditor_form_runs_and_exports_a_partial_evidence_report():
    import numpy as np
    import pandas as pd
    from src.artifact_manager import dataset_fingerprint
    from src.train import TrainConfig, train_model

    rng = np.random.default_rng(10)
    frame = pd.DataFrame({"x": rng.normal(size=160), "z": rng.normal(size=160)})
    frame["target"] = np.where(frame.x > 0, "positive", "negative")
    fingerprint = dataset_fingerprint(frame)
    model = train_model(frame, TrainConfig("target", "classification", "Decision Tree"))
    page = Path(__file__).resolve().parents[1] / "pages/7_Model_Auditor.py"
    at = AppTest.from_file(page, default_timeout=120)
    at.session_state["df"] = frame
    at.session_state["df_fp"] = fingerprint
    at.session_state["dataset_name"] = "synthetic-test"
    at.session_state["model_data_fp"] = fingerprint
    at.session_state["result"] = model
    at.run()
    at.checkbox[0].uncheck()
    at.checkbox[1].uncheck()
    at.form_submit_button[0].click().run()
    assert not at.exception
    assessment = at.session_state["model_auditor_result"]
    assert assessment.risk.risk_level == "Insufficient Evidence"
    assert any(check.check_id == "dataset.overview" for check in assessment.checks)
    assert any(check.check_id == "model.performance" for check in assessment.checks)
    assert len(at.get("download_button")) == 2
