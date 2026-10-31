from src.auditor import (
    AuditCategory,
    AuditCheck,
    AuditRequest,
    CheckStatus,
    run_stages,
)


def test_audit_request_keeps_inputs_in_memory_without_serialization_contract():
    request = AuditRequest(
        dataset=None,
        dataset_name=None,
        dataset_fingerprint=None,
        prompt_sample="private prompt",
    )
    assert request.prompt_sample == "private prompt"
    assert request.run_security is False


def test_orchestrator_retains_successful_stage_results():
    check = AuditCheck("data.shape", AuditCategory.DATASET, CheckStatus.PASSED, "Shape available")
    result = run_stages(
        AuditRequest(None, None, None),
        {"dataset": (AuditCategory.DATASET, lambda request: [check])},
    )
    assert result.checks == (check,)
    assert result.failed_stages == ()


def test_orchestrator_isolates_stage_failure_and_keeps_success():
    success = AuditCheck("data.shape", AuditCategory.DATASET, CheckStatus.PASSED, "Shape available")

    def fail(_request):
        raise RuntimeError("sensitive raw row value")

    result = run_stages(
        AuditRequest(None, None, None),
        {
            "dataset": (AuditCategory.DATASET, lambda request: [success]),
            "model": (AuditCategory.MODEL, fail),
        },
    )
    assert result.checks[0] == success
    assert result.checks[1].status is CheckStatus.FAILED
    assert "sensitive raw row value" not in result.checks[1].summary
    assert result.failed_stages == ("model",)


def test_orchestrator_marks_invalid_stage_output_as_failed():
    result = run_stages(
        AuditRequest(None, None, None),
        {"bad": (AuditCategory.DATASET, lambda request: ["not a check"])},
    )
    assert result.checks[0].status is CheckStatus.FAILED
    assert result.failed_stages == ("bad",)


def test_assessments_get_distinct_ids_and_utc_timestamps():
    first = run_stages(AuditRequest(None, None, None), {})
    second = run_stages(AuditRequest(None, None, None), {})
    assert first.assessment_id != second.assessment_id
    assert first.created_at.utcoffset().total_seconds() == 0
