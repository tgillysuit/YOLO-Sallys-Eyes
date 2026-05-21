"""Tests for the submit_job transition (CREATED -> QUEUED)."""
from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest

from salamander.contracts import (
    CONTRACT_VERSION,
    Event,
    JobRequest,
    JobStatus,
    State,
)
from salamander.state import (
    TRANSITION_TABLE,
    Context,
    InvalidTransitionError,
    WriteManifestSideEffect,
    fire,
)


@pytest.fixture(autouse=True)
def reset_transition_table():
    """Register transitions for this test, then restore the previous table."""
    snapshot = dict(TRANSITION_TABLE)
    from salamander import transitions

    importlib.reload(transitions)
    yield
    TRANSITION_TABLE.clear()
    TRANSITION_TABLE.update(snapshot)


@pytest.fixture
def created_context():
    """Fresh Context with status in CREATED state, ready for submit_job."""
    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    job_request = JobRequest(
        contract_version=CONTRACT_VERSION,
        video_id="test_video_001",
        device="auto",
        sample_stride=2,
        confidence_threshold=0.25,
        iou_threshold=0.7,
    )
    status = JobStatus(
        job_id="",
        state=State.CREATED,
        created_at=now,
        updated_at=now,
    )
    return Context(
        job_id="job_abc123",
        job_request=job_request,
        status=status,
    )


class TestSubmitJob:
    """Behavioral tests for the submit_job transition."""

    def test_happy_path_transitions_to_queued(self, created_context):
        """CREATED + SUBMIT_JOB -> QUEUED with all state mutations applied."""
        new_state, side_effects = fire(
            State.CREATED, Event.SUBMIT_JOB, created_context
        )

        assert new_state == State.QUEUED
        assert created_context.status.state == State.QUEUED
        assert created_context.status.job_id == "job_abc123"
        assert isinstance(created_context.status.created_at, datetime)
        assert created_context.status.created_at.tzinfo is not None
        assert len(side_effects) == 1
        assert isinstance(side_effects[0], WriteManifestSideEffect)
        assert "job_abc123" in side_effects[0].output_path

    @pytest.mark.parametrize(
        "wrong_state",
        [
            State.QUEUED,
            State.INITIALIZING,
            State.TRACKING,
            State.COMPUTING,
            State.RENDERING,
            State.COMPLETE,
            State.FAILED,
        ],
    )
    def test_invalid_state_raises(self, created_context, wrong_state):
        """submit_job must raise InvalidTransitionError from any non-CREATED state."""
        created_context.status.state = wrong_state

        with pytest.raises(InvalidTransitionError) as exc_info:
            fire(State.CREATED, Event.SUBMIT_JOB, created_context)

        assert wrong_state.value in str(exc_info.value)

    def test_docstring_contains_required_fields(self):
        """submit_job docstring must contain all required contract field labels."""
        from salamander.transitions import submit_job

        doc = submit_job.__doc__ or ""
        required_labels = [
            "From:",
            "To:",
            "Trigger:",
            "Reads:",
            "Writes:",
            "May call:",
            "Must NOT call:",
            "Postcondition:",
            "Errors:",
        ]
        missing = [label for label in required_labels if label not in doc]
        assert not missing, f"submit_job docstring missing labels: {missing}"

    def test_updated_at_is_advanced(self, created_context):
        """submit_job advances updated_at from the initial fixture timestamp."""
        initial_updated_at = created_context.status.updated_at

        fire(State.CREATED, Event.SUBMIT_JOB, created_context)

        assert created_context.status.updated_at > initial_updated_at

    def test_side_effect_output_path_includes_job_id(self, created_context):
        """The WriteManifestSideEffect must reference the job's ID for storage routing."""
        _, side_effects = fire(State.CREATED, Event.SUBMIT_JOB, created_context)

        manifest_effect = side_effects[0]
        assert isinstance(manifest_effect, WriteManifestSideEffect)
        assert "job_abc123" in manifest_effect.output_path
        assert manifest_effect.output_path.endswith("manifest.json")
