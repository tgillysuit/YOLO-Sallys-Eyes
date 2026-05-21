"""Tests for the start_processing transition (QUEUED -> INITIALIZING)."""
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
    LoadModelSideEffect,
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
def queued_context():
    """Fresh Context with status in QUEUED state."""
    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    job_request = JobRequest(
        contract_version=CONTRACT_VERSION,
        video_id="test_video_001",
        device="cpu",
        sample_stride=2,
        confidence_threshold=0.25,
        iou_threshold=0.7,
    )
    status = JobStatus(
        job_id="job_abc123",
        state=State.QUEUED,
        created_at=now,
        updated_at=now,
    )
    return Context(
        job_id="job_abc123",
        job_request=job_request,
        status=status,
    )


class TestStartProcessing:
    """Behavioral tests for the start_processing transition."""

    def test_happy_path_transitions_to_initializing(self, queued_context):
        """QUEUED + START_PROCESSING -> INITIALIZING."""
        new_state, side_effects = fire(
            State.QUEUED, Event.START_PROCESSING, queued_context
        )

        assert new_state == State.INITIALIZING
        assert queued_context.status.state == State.INITIALIZING
        assert len(side_effects) == 1
        assert isinstance(side_effects[0], LoadModelSideEffect)

    @pytest.mark.parametrize(
        "wrong_state",
        [
            State.CREATED,
            State.INITIALIZING,
            State.TRACKING,
            State.COMPUTING,
            State.RENDERING,
            State.COMPLETE,
            State.FAILED,
        ],
    )
    def test_invalid_state_raises(self, queued_context, wrong_state):
        """start_processing must reject every non-QUEUED state."""
        queued_context.status.state = wrong_state

        with pytest.raises(InvalidTransitionError) as exc_info:
            fire(State.QUEUED, Event.START_PROCESSING, queued_context)

        assert wrong_state.value in str(exc_info.value)

    def test_docstring_contains_required_fields(self):
        """start_processing docstring must contain all required labels."""
        from salamander.transitions import start_processing

        doc = start_processing.__doc__ or ""
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
        assert not missing, f"start_processing docstring missing labels: {missing}"

    def test_emits_load_model_side_effect_with_requested_device(self, queued_context):
        """The model loading side effect carries the requested device preference."""
        _, side_effects = fire(State.QUEUED, Event.START_PROCESSING, queued_context)

        model_effect = side_effects[0]
        assert isinstance(model_effect, LoadModelSideEffect)
        assert model_effect.model_path == "models/yolo11n.pt"
        assert model_effect.device == "cpu"

    def test_updated_at_is_advanced(self, queued_context):
        """start_processing advances updated_at."""
        initial_updated_at = queued_context.status.updated_at

        fire(State.QUEUED, Event.START_PROCESSING, queued_context)

        assert queued_context.status.updated_at > initial_updated_at
