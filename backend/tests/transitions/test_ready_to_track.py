"""Tests for the ready_to_track transition (INITIALIZING -> TRACKING)."""
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
def initializing_context():
    """Fresh Context with status in INITIALIZING state."""
    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    job_request = JobRequest(
        contract_version=CONTRACT_VERSION,
        video_id="test_video_001",
    )
    status = JobStatus(
        job_id="job_abc123",
        state=State.INITIALIZING,
        created_at=now,
        updated_at=now,
    )
    return Context(
        job_id="job_abc123",
        job_request=job_request,
        status=status,
        model=object(),
        frame_reader=object(),
    )


class TestReadyToTrack:
    """Behavioral tests for the ready_to_track transition."""

    def test_happy_path_transitions_to_tracking(self, initializing_context):
        """INITIALIZING + READY_TO_TRACK -> TRACKING."""
        new_state, side_effects = fire(
            State.INITIALIZING, Event.READY_TO_TRACK, initializing_context
        )

        assert new_state == State.TRACKING
        assert initializing_context.status.state == State.TRACKING
        assert side_effects == []

    @pytest.mark.parametrize(
        "wrong_state",
        [
            State.CREATED,
            State.QUEUED,
            State.TRACKING,
            State.COMPUTING,
            State.RENDERING,
            State.COMPLETE,
            State.FAILED,
        ],
    )
    def test_invalid_state_raises(self, initializing_context, wrong_state):
        """ready_to_track must reject every non-INITIALIZING state."""
        initializing_context.status.state = wrong_state

        with pytest.raises(InvalidTransitionError) as exc_info:
            fire(State.INITIALIZING, Event.READY_TO_TRACK, initializing_context)

        assert wrong_state.value in str(exc_info.value)

    def test_docstring_contains_required_fields(self):
        """ready_to_track docstring must contain all required labels."""
        from salamander.transitions import ready_to_track

        doc = ready_to_track.__doc__ or ""
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
        assert not missing, f"ready_to_track docstring missing labels: {missing}"

    def test_no_side_effects_emitted(self, initializing_context):
        """ready_to_track is a pure state advance."""
        _, side_effects = fire(
            State.INITIALIZING, Event.READY_TO_TRACK, initializing_context
        )

        assert side_effects == []

    def test_updated_at_is_advanced(self, initializing_context):
        """ready_to_track advances updated_at."""
        initial_updated_at = initializing_context.status.updated_at

        fire(State.INITIALIZING, Event.READY_TO_TRACK, initializing_context)

        assert initializing_context.status.updated_at > initial_updated_at
