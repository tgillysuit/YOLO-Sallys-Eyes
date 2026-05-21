"""Tests for the frame_processed transition (TRACKING self-loop)."""
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
def tracking_context():
    """Fresh Context with status in TRACKING state."""
    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    job_request = JobRequest(
        contract_version=CONTRACT_VERSION,
        video_id="test_video_001",
    )
    status = JobStatus(
        job_id="job_abc123",
        state=State.TRACKING,
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


class TestFrameProcessed:
    """Behavioral tests for the frame_processed transition."""

    def test_happy_path_stays_in_tracking(self, tracking_context):
        """TRACKING + FRAME_PROCESSED remains in TRACKING."""
        frame_records = tracking_context.frame_records

        new_state, side_effects = fire(
            State.TRACKING, Event.FRAME_PROCESSED, tracking_context
        )

        assert new_state == State.TRACKING
        assert tracking_context.status.state == State.TRACKING
        assert tracking_context.frame_records is frame_records
        assert side_effects == []

    @pytest.mark.parametrize(
        "wrong_state",
        [
            State.CREATED,
            State.QUEUED,
            State.INITIALIZING,
            State.COMPUTING,
            State.RENDERING,
            State.COMPLETE,
            State.FAILED,
        ],
    )
    def test_invalid_state_raises(self, tracking_context, wrong_state):
        """frame_processed must reject every non-TRACKING state."""
        tracking_context.status.state = wrong_state

        with pytest.raises(InvalidTransitionError) as exc_info:
            fire(State.TRACKING, Event.FRAME_PROCESSED, tracking_context)

        assert wrong_state.value in str(exc_info.value)

    def test_docstring_contains_required_fields(self):
        """frame_processed docstring must contain all required labels."""
        from salamander.transitions import frame_processed

        doc = frame_processed.__doc__ or ""
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
        assert not missing, f"frame_processed docstring missing labels: {missing}"

    def test_no_side_effects_emitted(self, tracking_context):
        """frame_processed emits no new side effects."""
        _, side_effects = fire(
            State.TRACKING, Event.FRAME_PROCESSED, tracking_context
        )

        assert side_effects == []

    def test_updated_at_is_advanced(self, tracking_context):
        """frame_processed advances updated_at."""
        initial_updated_at = tracking_context.status.updated_at

        fire(State.TRACKING, Event.FRAME_PROCESSED, tracking_context)

        assert tracking_context.status.updated_at > initial_updated_at
