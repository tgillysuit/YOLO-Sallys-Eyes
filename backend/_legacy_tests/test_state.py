"""Tests for the state machine spine: Context, TRANSITION_TABLE, fire(), SideEffects."""
import dataclasses
from datetime import datetime, timezone

import pytest

from salamander.contracts import CONTRACT_VERSION, JobRequest, JobStatus
from salamander.state import (
    TRANSITION_TABLE,
    Context,
    Event,
    FfmpegEncodeSideEffect,
    InvalidTransitionError,
    RenderAnnotatedVideoSideEffect,
    State,
    WriteCsvSideEffect,
    WriteFramesJsonSideEffect,
    WriteSummaryJsonSideEffect,
    fire,
    register_transition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_transition_table():
    """Snapshot TRANSITION_TABLE before each test; restore after.

    Prevents test c and d (which register fake transitions) from polluting
    later tests.
    """
    snapshot = dict(TRANSITION_TABLE)
    yield
    TRANSITION_TABLE.clear()
    TRANSITION_TABLE.update(snapshot)


@pytest.fixture
def minimal_request() -> JobRequest:
    return JobRequest(
        contract_version=CONTRACT_VERSION,
        video_id="test-video",
    )


@pytest.fixture
def minimal_status() -> JobStatus:
    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    return JobStatus(
        job_id="job-test-001",
        state=State.CREATED.value,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def minimal_ctx(minimal_request, minimal_status) -> Context:
    return Context(
        job_id="job-test-001",
        job_request=minimal_request,
        status=minimal_status,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_context_requires_status(minimal_request):
    """Context must not accept construction without an explicit status."""
    with pytest.raises(TypeError):
        Context(job_id="j1", job_request=minimal_request)  # missing status


def test_fire_raises_when_no_transition_registered(minimal_ctx):
    """fire() must raise InvalidTransitionError for unregistered (state, event) pairs."""
    with pytest.raises(InvalidTransitionError) as exc_info:
        fire(State.CREATED, Event.SUBMIT_JOB, minimal_ctx)
    assert "CREATED" in str(exc_info.value)
    assert "SUBMIT_JOB" in str(exc_info.value)


def test_fire_invokes_registered_transition(minimal_ctx):
    """fire() must call the registered function and return its result."""
    invoked = []

    def fake_submit(ctx):
        invoked.append(True)
        return (State.QUEUED, [])

    register_transition(State.CREATED, Event.SUBMIT_JOB)(fake_submit)

    result = fire(State.CREATED, Event.SUBMIT_JOB, minimal_ctx)

    assert result == (State.QUEUED, [])
    assert invoked == [True], "fake transition was never called"


def test_register_transition_decorator_adds_entry():
    """@register_transition must insert the function into TRANSITION_TABLE."""
    def fake_fn(ctx):
        return (State.TRACKING, [])

    register_transition(State.QUEUED, Event.START_PROCESSING)(fake_fn)

    assert TRANSITION_TABLE[(State.QUEUED, Event.START_PROCESSING)] is fake_fn


def test_side_effect_classes_are_dataclasses():
    """Every new SideEffect subclass must be a dataclass and accept its fields."""
    se1 = WriteFramesJsonSideEffect(output_path="frames.json", frame_records=[])
    assert dataclasses.is_dataclass(se1)
    assert se1.output_path == "frames.json"
    assert se1.frame_records == []

    se2 = WriteSummaryJsonSideEffect(output_path="summary.json", job_metrics=None)
    assert dataclasses.is_dataclass(se2)
    assert se2.output_path == "summary.json"

    se3 = WriteCsvSideEffect(output_path="tracks.csv", frame_records=[])
    assert dataclasses.is_dataclass(se3)
    assert se3.output_path == "tracks.csv"

    se4 = RenderAnnotatedVideoSideEffect(
        input_video_path="input.mp4",
        output_video_path="output_annotated.mp4",
        frame_records=[],
    )
    assert dataclasses.is_dataclass(se4)
    assert se4.input_video_path == "input.mp4"
    assert se4.output_video_path == "output_annotated.mp4"

    se5 = FfmpegEncodeSideEffect(input_path="raw.mp4", output_path="final.mp4")
    assert dataclasses.is_dataclass(se5)
    assert se5.input_path == "raw.mp4"
    assert se5.output_path == "final.mp4"
