"""
State machine for the salamander tracking job lifecycle.

Transitions are registered via @register_transition(state, event) decorators
elsewhere in the codebase. This module owns the table, the dispatcher, and
the runtime context passed to every transition function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .contracts import FrameRecord, JobRequest, JobStatus, ProcessingMetadata, TrackSummary


# ---------------------------------------------------------------------------
# State and Event enumerations
# ---------------------------------------------------------------------------

class State(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    COMPUTING = "COMPUTING"
    RENDERING = "RENDERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Event(str, Enum):
    SUBMIT_JOB = "SUBMIT_JOB"
    START_PROCESSING = "START_PROCESSING"
    READY_TO_TRACK = "READY_TO_TRACK"
    FRAME_PROCESSED = "FRAME_PROCESSED"
    TRACKING_COMPLETE = "TRACKING_COMPLETE"
    METRICS_COMPLETE = "METRICS_COMPLETE"
    ARTIFACTS_WRITTEN = "ARTIFACTS_WRITTEN"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Transition registry
# ---------------------------------------------------------------------------

# Maps (current_state, event) -> transition function.
# Transition functions have signature: (Context) -> tuple[State, list[SideEffect]]
# Populated via @register_transition decorators defined elsewhere.
TRANSITION_TABLE: dict[tuple[State, Event], Callable] = {}


def register_transition(state: State, event: Event) -> Callable:
    """Decorator that registers a transition function for (state, event)."""
    def decorator(fn: Callable) -> Callable:
        TRANSITION_TABLE[(state, event)] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Mutable runtime bag passed through every transition.

    Transitions read ctx.status to validate they were called in the right
    state, then mutate ctx.status to record progress, timestamps, errors, etc.
    The dispatcher returns (new_state, side_effects); callers read ctx.status
    for the full updated JobStatus after firing.
    """
    job_id: str
    job_request: JobRequest
    status: JobStatus                    # required — callers set starting state explicitly
    model: Any = None                    # loaded YOLO model handle
    frame_reader: Any = None             # open cv2.VideoCapture
    frame_records: list[FrameRecord] = field(default_factory=list)
    track_summaries: list[TrackSummary] = field(default_factory=list)
    processing_metadata: ProcessingMetadata | None = None


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------

@dataclass
class SideEffect:
    """Base class for all side effects returned by transition functions."""


@dataclass
class LoadModelSideEffect(SideEffect):
    model_path: str
    device: str


@dataclass
class OpenFrameReaderSideEffect(SideEffect):
    video_path: str


@dataclass
class WriteManifestSideEffect(SideEffect):
    output_path: str


@dataclass
class WriteFramesJsonSideEffect(SideEffect):
    output_path: str
    frame_records: list  # list[FrameRecord], untyped to avoid import cycles


@dataclass
class WriteSummaryJsonSideEffect(SideEffect):
    output_path: str
    job_metrics: Any  # JobMetrics


@dataclass
class WriteCsvSideEffect(SideEffect):
    output_path: str
    frame_records: list


@dataclass
class RenderAnnotatedVideoSideEffect(SideEffect):
    input_video_path: str
    output_video_path: str
    frame_records: list


@dataclass
class FfmpegEncodeSideEffect(SideEffect):
    input_path: str
    output_path: str


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised when fire() is called with no registered transition."""


def fire(
    state: State,
    event: Event,
    ctx: Context,
) -> tuple[State, list[SideEffect]]:
    """Look up and invoke the registered transition for (state, event).

    Returns (new_state, side_effects). Raises InvalidTransitionError if no
    transition is registered.
    """
    key = (state, event)
    transition_fn = TRANSITION_TABLE.get(key)
    if transition_fn is None:
        raise InvalidTransitionError(
            f"No transition registered for ({state.value}, {event.value})"
        )
    return transition_fn(ctx)
