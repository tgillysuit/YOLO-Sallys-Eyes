"""
Transition implementations for the salamander tracking state machine.

Each transition is a function taking Context and returning (new_state, side_effects).
Transitions read ctx.status.state to validate they were called in the right state,
then mutate ctx.status in place to record progress, timestamps, and state changes.

Module import has a side effect: each @register_transition decorator adds an entry
to state.TRANSITION_TABLE. Import this module to register all transitions.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .contracts import Event, State
from .state import (
    Context,
    InvalidTransitionError,
    LoadModelSideEffect,
    SideEffect,
    WriteManifestSideEffect,
    register_transition,
)


@register_transition(State.CREATED, Event.SUBMIT_JOB)
def submit_job(ctx: Context) -> tuple[State, list[SideEffect]]:
    """
    TRANSITION: submit_job
    From: CREATED -> To: QUEUED
    Trigger: API receives POST /api/jobs for an existing video
    Reads: ctx.status (must be in CREATED), ctx.job_id, ctx.job_request
    Writes: ctx.status.state = QUEUED, ctx.status.job_id, ctx.status.created_at, ctx.status.updated_at
    May call: storage.write_manifest() (via WriteManifestSideEffect)
    Must NOT call: vision.*, metrics.*, rendering.*
    Postcondition: ctx.status.state == QUEUED; ctx.status.job_id matches ctx.job_id; ctx.status.created_at and ctx.status.updated_at are set to UTC datetimes
    Errors: InvalidTransitionError if ctx.status.state != CREATED
    """
    if ctx.status.state != State.CREATED:
        raise InvalidTransitionError(
            f"submit_job requires CREATED state, got {ctx.status.state.value}"
        )

    now = datetime.now(timezone.utc)
    ctx.status.job_id = ctx.job_id
    ctx.status.created_at = now
    ctx.status.updated_at = now
    ctx.status.state = State.QUEUED

    side_effect = WriteManifestSideEffect(
        output_path=f"jobs/{ctx.job_id}/manifest.json"
    )

    return (State.QUEUED, [side_effect])


@register_transition(State.QUEUED, Event.START_PROCESSING)
def start_processing(ctx: Context) -> tuple[State, list[SideEffect]]:
    """
    TRANSITION: start_processing
    From: QUEUED -> To: INITIALIZING
    Trigger: Worker accepts a queued job and begins startup
    Reads: ctx.status (must be in QUEUED), ctx.job_request.device
    Writes: ctx.status.state = INITIALIZING, ctx.status.updated_at
    May call: vision.yolo_tracking.load_model() (via LoadModelSideEffect)
    Must NOT call: storage.*, metrics.*, rendering.*
    Postcondition: ctx.status.state == INITIALIZING; model loading is represented as a side effect
    Errors: InvalidTransitionError if ctx.status.state != QUEUED
    """
    if ctx.status.state != State.QUEUED:
        raise InvalidTransitionError(
            f"start_processing requires QUEUED state, got {ctx.status.state.value}"
        )

    ctx.status.state = State.INITIALIZING
    ctx.status.updated_at = datetime.now(timezone.utc)

    side_effect = LoadModelSideEffect(
        model_path="models/yolo11n.pt",
        device=ctx.job_request.device,
    )

    return (State.INITIALIZING, [side_effect])


@register_transition(State.INITIALIZING, Event.READY_TO_TRACK)
def ready_to_track(ctx: Context) -> tuple[State, list[SideEffect]]:
    """
    TRANSITION: ready_to_track
    From: INITIALIZING -> To: TRACKING
    Trigger: Model and frame reader are loaded by side-effect handlers
    Reads: ctx.status (must be in INITIALIZING), ctx.model, ctx.frame_reader
    Writes: ctx.status.state = TRACKING, ctx.status.updated_at
    May call: none
    Must NOT call: vision.*, storage.*, metrics.*, rendering.*
    Postcondition: ctx.status.state == TRACKING; no new side effects are emitted
    Errors: InvalidTransitionError if ctx.status.state != INITIALIZING
    """
    if ctx.status.state != State.INITIALIZING:
        raise InvalidTransitionError(
            f"ready_to_track requires INITIALIZING state, got {ctx.status.state.value}"
        )

    ctx.status.state = State.TRACKING
    ctx.status.updated_at = datetime.now(timezone.utc)

    return (State.TRACKING, [])


@register_transition(State.TRACKING, Event.FRAME_PROCESSED)
def frame_processed(ctx: Context) -> tuple[State, list[SideEffect]]:
    """
    TRANSITION: frame_processed
    From: TRACKING -> To: TRACKING
    Trigger: One frame has been processed by the tracking pipeline
    Reads: ctx.status (must be in TRACKING), ctx.frame_records, ctx.model, ctx.frame_reader
    Writes: ctx.status.updated_at
    May call: none
    Must NOT call: vision.*, storage.*, metrics.*, rendering.*
    Postcondition: ctx.status.state == TRACKING; ctx.frame_records is left in place
    Errors: InvalidTransitionError if ctx.status.state != TRACKING
    """
    if ctx.status.state != State.TRACKING:
        raise InvalidTransitionError(
            f"frame_processed requires TRACKING state, got {ctx.status.state.value}"
        )

    ctx.status.updated_at = datetime.now(timezone.utc)

    return (State.TRACKING, [])
