"""
Data-shape contracts for the salamander tracking pipeline.

Contract version: salamander.tracking.v1

Rules:
- No business logic here. Pure data shapes only.
- v1.1 deferred fields: Detection.region, TrackSummary.dwell_time_seconds,
  JobMetrics.region_grid. Do not add them until v1.1.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "CONTRACT_VERSION",
    "State",
    "Event",
    "VersionedContract",
    "Bbox",
    "Center",
    "Detection",
    "FrameRecord",
    "TrackSummary",
    "ProcessingMetadata",
    "MetricsWarnings",
    "JobMetrics",
    "JobRequest",
    "JobStatus",
    "JobProgress",
    "JobError",
    "ErrorResponse",
]

CONTRACT_VERSION = "salamander.tracking.v1"


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


class VersionedContract(BaseModel):
    """Base for all versioned API models. contract_version is required — no default.

    Every constructor call must pass contract_version=CONTRACT_VERSION explicitly.
    The frontend version-mismatch check relies on the field always being present
    in serialized form (TypeScript emits it as required, not optional).
    """
    contract_version: Literal["salamander.tracking.v1"]


# ---------------------------------------------------------------------------
# Primitive geometry
# ---------------------------------------------------------------------------

class Bbox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (corner format)."""
    x1: float
    y1: float
    x2: float
    y2: float


class Center(BaseModel):
    """Centroid of a bounding box in pixel coordinates."""
    x: float
    y: float


# ---------------------------------------------------------------------------
# Per-frame detections
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    """One tracked detection within a single frame.

    v1.1 deferred: region field.
    """
    track_id: int
    bbox: Bbox
    center: Center
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    frame_index: int = Field(..., ge=0)


class FrameRecord(VersionedContract):
    """All detections captured for one sampled frame."""
    frame_index: int
    timestamp_s: float
    detections: list[Detection]


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

class TrackSummary(BaseModel):
    """Lifetime summary for a single ByteTrack track.

    v1.1 deferred: dwell_time_seconds field.
    """
    track_id: int
    total_distance_px: float = Field(..., ge=0.0)
    frame_indices: list[int]
    first_seen: int = Field(..., ge=0)
    last_seen: int = Field(..., ge=0)
    detection_count: int = Field(..., ge=0)


class ProcessingMetadata(BaseModel):
    """Parameters and counters describing a completed processing run."""
    video_id: str
    model_path: str
    sample_stride: int = Field(..., ge=1)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    iou_threshold: float = Field(..., ge=0.0, le=1.0)
    device: str
    processed_frames: int = Field(..., ge=0)
    total_frames: int = Field(..., ge=0)
    duration_s: float = Field(..., ge=0.0)
    fps: float = Field(..., gt=0.0)


class MetricsWarnings(BaseModel):
    """Structured counters for non-fatal quality issues in a processing run."""
    skipped_jumps: int = Field(default=0, ge=0)
    dropped_low_confidence: int = Field(default=0, ge=0)


class JobMetrics(VersionedContract):
    """Summary manifest written at job completion.

    Frame-level data lives in frames.json, not here. Embedding frame_records
    would serialize the entire detection log into this model, causing memory
    and I/O problems on long videos.

    Field names follow the spec: `tracks` (not track_summaries),
    `processing` (not processing_metadata).

    v1.1 deferred: region_grid field.
    """
    video_id: str
    tracks: list[TrackSummary]
    processing: ProcessingMetadata
    metrics_warnings: MetricsWarnings


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

class JobRequest(VersionedContract):
    """Client-submitted parameters for a new tracking job."""
    video_id: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    sample_stride: int = Field(default=2, ge=1)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class JobStatus(BaseModel):
    """Current lifecycle state of a job (no progress detail)."""
    job_id: str
    state: State
    created_at: datetime
    updated_at: datetime


class JobProgress(BaseModel):
    """Live progress snapshot while a job is in TRACKING or RENDERING."""
    job_id: str
    state: State
    current_frame: int = Field(..., ge=0)
    total_frames: int = Field(..., ge=1)
    percent: float = Field(..., ge=0.0, le=100.0)


class JobError(BaseModel):
    """Error payload attached to a FAILED job."""
    job_id: str
    error_message: str
    error_type: str
    traceback: str | None = None


class ErrorResponse(BaseModel):
    """Generic API error envelope."""
    detail: str
    error_type: str | None = None
