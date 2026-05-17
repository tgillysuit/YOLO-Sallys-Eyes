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
from typing import Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = "salamander.tracking.v1"


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
    confidence: float
    class_name: str
    frame_index: int


class FrameRecord(BaseModel):
    """All detections captured for one sampled frame."""
    contract_version: Literal["salamander.tracking.v1"] = CONTRACT_VERSION
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
    total_distance_px: float
    frame_indices: list[int]
    first_seen: int
    last_seen: int
    detection_count: int


class ProcessingMetadata(BaseModel):
    """Parameters and counters describing a completed processing run."""
    video_id: str
    model_path: str
    sample_stride: int
    confidence_threshold: float
    iou_threshold: float
    device: str
    processed_frames: int
    total_frames: int
    duration_s: float
    fps: float


class MetricsWarnings(BaseModel):
    """Non-fatal warnings accumulated during a processing run."""
    warnings: list[str] = Field(default_factory=list)


class JobMetrics(BaseModel):
    """Full output manifest for a completed job.

    v1.1 deferred: region_grid field.
    """
    contract_version: Literal["salamander.tracking.v1"] = CONTRACT_VERSION
    video_id: str
    frame_records: list[FrameRecord]
    track_summaries: list[TrackSummary]
    processing_metadata: ProcessingMetadata
    metrics_warnings: MetricsWarnings


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """Client-submitted parameters for a new tracking job."""
    contract_version: Literal["salamander.tracking.v1"] = CONTRACT_VERSION
    video_id: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    sample_stride: int = 2
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7


class JobStatus(BaseModel):
    """Current lifecycle state of a job (no progress detail)."""
    job_id: str
    state: str
    created_at: datetime
    updated_at: datetime


class JobProgress(BaseModel):
    """Live progress snapshot while a job is in TRACKING or RENDERING."""
    job_id: str
    state: str
    current_frame: int
    total_frames: int
    pct: float  # 0.0–1.0


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
