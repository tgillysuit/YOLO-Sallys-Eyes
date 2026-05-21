"""Unit tests for raw tracking result normalization."""
from __future__ import annotations

import pytest

from salamander.contracts import CONTRACT_VERSION
from salamander.vision.normalization import ContractViolationError, normalize
from salamander.vision.yolo_tracking import RawDetection, RawFrameResult


def raw_frame(*detections: RawDetection) -> RawFrameResult:
    """Build a small raw frame fixture."""
    return RawFrameResult(
        frame_index=7,
        raw_detections=list(detections),
        frame_width=100,
        frame_height=50,
    )


def detection(
    *,
    track_id: int | None = 1,
    class_id: int = 0,
    confidence: float = 0.9,
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 5.0, 20.0, 15.0),
) -> RawDetection:
    """Build a raw detection fixture."""
    return RawDetection(
        track_id=track_id,
        class_id=class_id,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
    )


def normalize_frame(raw: RawFrameResult):
    """Normalize with stable defaults."""
    return normalize(
        raw=raw,
        fps=30.0,
        timestamp_s=0.25,
        video_id="test_video_001",
        class_name_map={0: "salamander"},
        confidence_threshold=0.5,
    )


def test_filters_below_confidence_threshold():
    frame_record = normalize_frame(
        raw_frame(
            detection(confidence=0.49),
            detection(track_id=2, confidence=0.5),
        )
    )

    assert [d.track_id for d in frame_record.detections] == [2]


def test_filters_none_track_ids():
    frame_record = normalize_frame(
        raw_frame(
            detection(track_id=None),
            detection(track_id=2),
        )
    )

    assert [d.track_id for d in frame_record.detections] == [2]


def test_invalid_bbox_x_order_raises_ContractViolationError():
    with pytest.raises(ContractViolationError):
        normalize_frame(raw_frame(detection(bbox_xyxy=(20.0, 5.0, 10.0, 15.0))))


def test_invalid_bbox_y_order_raises_ContractViolationError():
    with pytest.raises(ContractViolationError):
        normalize_frame(raw_frame(detection(bbox_xyxy=(10.0, 15.0, 20.0, 5.0))))


def test_clamps_slightly_out_of_bounds_bbox():
    frame_record = normalize_frame(
        raw_frame(detection(bbox_xyxy=(-0.25, -0.5, 100.25, 50.5)))
    )

    bbox = frame_record.detections[0].bbox
    assert bbox.x1 == 0.0
    assert bbox.y1 == 0.0
    assert bbox.x2 == 100.0
    assert bbox.y2 == 50.0


def test_significantly_out_of_bounds_bbox_raises():
    with pytest.raises(ContractViolationError):
        normalize_frame(raw_frame(detection(bbox_xyxy=(-0.51, 5.0, 20.0, 15.0))))


def test_empty_detections_produces_empty_framerecord():
    frame_record = normalize_frame(raw_frame())

    assert frame_record.detections == []
    assert frame_record.frame_index == 7
    assert frame_record.timestamp_s == 0.25


def test_framerecord_has_correct_contract_version():
    frame_record = normalize_frame(raw_frame(detection()))

    assert frame_record.contract_version == CONTRACT_VERSION


def test_normalization_does_not_compute_real_center():
    frame_record = normalize_frame(
        raw_frame(detection(bbox_xyxy=(10.0, 5.0, 20.0, 15.0)))
    )

    center = frame_record.detections[0].center
    assert center.x == 0.0
    assert center.y == 0.0
