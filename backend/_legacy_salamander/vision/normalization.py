"""Normalize raw tracking output into salamander contract records."""
from __future__ import annotations

from .yolo_tracking import RawFrameResult
from ..contracts import (
    CONTRACT_VERSION,
    Bbox,
    Center,
    Detection,
    FrameRecord,
)


class ContractViolationError(Exception):
    """Raised when raw tracking output cannot satisfy the public contract."""


def _clamp_coordinate(value: float, lower: float, upper: float, label: str) -> float:
    if lower <= value <= upper:
        return value
    if value < lower and lower - value <= 0.5:
        return lower
    if value > upper and value - upper <= 0.5:
        return upper
    raise ContractViolationError(f"{label}={value} is outside frame bounds")


def normalize(
    raw: RawFrameResult,
    fps: float,
    timestamp_s: float,
    video_id: str,
    class_name_map: dict[int, str],
    confidence_threshold: float,
) -> FrameRecord:
    """Convert raw detections into a FrameRecord without computing metrics."""
    _ = (fps, video_id)
    detections: list[Detection] = []

    for raw_detection in raw.raw_detections:
        if raw_detection.track_id is None:
            continue
        if raw_detection.confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = raw_detection.bbox_xyxy
        if x1 >= x2:
            raise ContractViolationError("bbox requires x1 < x2")
        if y1 >= y2:
            raise ContractViolationError("bbox requires y1 < y2")

        x1 = _clamp_coordinate(x1, 0.0, float(raw.frame_width), "x1")
        y1 = _clamp_coordinate(y1, 0.0, float(raw.frame_height), "y1")
        x2 = _clamp_coordinate(x2, 0.0, float(raw.frame_width), "x2")
        y2 = _clamp_coordinate(y2, 0.0, float(raw.frame_height), "y2")

        detections.append(
            Detection(
                track_id=raw_detection.track_id,
                bbox=Bbox(x1=x1, y1=y1, x2=x2, y2=y2),
                center=Center(x=0.0, y=0.0),
                confidence=raw_detection.confidence,
                class_name=class_name_map.get(
                    raw_detection.class_id,
                    str(raw_detection.class_id),
                ),
                frame_index=raw.frame_index,
            )
        )

    return FrameRecord(
        contract_version=CONTRACT_VERSION,
        frame_index=raw.frame_index,
        timestamp_s=timestamp_s,
        detections=detections,
    )
