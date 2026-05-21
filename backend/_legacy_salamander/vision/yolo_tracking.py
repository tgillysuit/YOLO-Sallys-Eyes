"""Thin Ultralytics YOLO wrapper for persistent ByteTrack tracking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ModelLoadError(Exception):
    """Raised when the YOLO model cannot be loaded or warmed up."""


class ModelInferenceError(Exception):
    """Raised when single-frame tracking inference fails."""


@dataclass(frozen=True)
class RawDetection:
    track_id: int | None
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RawFrameResult:
    frame_index: int
    raw_detections: list[RawDetection]
    frame_width: int
    frame_height: int


def load_model(model_path: Path, device: str):
    """Load YOLO and run one black-frame warm-up inference."""
    try:
        import numpy as np
        from ultralytics import YOLO

        model = YOLO(model_path)
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        kwargs = {"verbose": False}
        if device != "auto":
            kwargs["device"] = device
        model(warmup_frame, **kwargs)
        return model
    except Exception as exc:  # pragma: no cover - exercised by slow integration tests
        raise ModelLoadError(f"Failed to load model from {model_path}") from exc


def infer_one(
    model,
    frame: np.ndarray,
    frame_index: int,
    conf: float,
    iou: float,
) -> RawFrameResult:
    """Run persistent ByteTrack inference on one frame and return raw detections."""
    try:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            verbose=False,
        )
        frame_height, frame_width = frame.shape[:2]

        if not results:
            return RawFrameResult(frame_index, [], frame_width, frame_height)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return RawFrameResult(frame_index, [], frame_width, frame_height)

        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
        track_id_values = (
            [None] * len(xyxy_values)
            if boxes.id is None
            else boxes.id.cpu().tolist()
        )

        raw_detections = [
            RawDetection(
                track_id=None if track_id is None else int(track_id),
                class_id=int(class_id),
                confidence=float(confidence),
                bbox_xyxy=tuple(float(value) for value in bbox),
            )
            for track_id, class_id, confidence, bbox in zip(
                track_id_values,
                class_values,
                confidence_values,
                xyxy_values,
                strict=True,
            )
        ]
        return RawFrameResult(frame_index, raw_detections, frame_width, frame_height)
    except Exception as exc:
        raise ModelInferenceError(f"Inference failed for frame {frame_index}") from exc
