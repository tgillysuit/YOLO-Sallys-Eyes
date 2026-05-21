"""Slow integration tests for the YOLO tracking wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salamander.vision.yolo_tracking import (
    ModelLoadError,
    RawFrameResult,
    infer_one,
    load_model,
)


pytestmark = pytest.mark.slow


def test_load_model_returns_model():
    """A real YOLO model can be loaded and warmed up."""
    model = load_model(Path("yolo11n.pt"), device="cpu")

    assert model is not None


def test_load_model_unknown_path_raises_ModelLoadError():
    """Unknown model paths are wrapped in ModelLoadError."""
    with pytest.raises(ModelLoadError):
        load_model(Path("definitely_missing_model_file.pt"), device="cpu")


def test_infer_one_on_black_frame_returns_empty_detections():
    """A black frame should not produce tracked objects."""
    model = load_model(Path("yolo11n.pt"), device="cpu")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    result = infer_one(model, frame, frame_index=0, conf=0.25, iou=0.7)

    assert isinstance(result, RawFrameResult)
    assert result.frame_index == 0
    assert result.frame_width == 640
    assert result.frame_height == 640
    assert result.raw_detections == []
