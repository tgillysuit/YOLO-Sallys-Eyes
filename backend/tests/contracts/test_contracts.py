"""Round-trip serialization tests for every contract model."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from salamander.contracts import (
    Bbox,
    Center,
    Detection,
    ErrorResponse,
    FrameRecord,
    JobError,
    JobMetrics,
    JobProgress,
    JobRequest,
    JobStatus,
    MetricsWarnings,
    ProcessingMetadata,
    TrackSummary,
)

GOLDEN = Path(__file__).parent / "golden"


def load_golden(filename: str) -> dict:
    return json.loads((GOLDEN / filename).read_text())


def round_trip(model_cls, data: dict):
    """Deserialize, reserialize, re-deserialize and assert equality."""
    instance = model_cls.model_validate(data)
    serialized = instance.model_dump(mode="json")
    rebuilt = model_cls.model_validate(serialized)
    assert instance == rebuilt
    return instance


class TestBbox:
    def test_round_trip(self):
        round_trip(Bbox, load_golden("bbox.json"))

    def test_fields(self):
        b = Bbox(x1=0.0, y1=0.0, x2=100.0, y2=200.0)
        assert b.x1 == 0.0
        assert b.y2 == 200.0

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            Bbox(x1=0.0, y1=0.0, x2=100.0)  # missing y2


class TestCenter:
    def test_round_trip(self):
        round_trip(Center, load_golden("center.json"))

    def test_fields(self):
        c = Center(x=50.0, y=75.5)
        assert c.x == 50.0
        assert c.y == 75.5


class TestDetection:
    def test_round_trip(self):
        round_trip(Detection, load_golden("detection.json"))

    def test_no_region_field(self):
        # v1.1 deferred: region must NOT exist
        d = Detection(
            track_id=1,
            bbox=Bbox(x1=0, y1=0, x2=10, y2=10),
            center=Center(x=5, y=5),
            confidence=0.9,
            class_name="salamander",
            frame_index=0,
        )
        assert not hasattr(d, "region")


class TestFrameRecord:
    def test_round_trip(self):
        round_trip(FrameRecord, load_golden("frame_record.json"))

    def test_empty_detections(self):
        fr = FrameRecord(frame_index=0, timestamp_s=0.0, detections=[])
        assert fr.detections == []

    def test_contract_version_default(self):
        fr = FrameRecord(frame_index=0, timestamp_s=0.0, detections=[])
        assert fr.contract_version == "salamander.tracking.v1"


class TestTrackSummary:
    def test_round_trip(self):
        round_trip(TrackSummary, load_golden("track_summary.json"))

    def test_no_dwell_time_field(self):
        # v1.1 deferred
        ts = TrackSummary(
            track_id=1,
            total_distance_px=50.0,
            frame_indices=[0, 2, 4],
            first_seen=0,
            last_seen=4,
            detection_count=3,
        )
        assert not hasattr(ts, "dwell_time_seconds")


class TestProcessingMetadata:
    def test_round_trip(self):
        round_trip(ProcessingMetadata, load_golden("processing_metadata.json"))

    def test_all_fields_present(self):
        data = load_golden("processing_metadata.json")
        pm = ProcessingMetadata.model_validate(data)
        assert pm.video_id == "ensatina_short"
        assert pm.sample_stride == 2
        assert pm.fps == 24.0


class TestMetricsWarnings:
    def test_round_trip(self):
        round_trip(MetricsWarnings, load_golden("metrics_warnings.json"))

    def test_empty_default(self):
        mw = MetricsWarnings()
        assert mw.warnings == []

    def test_with_warnings(self):
        mw = MetricsWarnings(warnings=["foo", "bar"])
        assert len(mw.warnings) == 2


class TestJobMetrics:
    def test_round_trip(self):
        round_trip(JobMetrics, load_golden("job_metrics.json"))

    def test_contract_version_default(self):
        data = load_golden("job_metrics.json")
        jm = JobMetrics.model_validate(data)
        assert jm.contract_version == "salamander.tracking.v1"

    def test_no_region_grid_field(self):
        # v1.1 deferred
        data = load_golden("job_metrics.json")
        jm = JobMetrics.model_validate(data)
        assert not hasattr(jm, "region_grid")


class TestJobRequest:
    def test_round_trip(self):
        round_trip(JobRequest, load_golden("job_request.json"))

    def test_defaults(self):
        jr = JobRequest(video_id="test")
        assert jr.device == "auto"
        assert jr.sample_stride == 2
        assert jr.confidence_threshold == 0.25
        assert jr.iou_threshold == 0.7
        assert jr.contract_version == "salamander.tracking.v1"

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            JobRequest(video_id="test", device="tpu")

    def test_valid_devices(self):
        for dev in ("auto", "cpu", "cuda", "mps"):
            jr = JobRequest(video_id="test", device=dev)
            assert jr.device == dev


class TestJobStatus:
    def test_round_trip(self):
        round_trip(JobStatus, load_golden("job_status.json"))

    def test_fields(self):
        data = load_golden("job_status.json")
        js = JobStatus.model_validate(data)
        assert js.job_id == "job-abc123"
        assert js.state == "TRACKING"


class TestJobProgress:
    def test_round_trip(self):
        round_trip(JobProgress, load_golden("job_progress.json"))

    def test_pct_value(self):
        data = load_golden("job_progress.json")
        jp = JobProgress.model_validate(data)
        assert jp.pct == 0.5
        assert jp.current_frame == 180
        assert jp.total_frames == 360


class TestJobError:
    def test_round_trip(self):
        round_trip(JobError, load_golden("job_error.json"))

    def test_traceback_optional(self):
        je = JobError(
            job_id="j1", error_message="boom", error_type="RuntimeError"
        )
        assert je.traceback is None

    def test_traceback_present(self):
        je = JobError(
            job_id="j1",
            error_message="boom",
            error_type="RuntimeError",
            traceback="Traceback (most recent call last):\n  ...",
        )
        assert je.traceback is not None


class TestErrorResponse:
    def test_round_trip(self):
        round_trip(ErrorResponse, load_golden("error_response.json"))

    def test_error_type_optional(self):
        er = ErrorResponse(detail="something went wrong")
        assert er.error_type is None

    def test_with_error_type(self):
        er = ErrorResponse(detail="not found", error_type="NotFound")
        assert er.error_type == "NotFound"
