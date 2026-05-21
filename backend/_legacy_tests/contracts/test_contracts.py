"""Round-trip serialization and constraint tests for every contract model."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from salamander.contracts import (
    CONTRACT_VERSION,
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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-frame detections
# ---------------------------------------------------------------------------

class TestDetection:
    def test_round_trip(self):
        round_trip(Detection, load_golden("detection.json"))

    def test_no_region_field(self):
        d = Detection(
            track_id=1,
            bbox=Bbox(x1=0, y1=0, x2=10, y2=10),
            center=Center(x=5, y=5),
            confidence=0.9,
            class_name="salamander",
            frame_index=0,
        )
        assert not hasattr(d, "region")

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            Detection(
                track_id=1,
                bbox=Bbox(x1=0, y1=0, x2=10, y2=10),
                center=Center(x=5, y=5),
                confidence=1.01,
                class_name="salamander",
                frame_index=0,
            )

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            Detection(
                track_id=1,
                bbox=Bbox(x1=0, y1=0, x2=10, y2=10),
                center=Center(x=5, y=5),
                confidence=-0.01,
                class_name="salamander",
                frame_index=0,
            )

    def test_negative_frame_index_rejected(self):
        with pytest.raises(ValidationError):
            Detection(
                track_id=1,
                bbox=Bbox(x1=0, y1=0, x2=10, y2=10),
                center=Center(x=5, y=5),
                confidence=0.9,
                class_name="salamander",
                frame_index=-1,
            )


class TestFrameRecord:
    def test_round_trip(self):
        round_trip(FrameRecord, load_golden("frame_record.json"))

    def test_empty_detections(self):
        fr = FrameRecord(
            contract_version=CONTRACT_VERSION,
            frame_index=0,
            timestamp_s=0.0,
            detections=[],
        )
        assert fr.detections == []

    def test_contract_version_is_required(self):
        with pytest.raises(ValidationError):
            FrameRecord(frame_index=0, timestamp_s=0.0, detections=[])

    def test_contract_version_value(self):
        data = load_golden("frame_record.json")
        fr = FrameRecord.model_validate(data)
        assert fr.contract_version == CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

class TestTrackSummary:
    def test_round_trip(self):
        round_trip(TrackSummary, load_golden("track_summary.json"))

    def test_no_dwell_time_field(self):
        ts = TrackSummary(
            track_id=1,
            total_distance_px=50.0,
            frame_indices=[0, 2, 4],
            first_seen=0,
            last_seen=4,
            detection_count=3,
        )
        assert not hasattr(ts, "dwell_time_seconds")

    def test_negative_distance_rejected(self):
        with pytest.raises(ValidationError):
            TrackSummary(
                track_id=1,
                total_distance_px=-1.0,
                frame_indices=[0],
                first_seen=0,
                last_seen=0,
                detection_count=1,
            )

    def test_negative_detection_count_rejected(self):
        with pytest.raises(ValidationError):
            TrackSummary(
                track_id=1,
                total_distance_px=0.0,
                frame_indices=[],
                first_seen=0,
                last_seen=0,
                detection_count=-1,
            )


class TestProcessingMetadata:
    def test_round_trip(self):
        round_trip(ProcessingMetadata, load_golden("processing_metadata.json"))

    def test_all_fields_present(self):
        data = load_golden("processing_metadata.json")
        pm = ProcessingMetadata.model_validate(data)
        assert pm.video_id == "ensatina_short"
        assert pm.sample_stride == 2
        assert pm.fps == 24.0

    def test_zero_fps_rejected(self):
        data = load_golden("processing_metadata.json")
        data["fps"] = 0.0
        with pytest.raises(ValidationError):
            ProcessingMetadata.model_validate(data)

    def test_zero_stride_rejected(self):
        data = load_golden("processing_metadata.json")
        data["sample_stride"] = 0
        with pytest.raises(ValidationError):
            ProcessingMetadata.model_validate(data)

    def test_confidence_threshold_above_one_rejected(self):
        data = load_golden("processing_metadata.json")
        data["confidence_threshold"] = 1.1
        with pytest.raises(ValidationError):
            ProcessingMetadata.model_validate(data)

    def test_iou_threshold_negative_rejected(self):
        data = load_golden("processing_metadata.json")
        data["iou_threshold"] = -0.1
        with pytest.raises(ValidationError):
            ProcessingMetadata.model_validate(data)


class TestMetricsWarnings:
    def test_round_trip(self):
        round_trip(MetricsWarnings, load_golden("metrics_warnings.json"))

    def test_zero_defaults(self):
        mw = MetricsWarnings()
        assert mw.skipped_jumps == 0
        assert mw.dropped_low_confidence == 0

    def test_structured_counters(self):
        mw = MetricsWarnings(skipped_jumps=3, dropped_low_confidence=12)
        assert mw.skipped_jumps == 3
        assert mw.dropped_low_confidence == 12

    def test_negative_skipped_jumps_rejected(self):
        with pytest.raises(ValidationError):
            MetricsWarnings(skipped_jumps=-1)

    def test_negative_dropped_count_rejected(self):
        with pytest.raises(ValidationError):
            MetricsWarnings(dropped_low_confidence=-1)

    def test_no_warnings_list_field(self):
        assert "warnings" not in MetricsWarnings.model_fields


class TestJobMetrics:
    def test_round_trip(self):
        round_trip(JobMetrics, load_golden("job_metrics.json"))

    def test_contract_version_present(self):
        data = load_golden("job_metrics.json")
        jm = JobMetrics.model_validate(data)
        assert jm.contract_version == CONTRACT_VERSION

    def test_contract_version_is_required(self):
        data = load_golden("job_metrics.json")
        del data["contract_version"]
        with pytest.raises(ValidationError):
            JobMetrics.model_validate(data)

    def test_no_frame_records_field(self):
        assert "frame_records" not in JobMetrics.model_fields

    def test_field_names_tracks_and_processing(self):
        assert "tracks" in JobMetrics.model_fields
        assert "processing" in JobMetrics.model_fields
        assert "track_summaries" not in JobMetrics.model_fields
        assert "processing_metadata" not in JobMetrics.model_fields

    def test_no_region_grid_field(self):
        data = load_golden("job_metrics.json")
        jm = JobMetrics.model_validate(data)
        assert not hasattr(jm, "region_grid")


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

class TestJobRequest:
    def test_round_trip(self):
        round_trip(JobRequest, load_golden("job_request.json"))

    def test_non_version_defaults(self):
        jr = JobRequest(contract_version=CONTRACT_VERSION, video_id="test")
        assert jr.device == "auto"
        assert jr.sample_stride == 2
        assert jr.confidence_threshold == 0.25
        assert jr.iou_threshold == 0.7

    def test_contract_version_is_required(self):
        with pytest.raises(ValidationError):
            JobRequest(video_id="test")

    def test_invalid_device_rejected(self):
        with pytest.raises(ValidationError):
            JobRequest(contract_version=CONTRACT_VERSION, video_id="test", device="tpu")

    def test_valid_devices(self):
        for dev in ("auto", "cpu", "cuda", "mps"):
            jr = JobRequest(contract_version=CONTRACT_VERSION, video_id="test", device=dev)
            assert jr.device == dev

    def test_stride_zero_rejected(self):
        with pytest.raises(ValidationError):
            JobRequest(contract_version=CONTRACT_VERSION, video_id="test", sample_stride=0)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            JobRequest(contract_version=CONTRACT_VERSION, video_id="test",
                       confidence_threshold=1.1)

    def test_iou_negative_rejected(self):
        with pytest.raises(ValidationError):
            JobRequest(contract_version=CONTRACT_VERSION, video_id="test",
                       iou_threshold=-0.1)


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

    def test_percent_value(self):
        data = load_golden("job_progress.json")
        jp = JobProgress.model_validate(data)
        assert jp.percent == 50.0
        assert jp.current_frame == 180
        assert jp.total_frames == 360

    def test_percent_above_100_rejected(self):
        with pytest.raises(ValidationError):
            JobProgress(job_id="j", state="TRACKING",
                        current_frame=0, total_frames=100, percent=101.0)

    def test_negative_percent_rejected(self):
        with pytest.raises(ValidationError):
            JobProgress(job_id="j", state="TRACKING",
                        current_frame=0, total_frames=100, percent=-1.0)

    def test_negative_current_frame_rejected(self):
        with pytest.raises(ValidationError):
            JobProgress(job_id="j", state="TRACKING",
                        current_frame=-1, total_frames=100, percent=0.0)

    def test_zero_total_frames_rejected(self):
        with pytest.raises(ValidationError):
            JobProgress(job_id="j", state="TRACKING",
                        current_frame=0, total_frames=0, percent=0.0)

    def test_no_pct_field(self):
        assert "pct" not in JobProgress.model_fields


class TestJobError:
    def test_round_trip(self):
        round_trip(JobError, load_golden("job_error.json"))

    def test_traceback_optional(self):
        je = JobError(job_id="j1", error_message="boom", error_type="RuntimeError")
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
