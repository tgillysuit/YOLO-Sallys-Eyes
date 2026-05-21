"""Verify contract_version literals are present, correct, and required on every applicable model."""
import json
from pathlib import Path
from typing import get_args, get_type_hints

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
    VersionedContract,
)

VERSIONED_MODELS = [FrameRecord, JobMetrics, JobRequest]
UNVERSIONED_MODELS = [
    Bbox,
    Center,
    Detection,
    ErrorResponse,
    JobError,
    JobProgress,
    JobStatus,
    MetricsWarnings,
    ProcessingMetadata,
    TrackSummary,
]

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("model_cls", VERSIONED_MODELS)
def test_contract_version_field_exists(model_cls):
    assert "contract_version" in model_cls.model_fields


@pytest.mark.parametrize("model_cls", VERSIONED_MODELS)
def test_contract_version_literal_value(model_cls):
    hints = get_type_hints(model_cls)
    literal_args = get_args(hints["contract_version"])
    assert CONTRACT_VERSION in literal_args, (
        f"{model_cls.__name__}.contract_version literal does not include '{CONTRACT_VERSION}'"
    )


@pytest.mark.parametrize("model_cls", VERSIONED_MODELS)
def test_contract_version_field_is_required(model_cls):
    """contract_version must be required (no default) so TypeScript emits it without `?`."""
    field_info = model_cls.model_fields["contract_version"]
    assert field_info.is_required(), (
        f"{model_cls.__name__}.contract_version must have no default value"
    )


@pytest.mark.parametrize("model_cls", VERSIONED_MODELS)
def test_versioned_models_inherit_versioned_contract(model_cls):
    assert issubclass(model_cls, VersionedContract)


@pytest.mark.parametrize("model_cls", UNVERSIONED_MODELS)
def test_no_contract_version_on_unversioned_models(model_cls):
    assert "contract_version" not in model_cls.model_fields, (
        f"{model_cls.__name__} should not have a contract_version field in v1"
    )


def test_frame_record_golden_version():
    data = json.loads((GOLDEN / "frame_record.json").read_text())
    fr = FrameRecord.model_validate(data)
    assert fr.contract_version == CONTRACT_VERSION


def test_job_metrics_golden_version():
    data = json.loads((GOLDEN / "job_metrics.json").read_text())
    jm = JobMetrics.model_validate(data)
    assert jm.contract_version == CONTRACT_VERSION


def test_job_request_golden_version():
    data = json.loads((GOLDEN / "job_request.json").read_text())
    jr = JobRequest.model_validate(data)
    assert jr.contract_version == CONTRACT_VERSION


def test_contract_version_constant():
    assert CONTRACT_VERSION == "salamander.tracking.v1"
