#!/usr/bin/env python3
"""
Generate frontend/lib/contracts.ts from backend/salamander/contracts.py.

Uses pydantic v2's model_fields + type hints directly — no Node/npm required.
Run via: make generate-types
"""
from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import BaseModel

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

# Ordered so that referenced types appear before referencing types.
MODELS: list[type[BaseModel]] = [
    Bbox,
    Center,
    Detection,
    FrameRecord,
    TrackSummary,
    ProcessingMetadata,
    MetricsWarnings,
    JobMetrics,
    JobRequest,
    JobStatus,
    JobProgress,
    JobError,
    ErrorResponse,
]


def py_type_to_ts(hint) -> str:
    """Convert a Python type hint to a TypeScript type string."""
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is Literal:
        parts = [f'"{a}"' if isinstance(a, str) else str(a).lower() for a in args]
        return " | ".join(parts)

    # Optional[X] is Union[X, None]; PEP 604 `X | None` uses types.UnionType
    if origin is Union or isinstance(hint, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        ts_parts = [py_type_to_ts(a) for a in non_none]
        result = " | ".join(ts_parts)
        if has_none:
            result += " | null"
        return result

    if origin is list:
        inner = py_type_to_ts(args[0]) if args else "unknown"
        # Wrap in parens if it's a union to disambiguate: (A | B)[]
        if " | " in inner:
            return f"({inner})[]"
        return f"{inner}[]"

    if hint is str or hint is datetime:
        return "string"
    if hint is int or hint is float:
        return "number"
    if hint is bool:
        return "boolean"
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        return hint.__name__

    return "unknown"


def model_to_interface(model_cls: type[BaseModel]) -> str:
    hints = get_type_hints(model_cls)
    lines = [f"export interface {model_cls.__name__} {{"]
    for name, field_info in model_cls.model_fields.items():
        ts_type = py_type_to_ts(hints[name])
        # Fields with a default value are optional in JSON (may be omitted by caller).
        # Fields with no default are required.
        optional = "?" if not field_info.is_required() else ""
        lines.append(f"  {name}{optional}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    out_path = ROOT / "frontend" / "lib" / "contracts.ts"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "// AUTO-GENERATED — do not edit by hand.\n"
        "// Regenerate with: make generate-types\n"
        f"// Source: backend/salamander/contracts.py\n"
    )

    interfaces = "\n\n".join(model_to_interface(m) for m in MODELS)
    out_path.write_text(header + "\n" + interfaces + "\n", encoding="utf-8")
    print(f"Written {len(MODELS)} interfaces -> {out_path}")


if __name__ == "__main__":
    main()
