# YOLO Sally's Eyes - Project Status Summary

## Project Overview
Building a YOLO-based salamander detection and tracking system.

**Strategy:** Path B - Pretrained model spike first, custom model later

---

## Day 5 Pre-Flight Cleanup ✅

### Task Status

| Task | Status | Notes |
|------|--------|-------|
| 1. Filename fix (`ensantina` → `ensatina`) | DONE | `git mv` used; `PROJECT_STATUS.md` updated; `spike.py` was already correct |
| 2. `.gitignore` created | DONE | 29-line file with weights, spike artifacts, Python, Node, IDE, OS, training-data sections |
| 3. Untrack `yolo11n.pt` | DONE | `git rm --cached yolo11n.pt` succeeded |
| 4. Move spike output to docs | DONE | `spike_output.mp4` → `docs/screenshots/spike_pretrained_baseline.mp4`; `spike_output_raw.mp4` removed |
| 5. Contract tests | DONE | **85 passed, 0 failed** |
| 6. Transitions state inspection | DONE | See below |

### Transitions State (Task 6)

| Check | Result |
|-------|--------|
| `backend/salamander/transitions.py` | **MISSING** — not yet created |
| `backend/tests/transitions/` | **MISSING** — not yet created |
| `TRANSITION_TABLE` entries | **0 registered** — empty dict as designed |

`transitions.py` has not been started. The state machine spine (`State`, `Event`, `TRANSITION_TABLE`, `fire()`, `register_transition`) is fully wired in `state.py` and ready to receive transition implementations.

### Next Step

**Awaiting Day 5 Phase 1 prompts based on transitions.py state.**

Day 5 starts from a clean baseline:
- Contracts locked and tested (85 tests passing)
- State machine skeleton in place, 0 transitions registered
- `transitions.py` does not exist yet (blank file to create)
- `tests/transitions/` directory does not exist yet

---

## Previous Steps

### 1. Frame Extraction Tool ✅
**Created:** `tools/extract_frames.py`

**Features:**
- Standalone script for extracting frames from videos
- Accepts video file path and output directory as arguments
- `--stride N` flag (default 15) to extract every Nth frame
- Saves frames as zero-padded PNGs: `frame_00001.png`, `frame_00002.png`, etc.
- Prints progress and summary (total frames, frames extracted, output directory)
- Uses OpenCV (`cv2.VideoCapture`)

**Status:** Created and ready to use

---

### 2. Video Trimming Tool ✅
**Created:** `tools/trim_video.py`

**Features:**
- Trims videos to specified duration using OpenCV
- Used to create 30-second test video from full source

**Status:** Created and tested successfully

---

### 3. Source Video Information ✅
**File:** `samples/ensatina.mp4`

**Metadata:**
- Duration: 478.67 seconds (~8 minutes)
- Resolution: 408x360
- Frame rate: 24.0 fps
- Codec: h264
- File size: 11M

**Trimmed version:** `samples/ensatina_short.mp4` (30 seconds, 765K)

---

### 4. Spike Test - YOLO Tracking Pipeline ✅✅✅
**Created:** `spike.py`

**Purpose:** Validate end-to-end pipeline architecture using pretrained yolo11n.pt model

**Results:**
- ✅ Script ran without errors
- ✅ Processed 720 frames (30 seconds @ 24fps)
- ✅ Generated annotated video outputs:
  - `spike_output_raw.mp4` (1.6M)
  - `spike_output.mp4` (530K, H.264 baseline for browser compatibility)
- ✅ ffmpeg encoding succeeded (return code 0)
- ✅ ByteTrack assigned 6 unique track IDs
- ✅ Total detections: 1,091
- ✅ All distances finite and non-negative

**Detections (pretrained model - NOT salamander-specific):**
As expected, yolo11n.pt did NOT detect salamanders (not in training set). Misclassified as:
- person: 330
- scissors: 425
- bird: 123
- umbrella: 113
- toothbrush: 68
- bed: 30
- cat: 2

**This is EXPECTED and ACCEPTABLE for the spike** - we're validating the pipeline, not detection accuracy.

**Per-Track Distances (pixels):**
- Track 1: 86.26
- Track 7: 0.00
- Track 11: 23.71
- Track 12: 2.40
- Track 13: 0.67
- Track 14: 0.04

**Critical Validations Passed:**
- ✅ Pipeline runs end-to-end without crashing
- ✅ Annotated video output generated and plays correctly
- ✅ ByteTrack assigns track IDs (even to misdetections)
- ✅ Distance calculations produce finite, non-NaN values

**Spike Status:** ✅ **PASSED** - Pipeline architecture validated

---

## Current Directory Structure

```
YOLO-Sallys-Eyes/
├── .claude/
├── .git/
├── data/
│   └── extracted_frames/  (ready for future use)
├── samples/
│   ├── ensatina.mp4       (full 8-minute source video)
│   └── ensatina_short.mp4 (30-second trimmed version)
├── tools/
│   ├── extract_frames.py  ✅
│   └── trim_video.py      ✅
├── spike.py               ✅ (tracking pipeline spike test)
├── spike_output_raw.mp4   ✅ (raw annotated output)
├── spike_output.mp4       ✅ (H.264 encoded for browsers)
├── yolo11n.pt            (pretrained model - auto-downloaded)
├── PROJECT_STATUS.md
├── .cursorrules
└── LICENSE
```

---

---

## Day 4: Architectural Spine ✅

### Files Created

| File | Purpose |
|------|---------|
| `backend/salamander/contracts.py` | All pydantic v2 data-shape models (v1 scope) |
| `backend/salamander/state.py` | State/Event enums, TRANSITION_TABLE, Context, SideEffects, fire() |
| `backend/pyproject.toml` | Backend package config; pydantic-to-typescript in dev deps |
| `backend/tests/contracts/test_contracts.py` | Round-trip + field tests |
| `backend/tests/contracts/test_contract_version.py` | Version literal enforcement |
| `backend/tests/contracts/golden/*.json` | 13 golden JSON fixtures, one per model |
| `tools/generate_ts_types.py` | Python-only TS interface generator (no Node required) |
| `Makefile` | `generate-types` and `test-contracts` targets |
| `frontend/lib/contracts.ts` | Generated TypeScript interfaces (do not hand-edit) |

### Test Results
```
57 passed in 0.58s
```

### Contract Decisions Made (review these)

1. **Bbox coordinates** — pixel space, corner format `(x1, y1, x2, y2)` matching YOLO's `xyxy` output. Not normalized.
2. **JobStatus fields** — `job_id, state: str, created_at: datetime, updated_at: datetime`. No `video_id` on status (kept minimal).
3. **JobProgress.pct** — range `0.0–1.0` (not 0–100).
4. **MetricsWarnings** — a model with `warnings: list[str]`, not a bare list. Allows adding severity levels later without breaking the contract.
5. **TypeScript generator** — uses pydantic v2 `model_fields` + `get_type_hints()` directly. No Node.js / npm required. `pydantic-to-typescript` is still listed in dev deps per spec, but the script does not depend on it.
6. **state.py imports contracts.py** — one-way dependency. contracts.py has zero imports from state.py.
7. **`from __future__ import annotations`** — used in both modules for forward-reference hygiene.

### v1.1 Deferred Fields (explicitly absent)
- `Detection.region`
- `TrackSummary.dwell_time_seconds`
- `JobMetrics.region_grid`

### Next Step
**Day 4 PM** — implement `submit_job` and `start_processing` transitions in Cursor (register via `@register_transition` decorators in a new `backend/salamander/transitions.py`).

---

## Previous Steps

### Future: Custom Model Training
After Day 4, we can:
1. Extract frames from full video using `tools/extract_frames.py`
2. Label salamanders in Label Studio
3. Export to `datasets/salamander/` in YOLO format
4. Train custom YOLO model on salamander data
5. Swap pretrained yolo11n.pt for custom model in pipeline
6. Re-run spike with custom model for accurate salamander detection

---

## Technical Environment

**Platform:** Windows (MINGW64_NT)
**Python:** 3.12
**Git:** Initialized, on `main` branch

**Python Packages Installed:**
- opencv-python==4.13.0.92
- numpy==2.4.4
- imageio-ffmpeg==0.6.0
- ultralytics==8.4.50
- torch==2.12.0
- torchvision==0.27.0
- matplotlib==3.10.9
- pillow==12.2.0
- pyyaml==6.0.3
- scipy==1.17.1
- polars==1.40.1
- psutil==7.2.2
- lap==0.5.13 (ByteTrack dependency)
- And various dependencies

**Tools Available:**
- ffmpeg (via imageio-ffmpeg package)
- YOLO CLI (via ultralytics package)
