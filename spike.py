#!/usr/bin/env python3
"""
Spike test: YOLO tracking pipeline with pretrained model.

This script validates the end-to-end pipeline architecture using a pretrained
yolo11n.pt model. The model is NOT expected to detect salamanders accurately,
as it wasn't trained on them. This is acceptable - we're validating the pipeline,
not detection accuracy.

Expected outcomes:
- Model may detect nothing (salamanders not in training set)
- Model may misclassify as something visually similar
- Either outcome validates the pipeline works

Critical validations:
- Pipeline runs without crashes
- Annotated video output is generated
- ByteTrack assigns track IDs (even to misdetections)
- Distance calculations produce finite, non-NaN values
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe


def calculate_track_distances(tracks_data):
    """
    Calculate total distance traveled for each track.

    Args:
        tracks_data: Dict mapping track_id -> [(frame_idx, center_x, center_y), ...]

    Returns:
        Dict mapping track_id -> total_distance
    """
    distances = {}

    for track_id, positions in tracks_data.items():
        if len(positions) < 2:
            distances[track_id] = 0.0
            continue

        # Sort by frame index
        positions = sorted(positions, key=lambda x: x[0])

        total_distance = 0.0
        prev_frame, prev_x, prev_y = positions[0]

        for frame_idx, x, y in positions[1:]:
            # Skip gaps > 10 frames
            if frame_idx - prev_frame > 10:
                prev_frame, prev_x, prev_y = frame_idx, x, y
                continue

            # Calculate Euclidean distance
            dx = x - prev_x
            dy = y - prev_y
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance

            prev_frame, prev_x, prev_y = frame_idx, x, y

        distances[track_id] = total_distance

    return distances


def run_spike(input_video, output_video_raw, output_video_final):
    """
    Run the spike test: tracking with pretrained YOLO model.

    Args:
        input_video: Path to input video
        output_video_raw: Path for raw output video
        output_video_final: Path for H.264-encoded output video

    Returns:
        Dict with results summary
    """
    print("="*70)
    print("YOLO TRACKING SPIKE TEST")
    print("="*70)
    print(f"Input video: {input_video}")
    print(f"Output (raw): {output_video_raw}")
    print(f"Output (final): {output_video_final}")
    print()

    # Load pretrained YOLO model (auto-downloads if needed)
    print("Loading pretrained yolo11n.pt model...")
    model = YOLO("yolo11n.pt")
    print(f"Model loaded successfully")
    print(f"Model classes: {len(model.names)} classes")
    print()

    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
    print()

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_raw), fourcc, fps, (width, height))

    # Track data: track_id -> [(frame_idx, center_x, center_y), ...]
    tracks_data = defaultdict(list)

    # Statistics
    frame_idx = 0
    total_detections = 0
    detections_by_class = defaultdict(int)

    print("Running YOLO tracking with ByteTrack...")
    print("Note: Pretrained model may not detect salamanders. This is expected.")
    print()

    # Run tracking on video
    # persist=True keeps ByteTrack state across frames
    results = model.track(str(input_video), persist=True, verbose=False, stream=True)

    for result in results:
        frame = result.orig_img.copy()

        # Get detections with tracking IDs
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Get track ID if available
                track_id = int(box.id[0]) if box.id is not None else None

                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Record detection
                total_detections += 1
                class_name = model.names[cls]
                detections_by_class[class_name] += 1

                # Store track data
                if track_id is not None:
                    tracks_data[track_id].append((frame_idx, center_x, center_y))

                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label with track ID
                if track_id is not None:
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                else:
                    label = f"{class_name} {conf:.2f}"

                # Add background for text
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10),
                            (int(x1) + label_w, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write frame
        out.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\nProcessed {frame_idx} frames")
    print(f"Raw output saved to: {output_video_raw}")
    print()

    # Calculate distances
    print("Calculating per-track distances...")
    distances = calculate_track_distances(tracks_data)

    print("\n" + "="*70)
    print("PER-TRACK DISTANCES (pixels)")
    print("="*70)

    if distances:
        for track_id in sorted(distances.keys()):
            dist = distances[track_id]
            print(f"Track {track_id:3d}: {dist:10.2f} pixels")
    else:
        print("No tracks found (no detections with track IDs)")

    print()

    # Re-encode with H.264 baseline for browser compatibility
    print("Re-encoding video with H.264 baseline profile...")
    ffmpeg_exe = get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        '-i', str(output_video_raw),
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-y',  # Overwrite output
        str(output_video_final)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Successfully encoded to: {output_video_final}")
        final_size = Path(output_video_final).stat().st_size
        print(f"  File size: {final_size / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ ffmpeg encoding failed with return code: {result.returncode}")
        print(f"  stderr: {result.stderr}")
        return None

    print()

    # Summary
    print("="*70)
    print("SPIKE TEST SUMMARY")
    print("="*70)
    print(f"Total frames processed: {frame_idx}")
    print(f"Total detections: {total_detections}")
    print(f"Unique track IDs: {len(tracks_data)}")
    print()

    if detections_by_class:
        print("Detections by class:")
        for class_name in sorted(detections_by_class.keys()):
            count = detections_by_class[class_name]
            print(f"  {class_name}: {count}")
    else:
        print("No detections (expected - salamanders not in pretrained model)")

    print()

    # Validate distances
    if distances:
        all_finite = all(np.isfinite(d) for d in distances.values())
        all_non_negative = all(d >= 0 for d in distances.values())

        print(f"Distance validation:")
        print(f"  All distances finite: {all_finite}")
        print(f"  All distances non-negative: {all_non_negative}")

    print("="*70)

    return {
        'frames_processed': frame_idx,
        'total_detections': total_detections,
        'unique_tracks': len(tracks_data),
        'detections_by_class': dict(detections_by_class),
        'distances': distances,
        'ffmpeg_return_code': result.returncode,
        'output_exists': Path(output_video_final).exists(),
        'output_size': Path(output_video_final).stat().st_size if Path(output_video_final).exists() else 0
    }


def main():
    input_video = Path("samples/ensatina_short.mp4")
    output_video_raw = Path("spike_output_raw.mp4")
    output_video_final = Path("spike_output.mp4")

    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return 1

    try:
        results = run_spike(input_video, output_video_raw, output_video_final)

        if results is None:
            print("\n✗ SPIKE TEST FAILED: ffmpeg encoding error")
            return 1

        # Validation checks
        success = True

        if results['ffmpeg_return_code'] != 0:
            print("\n✗ SPIKE TEST FAILED: ffmpeg returned non-zero")
            success = False

        if not results['output_exists']:
            print("\n✗ SPIKE TEST FAILED: output file does not exist")
            success = False

        if results['output_size'] == 0:
            print("\n✗ SPIKE TEST FAILED: output file is empty")
            success = False

        if results['distances']:
            if not all(np.isfinite(d) for d in results['distances'].values()):
                print("\n✗ SPIKE TEST FAILED: non-finite distances")
                success = False

            if not all(d >= 0 for d in results['distances'].values()):
                print("\n✗ SPIKE TEST FAILED: negative distances")
                success = False

        if success:
            print("\n✓ SPIKE TEST PASSED")
            print("\nNext steps:")
            print("1. Open spike_output.mp4 in Chrome/Safari to verify playback")
            print("2. If playback works, proceed to Day 4 (lock contracts.py and state.py)")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n✗ SPIKE TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
