#!/usr/bin/env python3
"""
Trim video to specified duration using OpenCV.
"""
import argparse
import cv2
from pathlib import Path


def trim_video(input_path, output_path, duration_seconds):
    """
    Trim video to specified duration.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        duration_seconds: Duration in seconds to trim to
    """
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Calculate target frame count
    target_frames = int(duration_seconds * fps)

    print(f"Input video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print(f"Trimming to {duration_seconds} seconds ({target_frames} frames)...")

    # Create output video writer
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    while frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Reached end of video at frame {frame_count}")
            break

        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{target_frames} frames...")

    cap.release()
    out.release()

    print(f"\nTrimmed video saved to: {output_path}")
    print(f"Frames written: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description="Trim video to specified duration")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("-t", "--duration", type=float, default=30.0,
                       help="Duration in seconds (default: 30)")

    args = parser.parse_args()
    trim_video(args.input, args.output, args.duration)


if __name__ == "__main__":
    main()
