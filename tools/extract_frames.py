#!/usr/bin/env python3
"""
Extract frames from video files for YOLO training data.
"""
import argparse
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, stride=15):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        stride: Extract every Nth frame (default: 15)

    Returns:
        Tuple of (total_frames, extracted_frames)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    extracted_count = 0

    print(f"Processing video: {video_path}")
    print(f"Total frames in video: {total_frames}")
    print(f"Extracting every {stride} frame(s)...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame if it matches the stride
        if frame_idx % stride == 0:
            extracted_count += 1
            # Zero-padded filename: frame_00001.png
            output_file = output_path / f"frame_{extracted_count:05d}.png"
            cv2.imwrite(str(output_file), frame)

            if extracted_count % 100 == 0:
                print(f"  Extracted {extracted_count} frames...")

        frame_idx += 1

    cap.release()

    return total_frames, extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files for YOLO training data"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Extract every Nth frame (default: 15)"
    )

    args = parser.parse_args()

    # Extract frames
    total_frames, extracted_frames = extract_frames(
        args.video_path,
        args.output_dir,
        args.stride
    )

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total frames in video: {total_frames}")
    print(f"Frames extracted: {extracted_frames}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
