import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread

import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

VIDEOS_DIR = Path(__file__).parent / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Salamander Tracker POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

model = YOLO("best.pt")

job = {"status": "idle"}


def run_track_job():
    try:
        input_path = VIDEOS_DIR / "input.mp4"

        # FIX 2: Reset ByteTrack state so track IDs restart from 1 each upload
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor = None

        # Read video metadata
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"fps={fps} dims={width}x{height} frames={total}", flush=True)

        # Write mp4v to a temp file, then transcode to H.264 for browser playback
        raw_path = VIDEOS_DIR / "output_raw.mp4"
        output_path = VIDEOS_DIR / "output.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        # Per-track metrics accumulators
        frames_seen = defaultdict(int)
        label_for = {}
        last_center = {}          # track_id -> (cx, cy) of previous frame
        distance_px = defaultdict(float)  # track_id -> total pixels traveled

        # Frame loop — FIX 3: raise conf to 0.4 to reduce false positives
        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, verbose=False, conf=0.4)[0]
            writer.write(result.plot())
            job["percent"] = int((frame_idx + 1) / total * 100)
            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}/{total}", flush=True)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for tid, cls_id, xyxy in zip(
                    boxes.id.tolist(), boxes.cls.tolist(), boxes.xyxy.tolist()
                ):
                    tid = int(tid)
                    frames_seen[tid] += 1
                    label_for[tid] = model.names[int(cls_id)]

                    # Distance tracking: center of bounding box
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if tid in last_center:
                        last_cx, last_cy = last_center[tid]
                        dist = math.hypot(cx - last_cx, cy - last_cy)
                        if dist <= 200:  # ignore jumps — likely tracking errors
                            distance_px[tid] += dist
                    last_center[tid] = (cx, cy)

        cap.release()
        writer.release()

        # FIX 1: Transcode mp4v -> H.264 so browsers can play it natively
        print("Transcoding to H.264...", flush=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(raw_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        raw_path.unlink(missing_ok=True)  # clean up the intermediate file
        print("Transcode done.", flush=True)

        tracks = [
            {
                "track_id": tid,
                "time_on_screen_s": round(frames_seen[tid] / fps, 2),
                "distance_px": round(distance_px[tid], 1),
                "label": label_for[tid],
            }
            for tid in frames_seen
        ]

        job.clear()
        job["status"] = "done"
        job["percent"] = 100
        job["result"] = {
            "video_url": f"http://localhost:8000/videos/output.mp4?t={int(time.time())}",
            "tracks": tracks,
        }

    except Exception as e:
        print(f"error: {e}", flush=True)
        job.clear()
        job["status"] = "error"
        job["message"] = str(e)


@app.get("/")
def root():
    return {"ok": True}


@app.post("/track")
def start_track(video: UploadFile = File(...)):
    (VIDEOS_DIR / "input.mp4").write_bytes(video.file.read())
    job.clear()
    job["status"] = "processing"
    job["percent"] = 0
    Thread(target=run_track_job, daemon=True).start()
    return {"status": "processing"}


@app.get("/track")
def get_track():
    return job


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
