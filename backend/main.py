import time
from collections import defaultdict
from pathlib import Path

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


@app.get("/")
def root():
    return {"ok": True}


@app.post("/track")
async def track(video: UploadFile = File(...)):
    # Save upload
    input_path = VIDEOS_DIR / "input.mp4"
    input_path.write_bytes(await video.read())

    # Read video metadata
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps={fps} dims={width}x{height} frames={total}")

    # Set up output writer
    output_path = VIDEOS_DIR / "output.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Per-track metrics accumulators
    frames_seen = defaultdict(int)
    label_for = {}

    # Frame loop
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, verbose=False)[0]
        writer.write(result.plot())
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}/{total}")

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for tid, cls_id in zip(boxes.id.tolist(), boxes.cls.tolist()):
                frames_seen[int(tid)] += 1
                label_for[int(tid)] = model.names[int(cls_id)]

    cap.release()
    writer.release()

    tracks = [
        {
            "track_id": tid,
            "time_on_screen_s": round(count / fps, 2),
            "label": label_for[tid],
        }
        for tid, count in frames_seen.items()
    ]

    return {
        "status": "done",
        "video_url": f"http://localhost:8000/videos/output.mp4?t={int(time.time())}",
        "tracks": tracks,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
