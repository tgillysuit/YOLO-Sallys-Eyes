# Salamander AI Tracking System

A web application that detects and tracks individual salamanders in uploaded video using a custom-trained YOLO model. Upload a clip through the browser interface and the system produces an annotated video with bounding boxes, assigns stable per-individual track IDs, and displays a per-salamander metrics table showing time on screen and total distance traveled.

---

## Features

- **Custom-trained YOLO salamander detection** — fine-tuned YOLO11n on 160 hand-labeled frames
- **Multi-object tracking** — ByteTrack assigns stable IDs across frames so each individual is tracked consistently
- **Annotated video output** — bounding boxes and track IDs rendered on every frame
- **Per-salamander time-on-screen metric** — seconds each individual appeared in the video
- **Per-salamander distance traveled metric** — total pixels traveled across consecutive bounding-box centers
- **Async processing with live progress bar** — background thread + polling so the browser stays responsive during the 30–60 s processing window

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, FastAPI, Ultralytics YOLO, OpenCV |
| Frontend | React (Vite) |
| Model | YOLO11n fine-tuned on custom salamander data |

---

## Dataset and Training

- Extracted **160 frames** from an 8-minute *Ensatina* salamander video (approximately 1 frame per 3 seconds)
- Labeled in **Label Studio** with bounding boxes — one class: `Salamander`
- Split **80/20 train/val** (128 train, 32 val)
- Trained **YOLO11n** for **50 epochs** at `imgsz=320`, `batch=8` on CPU
- Final validation **mAP50 = 0.995**, mAP50-95 = 0.972 — near-perfect on the held-out val set

**Known limitation:** the model was trained on a single recording under consistent conditions. Performance on video from different cameras, lighting environments, or salamander species may be lower. More diverse footage and additional labeled data would improve generalization.

---

## How to Run

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Mac/Linux
# source venv/Scripts/activate    # Windows Git Bash
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

---

## How to Use

1. Open **http://localhost:5173** in your browser
2. Click **Choose File** and select an `.mp4` video
3. Click **Upload** — the button disables and a progress bar climbs from 0–100%
4. When processing completes, the annotated video plays automatically with bounding boxes drawn on every detected salamander
5. A metrics table below the video shows each individual's track ID, label, time on screen (seconds), and total distance traveled (pixels)

---

## Color Masking vs. YOLO

Color masking is a straightforward approach that isolates objects by their hue — fast to implement, requires no training data, and works reliably when the subject is a distinctly different color from a clean, uniform background. In a controlled setup with stable lighting and a plain backdrop it can be entirely sufficient, and it would have detected our salamanders in the simplest frames of our footage where the animal was well-lit against an uncluttered surface.

The limitation of color masking becomes clear as soon as real-world complexity enters the picture. Even in our controlled footage, subtle lighting variation across the frame shifts the apparent hue of the salamander's skin, causing a fixed HSV threshold to fragment or drop detections. In natural field footage — rocks, leaf litter, soil — the background contains similar earth tones and the approach would produce a flood of false positives. Wet versus dry skin further shifts color, and two overlapping individuals are indistinguishable.

YOLO handles these harder cases because it learns spatial features — shape, texture, edge patterns — rather than relying on a single color signal. Paired with ByteTrack it assigns each individual a stable ID across frames even when animals briefly leave the frame or cross paths, something color masking alone cannot do. The trade-off is real: YOLO requires labeled training data and time to prepare it, whereas color masking can be tuned in minutes. For a controlled lab setup where labeling cost matters, color masking is a reasonable choice. For variable field conditions and per-individual identity, YOLO is the appropriate tool.

---

## Known Limitations

- **Single-video training data** — the model is tuned to one recording's conditions; generalization to new environments is untested
- **Distance in pixels only** — `distance_px` has no real-world calibration. Converting to cm/m would require a known reference object (e.g. a ruler) in frame to establish a px-per-mm ratio
- **One job at a time** — submitting a new video while one is processing overwrites the previous job; a job queue would be needed for concurrent use
- **Track ID swaps** — when two salamanders cross or overlap, ByteTrack can occasionally swap their IDs for a few frames before recovering
