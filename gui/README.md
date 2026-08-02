# DeepFakeDetector — Test GUI

A small desktop GUI (tkinter) that lets you visually verify the Phase 1
preprocessing pipeline: pick a video, run it, and inspect the representative
frames + aligned face crops that the AI detector would be trained on.

## Features

- **Scene-based frame selection** by default — the pipeline samples frames, runs
  them through `RingBuffer → dHash dedup → LSH → Scene Graph` and only keeps one
  (sharpest) representative frame per scene.
- Runs the **AI detector (face detection + alignment) only on representatives**.
- Shows representative frames with face bounding boxes, and the aligned 299×299
  face crops, in a scrollable thumbnail view.
- A live log pane streams all pipeline output (including MediaPipe logs).
- Background processing — the UI stays responsive while a video runs.

## Run from source

```bat
pip install -r requirements.txt -r requirements-gui.txt
python -m gui.app
```

## Build a standalone .exe

The app is built as **`--onedir`** (a single `dist\DeepFakeDetectorTester\DeepFakeDetectorTester.exe`
plus a small `_internal` folder). This is on purpose: a `--onefile` exe packs ~340 MB into itself and
**re-extracts that archive on every launch (20-30s of waiting)** — unacceptable for a tool you run
repeatedly. The onedir folder only unpacks once, so **startup is instant** after the first build.

Bundled inside are Mediapipe (~340 MB) and a static FFmpeg binary (~50 MB) used for fast
hardware-friendly video sampling, so the whole thing runs with no Python install and no internet
(first run seeds MediaPipe's small model assets into `%LOCALAPPDATA%\DeepFakeDetector\models`).

```bat
gui\build_exe.bat
```

Output: `dist\DeepFakeDetectorTester\DeepFakeDetectorTester.exe`
Requirements: `pip install -r requirements.txt -r requirements-gui.txt`

### Verify the frozen exe offline

```bat
dist\DeepFakeDetectorTester\DeepFakeDetectorTester.exe --selftest data\raw\real\testvideo.mp4 out.txt
```

prints `OK frames=N faces=M` to `out.txt` (and exits `0`) if the frozen bundle
is healthy. Run it twice — the second run is warm because the model assets are
seeded to `%LOCALAPPDATA%` once.

## Using the GUI

1. Click **Select Video...** and pick a `.mp4`/etc.
2. Adjust **Target FPS**, **Model**, and toggle **Scene selection** if you want
   to fall back to processing every sampled frame.
3. Click **Process** — watch the log at the bottom.
4. When done, the **Representative Frames** and **Aligned Faces** tabs populate
   with thumbnails. Use the mouse wheel to scroll.

Output is written to `<user>\AppData\Local\DeepFakeDetector\data\processed\`
(when running the frozen exe) or `./data/processed/` (when running from source).
