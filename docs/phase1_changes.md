# Phase 1 — Preprocessing Pipeline

## Frame Selection Pipeline (New)

Storage-efficient keyframe selection so the AI detector only runs on a handful of representative frames:

```
Video → Ring Buffer → HashSet (dHash dedup) → LSH → Scene Graph → Representative Frames → AI Detector
```

### `src/preprocessing/frame_selection.py`
**New** — streaming frame selector.

- `dhash()` / `hamming_distance()` — perceptual hashing; near-duplicate frames collapse to one (static talking heads → 1 frame).
- `RingBuffer` — bounded streaming window (only recent frames kept in memory).
- `feature_vector()` — 8×8 RGB grid (192-d), intentionally **not** L2-normalised; similarity is distance-based (`_sim = exp(-mean_abs_diff * 4)`).
- `LSHIndex` — random-hyperplane LSH to find *similar non-adjacent* frames (e.g. camera returning to an earlier scene).
- `SceneGraph` — union-find over temporal + LSH edges; connected components = scenes.
- `FrameSelector` — `push(frame, video_frame_idx)` streaming API; `representatives()` returns the sharpest frame per scene, capped by `max_scenes`.
- `select_frames_from_video()` — convenience for one-shot selection.

`FrameSelectionConfig` defaults: hash_size=8, dedup_hamming=10, dedup_window=64, feature_dim=192, lsh_num_tables=4, lsh_num_projections=8, similarity_threshold=0.80, temporal_threshold=0.75, max_scenes=50, min_scene_frames=2.

### `src/preprocessing/preprocess_video.py` (modified)
- `PipelineConfig` gained `use_scene_selection: bool = True` and `selection_config: FrameSelectionConfig`.
- `process()` now streams the video through the `FrameSelector`, then runs face detection/alignment **only on representative frames** (re-read from the video by index). Legacy "process every frame" path still available via `use_scene_selection=False`.

**Result on the real 175 MB test video:** 13,681 frames sampled → 19 deduped → 4 representative scenes → 4 frames processed, 3 faces detected (vs ~4,560 frames in the old path).

### `tests/test_pipeline.py`
7 new frame-selection tests (dhash/hamming, ring-buffer eviction, LSH, static-scene collapse, distinct-scene separation, similar-scene merge, pipeline stores reps only) → **23 tests total, all passing**.

---

## Files Created

### `src/preprocessing/preprocess_video.py`
Main pipeline orchestrator. Ties frame extraction → face detection → face alignment into a single `VideoPreprocessor.process()` call.

**Key components:**
- `PipelineConfig` — configures all sub-stages (frame extraction, alignment, face detection confidence, model selection, HDF5 store)
- `VideoPreprocessor` — context manager that creates FrameExtractor, MediaPipeFaceDetector, and FaceAligner once, then loops over frames
- `process_video()` / `process_directory()` — convenience functions for single/batch processing

**Output structure** (per video):
```
data/processed/<video_name>/
    <video_name>.h5     (HDF5 store: compressed frames + aligned faces + metadata)
    metadata.json       (maps frames ↔ faces, bounding boxes, config)
```
Or in legacy JPEG mode (`use_hdf5_store=False`):
```
data/processed/<video_name>/
    frames/             (extracted frames, optional via save_frames=False)
    faces/              (aligned 299×299 face crops)
    metadata.json
```

### `src/preprocessing/frame_store.py`
**New** — HDF5-based efficient storage for frames and aligned faces.

**Why HDF5 instead of thousands of JPEG files?**
- **One file per video** instead of hundreds/thousands of files → zero filesystem overhead, trivial to move/archive
- **gzip compression** on raw arrays (compression_opts=4 = good balance of speed/size)
- **Metadata lives with pixel data** in the same file — one self-contained artifact per video
- **Chunked + buffered writes** — frames buffered in RAM (buffer_size=32) and flushed in batches, avoiding per-frame dataset resizes (~5x faster writes)
- **Fast random access** via chunked datasets

**File layout:**
```
<video_name>.h5
    frames/img        (N, H, W, 3) uint8 BGR frames
    faces/img         (M, H, W, 3) uint8 aligned face crops
    meta/frame_map    (M,) int — 0-based source frame index for each face
    meta/bboxes       (M, 4) int — x, y, w, h per face
    meta/confidences  (M,) float — detection confidence
    attrs: video_path, num_frames, num_faces, config...
```

**Key classes:** `StoreConfig` (compression, chunk_rows, buffer_size), `FrameStore` (add_frame/add_face/flush/close/get_* methods).

### `src/__init__.py`
Marks `src/` as a Python package.

### `src/preprocessing/__init__.py`
Exports all public symbols from preprocessing submodules for clean `from src.preprocessing import ...`.

### `src/models/__init__.py`
Exports `XceptionBackbone`.

### `tests/test_pipeline.py`
16 tests:
- 6 unit tests (dataclasses, FaceBox math, config serialization)
- 4 pipeline integration tests (synthetic video → HDF5 store, JPEG fallback, save_frames flag, batch processing)
- 3 FrameStore tests (add/read, buffered flush, attrs)
- 1 error-path test (nonexistent video → FileNotFoundError)

---

## Files Modified

### `src/preprocessing/face_detector.py`
Two changes:

**1. MediaPipe 1.0 API migration** (`mp.solutions` → `mp.tasks`):

| Before | After |
|--------|-------|
| `mp.solutions.face_detection.FaceDetection(...)` | `mp.tasks.vision.FaceDetector.create_from_options(...)` |
| `detector.process(rgb)` | `detector.detect(mp.Image(...))` |
| `det.score[0]`, `det.location_data.relative_bounding_box` | `det.categories[0].score`, `det.bounding_box` |

**2. Face detection accuracy fix** — `model_selection` was previously a bug:
- Before: the param was accepted but **never used** — the short-range (selfie) model was always downloaded, which underperforms on regular video.
- After: `model_selection=1` downloads the **full-range** model (better for arbitrary video footage); `model_selection=0` uses short-range (selfies). Default is now `1`.
- Added `fallback_to_other_model=True`: if the selected model finds nothing, it automatically retries with the other BlazeFace model before giving up.
- Both models auto-download and cache in `src/preprocessing/models/`.

### `src/preprocessing/face_alignment.py`
MediaPipe 1.0 migration for Face Mesh → FaceLandmarker:

| Before | After |
|--------|-------|
| `mp.solutions.face_mesh.FaceMesh(...)` | `mp.tasks.vision.FaceLandmarker.create_from_options(...)` |
| `self._mesh.process(rgb)` | `self._landmarker.detect(mp.Image(...))` |
| `results.multi_face_landmarks[0].landmark` | `results.face_landmarks[0]` |
| `min_detection_confidence` param | `min_face_detection_confidence` param |
| — | Added `min_tracking_confidence: float = 0.5` to `AlignmentConfig` |
| — | Model auto-downloads from GCS (face_landmarker.task) via `ensure_model_asset()` |

### `src/preprocessing/frame_extractor.py`
- Changed `from utils import ...` → `from .utils import ...` (relative imports for package structure).

### `src/preprocessing/preprocess_video.py`
- Added `use_hdf5_store`, `model_selection`, `store_config` to `PipelineConfig`.
- `process()` now writes frames/faces into an HDF5 `FrameStore` by default instead of individual JPEGs.
- Added `_save_store_attrs()` to record video/config metadata as HDF5 attributes.
- CLI gained `--no-h5`, `--model-selection`, `--min-confidence` flags.

### `src/preprocessing/__init__.py`
Exports `FrameStore`, `StoreConfig`.

### `requirements.txt`
Added `h5py>=3.10.0`.

---

## Dependencies Added
| Package | Version | Why |
|---------|---------|-----|
| h5py | >=3.10.0 | HDF5-based efficient frame/face storage |
| pytest | (dev) | Test runner |

## Models Downloaded
| Model | URL | Size | Used By |
|-------|-----|------|---------|
| BlazeFace (short-range) | `blaze_face_short_range.tflite` | ~280 KB | model_selection=0 |
| BlazeFace (full-range) | `blaze_face_full_range.tflite` | ~1 MB | model_selection=1 (default) |
| Face Landmarker | `face_landmarker.task` | ~8 MB | FaceAligner |

Cached at `src/preprocessing/models/` via `ensure_model_asset()`.

## How to Run

```bash
# Single video (default: full-range model + HDF5 store)
python -m src.preprocessing.preprocess_video data/raw/fake/video.mp4 --output-dir data/processed

# Batch process directory
python -c "
from src.preprocessing import process_directory
process_directory('data/raw', 'data/processed')
"

# Legacy JPEG output instead of HDF5
python -m src.preprocessing.preprocess_video video.mp4 --output-dir data/processed --no-h5

# Inspect an HDF5 store
python -c "
from src.preprocessing.frame_store import FrameStore
FrameStore.report_size('data/processed/video/video.h5')
"

# Tests
python -m pytest tests/ -v
```
