"""
frame_store.py

Efficient frame/face storage using HDF5 (h5py).

Why HDF5 instead of thousands of JPEG files?
  - One file instead of hundreds/thousands of files → no per-file filesystem
    overhead, simpler to move/archive.
  - gzip compression is applied on the raw arrays, giving good size reduction
    with no extra library dependencies (h5py bundles zlib).
  - Metadata (bboxes, config, video info) lives alongside the pixel data in
    the same file, so everything about a video is one artifact.
  - Chunked storage gives fast random access to individual frames.

Layout of a store file:
    /
    |-- frames/          group
    |   `-- img          (N, H, W, 3) uint8 BGR frames
    |-- faces/           group (only if save_faces)
    |   `-- img          (M, H, W, 3) uint8 aligned face crops
    |-- meta/            group
    |   `-- frame_map    (N,) int: 1-based index of source frame for each face
    |   `-- bboxes       (M, 4) int: x, y, w, h of each face
    |   `-- confidences  (M,) float: detection confidence
    |-- attrs: video_path, source_fps, target_fps, config, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .face_detector import FaceBox
from .utils import PathLike, ensure_dir, get_logger

logger = get_logger(__name__)

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


@dataclass
class StoreConfig:
    compression: str = "gzip"
    compression_opts: int = 4      # 0 (fast) - 9 (small); 4 is a good balance
    chunk_rows: int = 64           # chunk size for frames dataset
    max_frames: int = 10000        # preallocation hint to avoid repeated resizes
    buffer_size: int = 32          # frames/faces buffered in RAM before disk flush


class FrameStore:
    """HDF5-backed store for frames and aligned faces of one video.

    Frames are buffered in memory and written to disk in batches of
    ``buffer_size`` rows. This avoids the cost of resizing the HDF5 dataset
    on every single frame (each resize can trigger a disk operation).
    """

    def __init__(
        self,
        path: PathLike,
        config: Optional[StoreConfig] = None,
        max_faces: int = 10000,
        overwrite: bool = False,
    ):
        if h5py is None:
            raise ImportError(
                "h5py is required for FrameStore. Install with: pip install h5py"
            ) from _H5PY_IMPORT_ERROR

        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.config = config or StoreConfig()
        self.max_faces = max_faces

        # ``overwrite=True`` truncates any pre-existing store so re-running the
        # pipeline on the same video doesn't stack duplicate frames/faces on top
        # of the previous run (which would otherwise look like duplicates).
        file_mode = "w" if overwrite else "a"
        self._f = h5py.File(str(self.path), file_mode)
        self._frames = self._init_dataset(
            "frames/img",
            shape=(0,) + (0, 0, 3),
            maxshape=(self.config.max_frames,) + (None, None, 3),
        )
        self._faces = self._init_dataset(
            "faces/img",
            shape=(0,) + (0, 0, 3),
            maxshape=(max_faces,) + (None, None, 3),
        )
        self._frame_map = self._init_dataset(
            "meta/frame_map", shape=(0,), maxshape=(max_faces,)
        )
        self._bboxes = self._init_dataset(
            "meta/bboxes", shape=(0, 4), maxshape=(max_faces, 4)
        )
        self._confidences = self._init_dataset(
            "meta/confidences", shape=(0,), maxshape=(max_faces,)
        )

        self._n_frames = self._frames.shape[0]
        self._n_faces = self._faces.shape[0]

        # In-memory buffers
        self._frame_buf: list[np.ndarray] = []
        self._face_buf: list[np.ndarray] = []
        self._frame_map_buf: list[int] = []
        self._bbox_buf: list[list[int]] = []
        self._conf_buf: list[float] = []
        self._frame_idx_offset = self._n_frames
        self._face_idx_offset = self._n_faces

    def _init_dataset(self, name: str, shape, maxshape, dtype=None):
        if name in self._f:
            return self._f[name]

        if dtype is None:
            dtype = np.uint8 if "img" in name else np.int64 if "frame_map" in name else np.float64

        # Chunk shape: fixed for the spatial dims. For the append axis we use
        # chunk_rows; for the remaining (unknown) spatial dims we pick 64.
        rest = tuple(64 if d is None else d for d in maxshape[1:])
        chunk = (min(self.config.chunk_rows, maxshape[0] or self.config.chunk_rows),) + rest
        return self._f.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            compression=self.config.compression,
            compression_opts=self.config.compression_opts,
            chunks=chunk,
        )

    # ------------------------------------------------------------------ #
    def add_frame(self, frame: np.ndarray) -> int:
        """Append a frame to the buffer. Returns its 0-based index."""
        idx = self._n_frames
        self._frame_buf.append(frame)
        self._n_frames += 1
        self._maybe_flush_frames()
        return idx

    def add_face(
        self,
        face: np.ndarray,
        bbox: Optional[FaceBox] = None,
        frame_idx: Optional[int] = None,
        confidence: float = 0.0,
    ) -> int:
        """Append an aligned face crop to the buffer. Returns its 0-based index."""
        idx = self._n_faces
        self._face_buf.append(face)
        self._frame_map_buf.append(frame_idx if frame_idx is not None else -1)
        self._bbox_buf.append(
            [bbox.x, bbox.y, bbox.width, bbox.height] if bbox is not None else [-1, -1, -1, -1]
        )
        self._conf_buf.append(float(confidence))
        self._n_faces += 1
        self._maybe_flush_faces()
        return idx

    # ------------------------------------------------------------------ #
    def _maybe_flush_frames(self) -> None:
        if len(self._frame_buf) >= self.config.buffer_size:
            self.flush()

    def _maybe_flush_faces(self) -> None:
        if len(self._face_buf) >= self.config.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write any buffered frames/faces to disk in one batch."""
        if self._frame_buf:
            frames = np.stack(self._frame_buf)
            n = len(frames)
            start = self._frame_idx_offset
            self._frames.resize((start + n,) + frames.shape[1:])
            self._frames[start : start + n] = frames
            self._frame_buf = []
            self._frame_idx_offset = start + n

        if self._face_buf:
            faces = np.stack(self._face_buf)
            n = len(faces)
            start = self._face_idx_offset
            self._faces.resize((start + n,) + faces.shape[1:])
            self._faces[start : start + n] = faces
            self._frame_map.resize((start + n,))
            self._frame_map[start : start + n] = self._frame_map_buf
            self._bboxes.resize((start + n, 4))
            self._bboxes[start : start + n] = self._bbox_buf
            self._confidences.resize((start + n,))
            self._confidences[start : start + n] = self._conf_buf
            self._face_buf = []
            self._frame_map_buf = []
            self._bbox_buf = []
            self._conf_buf = []
            self._face_idx_offset = start + n

        self._f.flush()

    # ------------------------------------------------------------------ #
    def set_attrs(self, attrs: Dict) -> None:
        for k, v in attrs.items():
            try:
                self._f.attrs[k] = v
            except TypeError:
                logger.debug("Skipping attr %s=%r (not h5py-serializable)", k, v)

    def get_attrs(self) -> Dict:
        return dict(self._f.attrs)

    @property
    def num_frames(self) -> int:
        return self._n_frames

    @property
    def num_faces(self) -> int:
        return self._n_faces

    def get_frames(self) -> np.ndarray:
        self.flush()
        return self._frames[: self._n_frames]

    def get_faces(self) -> np.ndarray:
        self.flush()
        return self._faces[: self._n_faces]

    def get_frame_map(self) -> np.ndarray:
        self.flush()
        return self._frame_map[: self._n_faces]

    def get_bboxes(self) -> np.ndarray:
        self.flush()
        return self._bboxes[: self._n_faces]

    def get_confidences(self) -> np.ndarray:
        self.flush()
        return self._confidences[: self._n_faces]

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.flush()
        self._f.close()

    def __enter__(self) -> "FrameStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n_frames

    @classmethod
    def report_size(cls, path: PathLike) -> None:
        """Print a quick summary of what's in a store file (debug helper)."""
        import h5py

        with h5py.File(str(path), "r") as f:
            def _walk(g, indent=0):
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        print(" " * indent + f"{k}: shape={v.shape} dtype={v.dtype}")
                    else:
                        print(" " * indent + f"{k}/")
                        _walk(v, indent + 2)

            print(f"Store: {path}")
            print(f"  attrs: {dict(f.attrs)}")
            _walk(f)
