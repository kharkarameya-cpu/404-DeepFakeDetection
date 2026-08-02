"""
frame_selection.py

Storage-efficient frame selection pipeline:

    Video
       |
       v
    Ring Buffer          - bounded streaming window (only recent frames in memory)
       |
       v
    HashSet (dedup)      - dHash perceptual hashing; skip near-duplicate frames
       |
       v
    LSH / HNSW           - locality-sensitive hashing finds *similar* (non-adjacent) frames
       |
       v
    Scene Graph          - union-find over temporal + LSH edges; connected comps = scenes
       |
       v
    Representative Frames- one frame per scene (sharpest), i.e. the keyframes
       |
       v
    AI Detector          - face detection/alignment runs ONLY on representatives

Why this saves storage & compute:
  - We never keep all frames in memory (ring buffer).
  - Near-duplicate frames (e.g. a static talking head) collapse to ONE frame.
  - The heavy AI work runs only on a handful of representative frames per video
    instead of every sampled frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Perceptual hashing (dHash)
# --------------------------------------------------------------------------- #
def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash: hash_size*(hash_size+1) bits derived from horizontal
    gradient sign. Robust to brightness changes; small frames are enough."""
    gray = image if image.ndim == 2 else image[:, :, 0]
    resized = _resize_gray(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    return sum(1 << i for i, b in enumerate(bits) if b)


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _resize_gray(gray: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def feature_vector(image: np.ndarray, dim: int = 192) -> np.ndarray:
    """Compact descriptor for LSH/scene comparison.

    Downscales to a small RGB grid (color matters for scene detection) and
    normalises each channel pixel to [0,1]. We deliberately do NOT L2-normalise
    the whole vector: normalising makes flat (constant-colour) frames all point
    in the same direction, which would merge unrelated scenes. Similarity is
    instead measured with a distance-based function on the raw grid.
    """
    import cv2

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = 3
    side = int(round(np.sqrt(dim / channels)))
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(-1) / 255.0


def sharpness(image: np.ndarray) -> float:
    """Laplacian variance — used to pick the sharpest frame of a scene."""
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# Ring buffer
# --------------------------------------------------------------------------- #
class RingBuffer:
    """Bounded deque. When full, pushing evicts the oldest item."""

    def __init__(self, capacity: int):
        self._d: deque = deque(maxlen=capacity)

    def push(self, item) -> Optional[object]:
        """Add an item. Returns the evicted item, or None."""
        evicted = None
        if len(self._d) == self._d.maxlen:
            evicted = self._d[0]
        self._d.append(item)
        return evicted

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, item) -> bool:
        return item in self._d


# --------------------------------------------------------------------------- #
# LSH index (random hyperplanes)
# --------------------------------------------------------------------------- #
class LSHIndex:
    """Minimal locality-sensitive hash for finding similar feature vectors.

    Builds ``num_tables`` hash tables, each with ``num_projections`` random
    hyperplanes. A vector is hashed by the sign of its dot product with each
    plane. Similar vectors collide with high probability.
    """

    def __init__(self, dim: int, num_tables: int = 4, num_projections: int = 8, seed: int = 42):
        rng = np.random.RandomState(seed)
        self._tables = []
        self._buckets = []
        for _ in range(num_tables):
            planes = rng.randn(num_projections, dim)
            self._tables.append(planes)
            self._buckets.append(dict())

    def _bucket_key(self, vec: np.ndarray, table_idx: int) -> Tuple[int, ...]:
        proj = self._tables[table_idx] @ vec
        return tuple(int(b) for b in (proj > 0))

    def insert(self, key: int, vec: np.ndarray) -> None:
        for t, buckets in enumerate(self._buckets):
            k = self._bucket_key(vec, t)
            buckets.setdefault(k, set()).add(key)

    def query(self, vec: np.ndarray) -> Set[int]:
        candidates: Set[int] = set()
        for t, buckets in enumerate(self._buckets):
            k = self._bucket_key(vec, t)
            if k in buckets:
                candidates |= buckets[k]
        return candidates


# --------------------------------------------------------------------------- #
# Scene graph (union-find)
# --------------------------------------------------------------------------- #
class SceneGraph:
    """Union-find over frame indices. Edges connect frames that belong to the
    same scene. Connected components = scenes."""

    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def components(self) -> List[List[int]]:
        comps: Dict[int, List[int]] = {}
        for i in range(len(self._parent)):
            comps.setdefault(self.find(i), []).append(i)
        return sorted(comps.values(), key=lambda c: c[0])


# --------------------------------------------------------------------------- #
# Configuration + selector
# --------------------------------------------------------------------------- #
@dataclass
class FrameSelectionConfig:
    hash_size: int = 8
    dedup_hamming: int = 10        # frames closer than this (bits) are duplicates
    dedup_window: int = 64         # ring buffer size for the dedup window
    feature_dim: int = 192         # 8x8 RGB grid (3 channels)
    lsh_num_tables: int = 4
    lsh_num_projections: int = 8
    lsh_seed: int = 42
    similarity_threshold: float = 0.80   # sim above this => same scene (LSH edge)
    temporal_threshold: float = 0.75     # consecutive-frame sim cutoff
    max_scenes: int = 50                 # cap on representative frames per video
    min_scene_frames: int = 2            # ignore 1-frame "scenes" (noise) unless forced


@dataclass
class SelectedFrame:
    index: int          # 0-based index within the accepted (post-dedup) list
    video_frame_idx: int  # absolute frame index in the source video
    scene_id: int
    score: float
    image: Optional[np.ndarray] = None


class FrameSelector:
    """Streams frames, deduplicates, clusters into scenes, and picks one
    representative frame per scene.

    Usage:
        sel = FrameSelector(config)
        sel.push(frame, video_frame_idx)      # for every sampled frame
        reps = sel.representatives()          # at the end, get keyframes
    """

    def __init__(self, config: Optional[FrameSelectionConfig] = None):
        self.config = config or FrameSelectionConfig()
        self._ring = RingBuffer(self.config.dedup_window)
        self._hashes: List[int] = []
        self._features: List[np.ndarray] = []
        self._sharpness: List[float] = []
        self._video_indices: List[int] = []

    # ------------------------------------------------------------------ #
    def push(self, image: np.ndarray, video_frame_idx: int) -> bool:
        """Process one frame. Returns True if the frame was accepted
        (kept as a scene candidate), False if it was dropped as a duplicate."""
        h = dhash(image, self.config.hash_size)

        # Dedup: compare against recent frames in the ring window.
        for prev_h in self._ring:
            if hamming_distance(h, prev_h) <= self.config.dedup_hamming:
                return False

        self._ring.push(h)
        self._hashes.append(h)
        self._features.append(feature_vector(image, self.config.feature_dim))
        self._sharpness.append(sharpness(image))
        self._video_indices.append(video_frame_idx)
        return True

    # ------------------------------------------------------------------ #
    def representatives(self) -> List[SelectedFrame]:
        """Build the scene graph and return one representative per scene."""
        n = len(self._features)
        if n == 0:
            return []

        graph = SceneGraph(n)

        # Temporal edges: consecutive frames belonging to the same scene.
        for i in range(n - 1):
            if _sim(self._features[i], self._features[i + 1]) >= self.config.temporal_threshold:
                graph.union(i, i + 1)

        # LSH edges: non-adjacent similar frames (e.g. camera returns to a scene).
        lsh = LSHIndex(
            self.config.feature_dim,
            num_tables=self.config.lsh_num_tables,
            num_projections=self.config.lsh_num_projections,
            seed=self.config.lsh_seed,
        )
        for i, vec in enumerate(self._features):
            for j in lsh.query(vec):
                if j != i and _sim(self._features[i], self._features[j]) >= self.config.similarity_threshold:
                    graph.union(i, j)
            lsh.insert(i, vec)

        components = graph.components()
        # Drop singleton scenes unless we'd end up with no representatives.
        scenes = [c for c in components if len(c) >= self.config.min_scene_frames]
        if not scenes:
            scenes = components

        # Keep at most max_scenes — the largest scenes first (most informative).
        scenes.sort(key=len, reverse=True)
        scenes = scenes[: self.config.max_scenes]

        representatives: List[SelectedFrame] = []
        for scene_id, comp in enumerate(sorted(scenes, key=lambda c: c[0])):
            # Pick the sharpest frame in the scene.
            best = max(comp, key=lambda i: self._sharpness[i])
            representatives.append(
                SelectedFrame(
                    index=best,
                    video_frame_idx=self._video_indices[best],
                    scene_id=scene_id,
                    score=self._sharpness[best],
                )
            )
        return representatives

    @property
    def num_accepted(self) -> int:
        return len(self._features)


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity in [0, 1] from mean-absolute distance between two feature
    grids. Lower distance -> higher similarity. Uses exponential decay so the
    scale is sensitive around small differences."""
    dist = float(np.mean(np.abs(a - b)))
    return float(np.exp(-dist * 4.0))


def select_frames_from_video(
    video_path,
    config: Optional[FrameSelectionConfig] = None,
    target_fps: float = 7.5,
    max_frames: int = 500,
) -> List[SelectedFrame]:
    """Convenience: stream a video through the selector, return representatives.

    Frames are sampled at ``target_fps``. The heavy lifting (reading + hashing)
    happens here, but no full-resolution frames are kept in memory.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    selector = FrameSelector(config or FrameSelectionConfig())
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, round(source_fps / target_fps))

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                selector.push(frame, frame_idx)
            frame_idx += 1
    finally:
        cap.release()

    return selector.representatives()
