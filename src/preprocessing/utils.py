"""
utils.py

Shared helper functions used across the Phase 1 preprocessing pipeline:
- logging setup
- filesystem helpers
- blur detection
- frame enhancement (sharpening / deblurring) for blurry-but-usable frames

These utilities are deliberately dependency-light (OpenCV + NumPy only) so that
they can be reused by frame_extractor.py, face_detector.py, and
face_alignment.py without introducing circular imports.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, os.PathLike]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module-level logger.

    Safe to call multiple times (e.g. once per module) without producing
    duplicate log handlers/messages.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: PathLike) -> Path:
    """Create a directory (including parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_videos(directory: PathLike, extensions=(".mp4", ".avi", ".mov", ".mkv")) -> list[Path]:
    """Return all video files under `directory` matching the given extensions."""
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in extensions and p.is_file()
    )


# --------------------------------------------------------------------------- #
# MediaPipe model asset caching
# --------------------------------------------------------------------------- #
# As of mediapipe 0.10.x, the modern "Tasks" API (used here for face detection
# and face landmarking) requires downloading small .tflite / .task model
# files rather than bundling them in the package. This helper caches them
# locally on first use so subsequent runs don't re-download.
#
# When frozen into an .exe (PyInstaller), ``__file__`` points at the temporary
# extraction dir, so we keep the cache in a user-writable location instead and
# seed it from the bundled copy shipped inside the executable.
def _get_model_cache_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        return base / "DeepFakeDetector" / "models"
    return Path(__file__).resolve().parent / "models"


def _seed_from_bundle(filename: str, dest: Path) -> bool:
    """Copy a model that was bundled inside the frozen executable (if any)."""
    if not getattr(sys, "frozen", False):
        return False
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "models" / filename
    if bundled.exists() and bundled.stat().st_size > 0:
        shutil.copy2(bundled, dest)
        return True
    return False


def ensure_model_asset(url: str, filename: str) -> Path:
    """Download a MediaPipe model asset to a local cache dir if not already
    present, and return its local path.

    Args:
        url: Direct download URL for the model file.
        filename: Local filename to cache it under (e.g. "blaze_face_short_range.tflite").
    """
    import urllib.request

    cache_dir = _get_model_cache_dir()
    ensure_dir(cache_dir)
    dest = cache_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if _seed_from_bundle(filename, dest):
        logger = get_logger("utils.ensure_model_asset")
        logger.info("Using bundled model asset: %s -> %s", filename, dest)
        return dest

    logger = get_logger("utils.ensure_model_asset")
    logger.info("Downloading MediaPipe model asset: %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download required MediaPipe model asset from {url}. "
            "This requires internet access on first run (models are cached "
            f"locally afterwards under {cache_dir})."
        ) from exc

    return dest


def open_video_capture(path: PathLike):
    """Open a video for reading, preferring a hardware-accelerated (NVDEC)
    decode backend when the OpenCV build supports it.

    Software frame-by-frame decoding of a long video is the dominant cost in the
    selection pass, so enabling ``VIDEO_ACCELERATION_ANY`` (which lets OpenCV use
    a GPU decode surface) typically yields a large speedup with no code changes
    downstream. Falls back gracefully if the backend/flag is unavailable.
    """
    import cv2

    cap = cv2.VideoCapture(str(path), getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION,
                    getattr(cv2, "VIDEO_ACCELERATION_ANY", -1))
        except Exception:
            pass
    return cap


def iter_sampled_frames(path: PathLike, target_fps: float, preview_width: int = 256):
    """Yield ``(video_frame_idx, bgr_image)`` sampled at ``target_fps``.

    Fast path: pipes the video through a bundled FFmpeg, which decodes and scales
    to ``preview_width`` off the main thread and drops to the target frame rate —
    this is several times faster than frame-by-frame OpenCV decoding of a long,
    high-resolution clip. If FFmpeg is unavailable, raises ``RuntimeError`` so the
    caller can fall back to a pure-OpenCV scanner.

    The yielded frames are BGR (uint8) so they feed the selector unchanged.
    """
    from imageio_ffmpeg import get_ffmpeg_exe, get_ffmpeg_version  # raises if absent

    cap = cv2.VideoCapture(str(path))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError("Could not read video dimensions")

    # Keep aspect ratio; ensure even dimensions (ffmpeg requires it).
    new_h = int(round(src_h * preview_width / src_w))
    new_h = (new_h // 2) * 2
    out_w = (preview_width // 2) * 2

    ff = get_ffmpeg_exe()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vf", f"fps={target_fps},scale={out_w}:{new_h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = out_w * new_h * 3
    buf = bytearray()
    n = 0
    try:
        while True:
            chunk = proc.stdout.read(1 << 22)
            if not chunk:
                break
            buf.extend(chunk)
            while len(buf) >= frame_size:
                arr = np.frombuffer(buf[:frame_size], np.uint8).reshape(new_h, out_w, 3).copy()
                buf = buf[frame_size:]
                n += 1
                vidx = round((n - 1) * (src_fps / target_fps))
                yield vidx, arr
    finally:
        proc.stdout.close()
        proc.wait()


def iter_sampled_frames_cv2(path: PathLike, target_fps: float, preview_width: int = 256):
    """Pure-OpenCV fallback for iter_sampled_frames (no FFmpeg dependency)."""
    cap = open_video_capture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_h = int(round(src_h * preview_width / src_w))
        new_h = (new_h // 2) * 2
        out_w = (preview_width // 2) * 2
        interval = max(1, round(src_fps / target_fps))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                small = cv2.resize(frame, (out_w, new_h), interpolation=cv2.INTER_AREA)
                yield idx, small
            idx += 1
    finally:
        cap.release()


def sample_frames(path: PathLike, target_fps: float, preview_width: int = 256):
    """Sample frames at ``target_fps``; prefer FFmpeg, fall back to OpenCV."""
    try:
        return iter_sampled_frames(path, target_fps, preview_width)
    except Exception:
        return iter_sampled_frames_cv2(path, target_fps, preview_width)


# --------------------------------------------------------------------------- #
# Blur detection
# --------------------------------------------------------------------------- #
def variance_of_laplacian(image: np.ndarray) -> float:
    """Compute the variance of the Laplacian of an image.

    This is a standard, cheap focus-measure: sharp images have a wide spread
    of intensities in the Laplacian (edges), so the variance is high; blurry
    images are smoother and the variance is low.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Return True if the image's focus measure falls below `threshold`.

    The default threshold (100.0) is a commonly used starting point for the
    variance-of-Laplacian measure on natural video frames; it should be
    tuned per-dataset/camera in practice.
    """
    return variance_of_laplacian(image) < threshold


def is_severely_corrupted(image: np.ndarray, threshold: float = 15.0) -> bool:
    """Return True only for frames that are essentially unusable.

    This is a much stricter (lower) threshold than `is_blurry`. Per the
    pipeline spec, moderately blurry frames should be *enhanced* rather than
    discarded; only frames this severely degraded (or unreadable) should be
    dropped outright.
    """
    if image is None or image.size == 0:
        return True
    return variance_of_laplacian(image) < threshold


# --------------------------------------------------------------------------- #
# Frame enhancement (sharpening / light deblurring)
# --------------------------------------------------------------------------- #
def sharpen_frame(image: np.ndarray, amount: float = 1.5, radius: int = 3) -> np.ndarray:
    """Apply an unsharp-mask sharpening filter to a frame.

    This is a fast, general-purpose way to recover perceived detail in
    mildly blurry frames without the cost of a full deblurring model.

    Args:
        image: BGR frame as read by OpenCV.
        amount: Strength of the sharpening effect.
        radius: Gaussian blur kernel radius used to build the unsharp mask.
    """
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def deblur_frame(image: np.ndarray) -> np.ndarray:
    """Attempt a lightweight deblurring pass using Wiener-style deconvolution
    approximated via a Laplacian-sharpening + denoise combo.

    A full blind-deconvolution or learned deblurring model (e.g. DeblurGAN)
    is out of scope for Phase 1's classical CV pipeline, but this function
    provides a placeholder-quality restoration step that can later be
    swapped for a learned model without changing the calling code.
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    return sharpen_frame(denoised, amount=1.2, radius=2)


def enhance_frame(image: np.ndarray, blur_threshold: float = 100.0) -> np.ndarray:
    """Enhance a frame if it is blurry; otherwise return it unchanged.

    This is the single entry point `frame_extractor.py` should call: it
    decides whether enhancement is needed and applies the appropriate
    restoration technique.
    """
    if is_blurry(image, threshold=blur_threshold):
        return deblur_frame(image)
    return image


# --------------------------------------------------------------------------- #
# Image resize helper
# --------------------------------------------------------------------------- #
def resize_with_pad(image: np.ndarray, target_size: int = 299) -> np.ndarray:
    """Resize an image to (target_size, target_size), padding to preserve
    aspect ratio instead of distorting the face via a naive stretch resize.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas