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
_MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models"


def ensure_model_asset(url: str, filename: str) -> Path:
    """Download a MediaPipe model asset to a local cache dir if not already
    present, and return its local path.

    Args:
        url: Direct download URL for the model file.
        filename: Local filename to cache it under (e.g. "blaze_face_short_range.tflite").
    """
    import urllib.request

    ensure_dir(_MODEL_CACHE_DIR)
    dest = _MODEL_CACHE_DIR / filename

    if dest.exists() and dest.stat().st_size > 0:
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
            f"locally afterwards under {_MODEL_CACHE_DIR})."
        ) from exc

    return dest


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