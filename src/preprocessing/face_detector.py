from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .utils import ensure_model_asset, get_logger

logger = get_logger(__name__)

_BLAZEFACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_BLAZEFACE_MODEL_FILENAME = "blaze_face_short_range.tflite"


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def area(self) -> int:
        return self.width * self.height


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceBox]:
        raise NotImplementedError

    def detect_primary(self, image: np.ndarray) -> Optional[FaceBox]:
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area() * f.confidence)


class MediaPipeFaceDetector(BaseFaceDetector):
    """Face detector backed by MediaPipe Tasks API (v1.0+)."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for MediaPipeFaceDetector. "
                "Install with: pip install mediapipe"
            ) from exc

        model_url = (
            _BLAZEFACE_MODEL_URL
        )
        model_path = ensure_model_asset(model_url, _BLAZEFACE_MODEL_FILENAME)

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self.min_confidence = min_confidence

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        import mediapipe as mp
        import cv2

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        boxes: List[FaceBox] = []
        if not results.detections:
            return boxes

        for det in results.detections:
            score = det.categories[0].score
            if score < self.min_confidence:
                continue
            bbox = det.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            box_w = min(w - x, bbox.width)
            box_h = min(h - y, bbox.height)
            if box_w <= 0 or box_h <= 0:
                continue
            boxes.append(FaceBox(x=x, y=y, width=box_w, height=box_h, confidence=float(score)))

        return boxes

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeFaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def get_default_detector(backend: str = "mediapipe", **kwargs) -> BaseFaceDetector:
    if backend == "mediapipe":
        return MediaPipeFaceDetector(**kwargs)
    raise ValueError(f"Unsupported face detector backend: {backend!r}")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import cv2
    from face_detector import MediaPipeFaceDetector

    parser = argparse.ArgumentParser(description="Detect faces in a single image.")
    parser.add_argument("image", type=str, help="Path to an image file.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    with MediaPipeFaceDetector() as detector:
        faces = detector.detect(img)

    if not faces:
        print("No faces detected.")
    for i, f in enumerate(faces):
        print(f"Face {i}: bbox=({f.x}, {f.y}, {f.width}, {f.height}) confidence={f.confidence:.3f}")
