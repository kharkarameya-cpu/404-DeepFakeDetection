from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .face_detector import FaceBox
from .utils import ensure_model_asset, get_logger, resize_with_pad

logger = get_logger(__name__)

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_LANDMARKER_FILENAME = "face_landmarker.task"


@dataclass
class AlignmentConfig:
    output_size: int = 299
    margin_fraction: float = 0.25
    min_landmark_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class FaceAligner:
    """Aligns, crops, and resizes faces using MediaPipe FaceLandmarker."""

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for FaceAligner. Install with: pip install mediapipe"
            ) from exc

        model_path = ensure_model_asset(_FACE_LANDMARKER_MODEL_URL, _FACE_LANDMARKER_FILENAME)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_face_detection_confidence=self.config.min_landmark_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def align_and_crop(self, image: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        landmarks = self._get_landmarks(image, face_box)
        if landmarks is None:
            logger.debug("No landmarks found; skipping frame.")
            return None

        rotated, rotated_landmarks = self._rotate_to_level_eyes(image, landmarks)
        crop = self._crop_with_margin(rotated, rotated_landmarks)
        if crop is None or crop.size == 0:
            return None

        return resize_with_pad(crop, target_size=self.config.output_size)

    def _get_landmarks(self, image: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        import mediapipe as mp

        h, w = image.shape[:2]

        pad_x = int(face_box.width * 0.5)
        pad_y = int(face_box.height * 0.5)
        x0 = max(0, face_box.x - pad_x)
        y0 = max(0, face_box.y - pad_y)
        x1 = min(w, face_box.x + face_box.width + pad_x)
        y1 = min(h, face_box.y + face_box.height + pad_y)

        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        roi_h, roi_w = roi.shape[:2]
        face_landmarks = results.face_landmarks[0]
        points = np.array(
            [[lm.x * roi_w + x0, lm.y * roi_h + y0] for lm in face_landmarks],
            dtype=np.float32,
        )
        return points

    def _rotate_to_level_eyes(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        left_eye = landmarks[LEFT_EYE_OUTER]
        right_eye = landmarks[RIGHT_EYE_OUTER]

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

        h, w = image.shape[:2]
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(
            image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([landmarks, ones])
        rotated_landmarks = (rot_mat @ homogeneous.T).T

        return rotated_image, rotated_landmarks

    def _crop_with_margin(self, image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]

        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)

        box_w = x_max - x_min
        box_h = y_max - y_min
        margin = self.config.margin_fraction

        x0 = int(max(0, x_min - box_w * margin))
        y0 = int(max(0, y_min - box_h * margin))
        x1 = int(min(w, x_max + box_w * margin))
        y1 = int(min(h, y_max + box_h * margin))

        if x1 <= x0 or y1 <= y0:
            return None

        return image[y0:y1, x0:x1]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceAligner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from face_detector import MediaPipeFaceDetector

    parser = argparse.ArgumentParser(description="Align and crop the primary face in an image.")
    parser.add_argument("image", type=str, help="Path to an image file.")
    parser.add_argument("output", type=str, help="Path to save the aligned face image.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    with MediaPipeFaceDetector() as detector:
        box = detector.detect_primary(img)
    if box is None:
        raise SystemExit("No face detected.")

    with FaceAligner() as aligner:
        aligned = aligner.align_and_crop(img, box)

    if aligned is None:
        raise SystemExit("Could not align/crop face (no landmarks found).")

    cv2.imwrite(args.output, aligned)
    print(f"Saved aligned face to {args.output}")
