

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .utils import (
    PathLike,
    ensure_dir,
    enhance_frame,
    get_logger,
    is_severely_corrupted,
)

logger = get_logger(__name__)


@dataclass
class FrameExtractionConfig:
    """Configuration for frame sampling and enhancement."""

    target_fps: float = 7.5          # midpoint of the 5-10 FPS target range
    blur_threshold: float = 100.0    # variance-of-Laplacian cutoff for "blurry"
    corruption_threshold: float = 15.0  # below this, a frame is dropped, not fixed
    enhance_blurry_frames: bool = True
    jpeg_quality: int = 95


class FrameExtractor:
    """Extracts, filters, and (optionally) enhances frames from a video."""

    def __init__(self, config: Optional[FrameExtractionConfig] = None):
        self.config = config or FrameExtractionConfig()

    # ------------------------------------------------------------------ #
    def extract(self, video_path: PathLike, output_dir: PathLike) -> list[Path]:
        """Extract sampled frames from `video_path` and save them into
        `output_dir`.

        Returns the list of saved frame file paths, in temporal order.
        """
        video_path = Path(video_path)
        output_dir = ensure_dir(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if source_fps <= 0:
                logger.warning(
                    "Could not determine source FPS for %s; assuming 30.0", video_path
                )
                source_fps = 30.0

            sample_interval = max(1, round(source_fps / self.config.target_fps))
            logger.info(
                "Video: %s | source_fps=%.2f | target_fps=%.2f | sampling every %d frame(s)",
                video_path.name,
                source_fps,
                self.config.target_fps,
                sample_interval,
            )

            saved_paths: list[Path] = []
            frame_idx = 0
            saved_idx = 0
            dropped = 0

            for frame in self._read_frames(cap):
                if frame_idx % sample_interval == 0:
                    processed = self._process_frame(frame)
                    if processed is None:
                        dropped += 1
                    else:
                        saved_idx += 1
                        out_path = output_dir / f"frame_{saved_idx:04d}.jpg"
                        cv2.imwrite(
                            str(out_path),
                            processed,
                            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                        )
                        saved_paths.append(out_path)
                frame_idx += 1

            logger.info(
                "Extracted %d frame(s) (%d dropped as severely corrupted) from %s",
                len(saved_paths),
                dropped,
                video_path.name,
            )
            return saved_paths
        finally:
            cap.release()

    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Apply blur handling to a single sampled frame.

        Returns the (possibly enhanced) frame, or None if it should be
        dropped for being severely corrupted / unreadable.
        """
        if is_severely_corrupted(frame, threshold=self.config.corruption_threshold):
            return None

        if self.config.enhance_blurry_frames:
            return enhance_frame(frame, blur_threshold=self.config.blur_threshold)
        return frame

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
        """Generator yielding raw frames from an opened VideoCapture."""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    target_fps: float = 7.5,
) -> list[Path]:
    """Convenience function wrapping FrameExtractor for simple call sites."""
    extractor = FrameExtractor(FrameExtractionConfig(target_fps=target_fps))
    return extractor.extract(video_path, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video", type=str, help="Path to input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save frames into.")
    parser.add_argument(
        "--fps", type=float, default=7.5, help="Target sampling rate (frames per second)."
    )
    args = parser.parse_args()

    frames = extract_frames(args.video, args.output_dir, target_fps=args.fps)
    print(f"Saved {len(frames)} frame(s) to {args.output_dir}")
