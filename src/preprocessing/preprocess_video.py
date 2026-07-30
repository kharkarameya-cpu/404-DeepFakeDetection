"""
preprocess_video.py

Phase 1 pipeline orchestrator — ties together:
  1. Frame extraction (frame_extractor.py)
  2. Face detection  (face_detector.py)
  3. Face alignment / cropping / resizing (face_alignment.py)
  4. Organized output saving

Usage (CLI):
  python preprocess_video.py <video_path> --output-dir <dir>

Usage (Python):
  from preprocess_video import process_video
  process_video("data/raw/fake/video.mp4", "data/processed")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .face_alignment import FaceAligner, AlignmentConfig
from .face_detector import MediaPipeFaceDetector, FaceBox
from .frame_extractor import FrameExtractor, FrameExtractionConfig
from .utils import PathLike, ensure_dir, get_logger, resize_with_pad

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    frame_extraction: FrameExtractionConfig = None
    alignment: AlignmentConfig = None
    face_detection_min_confidence: float = 0.5
    save_frames: bool = True
    save_faces: bool = True
    skip_frames_no_face: bool = True

    def __post_init__(self):
        if self.frame_extraction is None:
            self.frame_extraction = FrameExtractionConfig()
        if self.alignment is None:
            self.alignment = AlignmentConfig()


@dataclass
class ProcessedFrame:
    frame_path: Optional[Path]
    face_path: Optional[Path]
    face_box: Optional[FaceBox]
    has_face: bool


@dataclass
class VideoResult:
    video_path: Path
    num_frames_extracted: int
    num_faces_detected: int
    frames: List[ProcessedFrame]
    output_dir: Path


class VideoPreprocessor:
    """End-to-end video preprocessing pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._frame_extractor = FrameExtractor(self.config.frame_extraction)
        self._face_detector = MediaPipeFaceDetector(
            min_confidence=self.config.face_detection_min_confidence
        )
        self._face_aligner = FaceAligner(self.config.alignment)

    def process(self, video_path: PathLike, output_dir: PathLike) -> VideoResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = ensure_dir(output_dir)
        video_stem = video_path.stem
        video_output = ensure_dir(output_dir / video_stem)

        frames_dir = ensure_dir(video_output / "frames") if self.config.save_frames else None
        faces_dir = ensure_dir(video_output / "faces") if self.config.save_faces else None

        frame_paths = self._frame_extractor.extract(video_path, frames_dir or (video_output / "_frames_tmp"))

        processed_frames: List[ProcessedFrame] = []
        num_faces = 0

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            face_box = self._face_detector.detect_primary(frame)
            face_path = None

            if face_box is not None:
                aligned = self._face_aligner.align_and_crop(frame, face_box)
                if aligned is not None and faces_dir is not None:
                    face_path = faces_dir / f"face_{i+1:04d}.jpg"
                    cv2.imwrite(str(face_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    num_faces += 1
                elif not self.config.skip_frames_no_face:
                    aligned = resize_with_pad(frame, target_size=self.config.alignment.output_size)
                    if faces_dir is not None:
                        face_path = faces_dir / f"face_{i+1:04d}.jpg"
                        cv2.imwrite(str(face_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed_frames.append(ProcessedFrame(
                frame_path=frame_path if frames_dir else None,
                face_path=face_path,
                face_box=face_box,
                has_face=face_box is not None,
            ))

        if not self.config.save_frames and frames_dir is None:
            _tmp = video_output / "_frames_tmp"
            if _tmp.exists():
                import shutil
                shutil.rmtree(_tmp)

        result = VideoResult(
            video_path=video_path,
            num_frames_extracted=len(frame_paths),
            num_faces_detected=num_faces,
            frames=processed_frames,
            output_dir=video_output,
        )

        self._save_metadata(result, video_output)
        logger.info(
            "Processed %s: %d frames, %d faces detected",
            video_path.name, result.num_frames_extracted, result.num_faces_detected
        )
        return result

    def _save_metadata(self, result: VideoResult, output_dir: Path) -> None:
        meta = {
            "video": str(result.video_path),
            "num_frames_extracted": result.num_frames_extracted,
            "num_faces_detected": result.num_faces_detected,
            "frames": [
                {
                    "frame": str(f.frame_path) if f.frame_path else None,
                    "face": str(f.face_path) if f.face_path else None,
                    "has_face": f.has_face,
                    "bbox": asdict(f.face_box) if f.face_box else None,
                }
                for f in result.frames
            ],
            "config": asdict(self.config),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def close(self) -> None:
        self._face_detector.close()
        self._face_aligner.close()

    def __enter__(self) -> "VideoPreprocessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def process_video(
    video_path: PathLike,
    output_dir: PathLike,
    config: Optional[PipelineConfig] = None,
) -> VideoResult:
    """Convenience function for single-video processing."""
    with VideoPreprocessor(config) as processor:
        return processor.process(video_path, output_dir)


def process_directory(
    input_dir: PathLike,
    output_dir: PathLike,
    config: Optional[PipelineConfig] = None,
    extensions=(".mp4", ".avi", ".mov", ".mkv"),
) -> List[VideoResult]:
    """Process all videos in a directory."""
    input_dir = Path(input_dir)
    results = []
    with VideoPreprocessor(config) as processor:
        for ext in extensions:
            for video_path in sorted(input_dir.rglob(f"*{ext}")):
                rel = video_path.relative_to(input_dir)
                out = Path(output_dir) / rel.with_suffix("")
                result = processor.process(video_path, out)
                results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline on a video.")
    parser.add_argument("video", type=str, help="Path to input video file.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Root output directory.")
    parser.add_argument("--fps", type=float, default=7.5, help="Target frame sampling rate.")
    parser.add_argument("--no-frames", action="store_true", help="Don't save intermediate frames.")
    parser.add_argument("--no-faces", action="store_true", help="Don't save aligned face crops.")
    args = parser.parse_args()

    config = PipelineConfig(
        frame_extraction=FrameExtractionConfig(target_fps=args.fps),
        save_frames=not args.no_frames,
        save_faces=not args.no_faces,
    )
    result = process_video(args.video, args.output_dir, config)
    print(f"Done — {result.num_faces_detected} faces from {result.num_frames_extracted} frames.")
