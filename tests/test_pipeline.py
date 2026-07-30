"""Tests for the preprocessing pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    PipelineConfig,
    ProcessedFrame,
    VideoResult,
    VideoPreprocessor,
    process_video,
    process_directory,
    FrameExtractionConfig,
    AlignmentConfig,
    MediaPipeFaceDetector,
    FaceBox,
)


class TestDataclasses:
    """Pure-Python tests — no dependencies needed."""

    def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert isinstance(config.frame_extraction, FrameExtractionConfig)
        assert isinstance(config.alignment, AlignmentConfig)
        assert config.face_detection_min_confidence == 0.5
        assert config.save_frames is True
        assert config.save_faces is True
        assert config.skip_frames_no_face is True

    def test_pipeline_config_custom(self):
        config = PipelineConfig(
            save_frames=False,
            save_faces=True,
            skip_frames_no_face=False,
            face_detection_min_confidence=0.7,
        )
        assert config.save_frames is False
        assert config.skip_frames_no_face is False
        assert config.face_detection_min_confidence == 0.7

    def test_processed_frame_dataclass(self):
        box = FaceBox(x=10, y=20, width=100, height=150, confidence=0.95)
        pf = ProcessedFrame(
            frame_path=Path("frame.jpg"),
            face_path=Path("face.jpg"),
            face_box=box,
            has_face=True,
        )
        assert pf.frame_path == Path("frame.jpg")
        assert pf.has_face is True
        assert pf.face_box.confidence == 0.95

    def test_video_result_dataclass(self):
        result = VideoResult(
            video_path=Path("v.mp4"),
            num_frames_extracted=10,
            num_faces_detected=8,
            frames=[],
            output_dir=Path("out"),
        )
        assert result.num_frames_extracted == 10
        assert result.num_faces_detected == 8


class TestFaceBox:
    def test_as_xyxy(self):
        box = FaceBox(x=10, y=20, width=100, height=150, confidence=0.9)
        assert box.as_xyxy() == (10, 20, 110, 170)

    def test_area(self):
        box = FaceBox(x=0, y=0, width=50, height=100, confidence=0.5)
        assert box.area() == 5000


class TestPipelineStructure:
    """Test that the pipeline creates the right directory structure."""

    def test_process_video_creates_output(self):
        """Creates a synthetic video, runs pipeline, checks output layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_video.mp4"
            output_dir = Path(tmpdir) / "output"

            # Create a small synthetic video (10 frames of random noise)
            _create_synthetic_video(str(video_path), num_frames=10, fps=10)

            # Run the pipeline
            result = process_video(str(video_path), str(output_dir))

            # Check result metadata
            assert result.video_path == video_path
            assert result.num_frames_extracted > 0
            assert result.output_dir.exists()

            # Check output directory structure
            video_out = output_dir / "test_video"
            assert video_out.exists()
            assert (video_out / "faces").exists()
            assert (video_out / "metadata.json").exists()

            # Check metadata content
            with open(video_out / "metadata.json") as f:
                meta = json.load(f)
            assert meta["video"] == str(video_path)
            assert len(meta["frames"]) == result.num_frames_extracted

    def test_process_video_no_frames_flag(self):
        """When save_frames=False, frame dir should not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "no_frames_test.mp4"
            _create_synthetic_video(str(video_path), num_frames=5, fps=10)

            config = PipelineConfig(save_frames=False)
            result = process_video(str(video_path), str(tmpdir), config=config)

            video_out = result.output_dir
            assert not (video_out / "frames").exists()

    def test_process_directory(self):
        """process_directory should find and process all videos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create two test videos
            _create_synthetic_video(str(input_dir / "vid1.mp4"), num_frames=3, fps=10)
            _create_synthetic_video(str(input_dir / "vid2.mp4"), num_frames=4, fps=10)

            results = process_directory(str(input_dir), str(output_dir))
            assert len(results) == 2


def test_pipeline_config_serialization():
    """PipelineConfig should serialize to dict correctly."""
    config = PipelineConfig(
        face_detection_min_confidence=0.8,
        frame_extraction=FrameExtractionConfig(target_fps=10.0),
    )
    d = {
        "frame_extraction": {
            "target_fps": 10.0,
            "blur_threshold": 100.0,
            "corruption_threshold": 15.0,
            "enhance_blurry_frames": True,
            "jpeg_quality": 95,
        },
            "alignment": {
                "output_size": 299,
                "margin_fraction": 0.25,
                "min_landmark_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
        "face_detection_min_confidence": 0.8,
        "save_frames": True,
        "save_faces": True,
        "skip_frames_no_face": True,
    }

    from dataclasses import asdict
    assert asdict(config) == d


def test_process_video_nonexistent():
    """Should raise FileNotFoundError for missing video."""
    with pytest.raises(FileNotFoundError):
        process_video("/nonexistent/video.mp4", "/tmp/out")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _create_synthetic_video(path: str, num_frames: int = 10, fps: float = 10.0, width: int = 640, height: int = 480):
    """Create a short synthetic video with random noise frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
