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
    FrameStore,
    StoreConfig,
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
    """Test that the pipeline creates the right output layout."""

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
            assert (video_out / "metadata.json").exists()

            # Check metadata content
            with open(video_out / "metadata.json") as f:
                meta = json.load(f)
            assert meta["video"] == str(video_path)
            assert len(meta["frames"]) == result.num_frames_extracted

    def test_process_video_hdf5_store(self):
        """With HDF5 store enabled, frames+faces go into a single .h5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "h5_test.mp4"
            _create_synthetic_video(str(video_path), num_frames=6, fps=10)

            config = PipelineConfig(use_hdf5_store=True)
            result = process_video(str(video_path), str(tmpdir), config=config)

            h5_path = result.output_dir / f"{result.video_path.stem}.h5"
            assert h5_path.exists()

            # The store should contain all extracted frames
            with FrameStore(h5_path) as store:
                assert store.num_frames == result.num_frames_extracted
                assert store.get_frames().shape[0] == result.num_frames_extracted

    def test_process_video_no_hdf5_jpeg_fallback(self):
        """With HDF5 disabled, frames/faces dirs are created as JPEGs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "jpeg_test.mp4"
            _create_synthetic_video(str(video_path), num_frames=4, fps=10)

            config = PipelineConfig(use_hdf5_store=False)
            result = process_video(str(video_path), str(tmpdir), config=config)

            video_out = result.output_dir
            assert (video_out / "frames").exists()
            assert (video_out / "faces").exists()
            assert not (video_out / f"{video_path.stem}.h5").exists()

    def test_process_video_no_frames_flag(self):
        """When save_frames=False, no standalone frame dir is kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "no_frames_test.mp4"
            _create_synthetic_video(str(video_path), num_frames=5, fps=10)

            config = PipelineConfig(save_frames=False, use_hdf5_store=False)
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


class TestFrameStore:
    """Tests for the HDF5-backed store."""

    def test_add_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.h5"
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            face = np.zeros((299, 299, 3), dtype=np.uint8)
            box = FaceBox(x=5, y=10, width=50, height=60, confidence=0.9)

            with FrameStore(path) as store:
                store.add_frame(frame)
                store.add_frame(np.full((64, 64, 3), 128, dtype=np.uint8))
                store.add_face(face, bbox=box, frame_idx=0, confidence=0.9)

            with FrameStore(path) as store:
                assert store.num_frames == 2
                assert store.num_faces == 1
                assert store.get_frames().shape == (2, 64, 64, 3)
                assert store.get_faces().shape == (1, 299, 299, 3)
                assert store.get_frame_map()[0] == 0
                assert list(store.get_bboxes()[0]) == [5, 10, 50, 60]
                assert store.get_confidences()[0] == 0.9

    def test_buffered_flush(self):
        """Many small appends should be buffered and flushed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buf.h5"
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            cfg = StoreConfig(buffer_size=8)

            with FrameStore(path, config=cfg) as store:
                for i in range(25):
                    store.add_frame(np.full_like(frame, i, dtype=np.uint8))

            with FrameStore(path) as store:
                assert store.num_frames == 25
                frames = store.get_frames()
                assert frames.shape == (25, 32, 32, 3)
                assert frames[24, 0, 0, 0] == 24

    def test_attrs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attr.h5"
            with FrameStore(path) as store:
                store.set_attrs({"video_path": "x.mp4", "num_faces": 3})
            with FrameStore(path) as store:
                attrs = store.get_attrs()
                assert attrs["video_path"] == "x.mp4"
                assert attrs["num_faces"] == 3


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
        "model_selection": 1,
        "save_frames": True,
        "save_faces": True,
        "skip_frames_no_face": True,
        "use_hdf5_store": True,
        "store_config": {
            "compression": "gzip",
            "compression_opts": 4,
            "chunk_rows": 64,
            "max_frames": 10000,
            "buffer_size": 32,
        },
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
