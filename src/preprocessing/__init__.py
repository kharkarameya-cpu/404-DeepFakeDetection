import os

# mediapipe's ``drawing_utils`` transitively imports ``matplotlib.pyplot``,
# whose default backend does an expensive font-cache scan (80+ seconds on a
# fresh profile). Pin the headless Agg backend so importing mediapipe is fast.
# Done here, at package import time, *before* mediapipe is ever imported.
os.environ.setdefault("MPLBACKEND", "Agg")

from .face_detector import MediaPipeFaceDetector, FaceBox, get_default_detector
from .face_alignment import FaceAligner, AlignmentConfig
from .frame_extractor import FrameExtractor, FrameExtractionConfig, extract_frames
from .frame_selection import (
    FrameSelector,
    FrameSelectionConfig,
    SelectedFrame,
    RingBuffer,
    LSHIndex,
    SceneGraph,
    dhash,
    hamming_distance,
    feature_vector,
    select_frames_from_video,
)
from .frame_store import FrameStore, StoreConfig
from .preprocess_video import VideoPreprocessor, PipelineConfig, ProcessedFrame, VideoResult, process_video, process_directory
from .utils import (
    PathLike,
    ensure_dir,
    get_logger,
    list_videos,
    variance_of_laplacian,
    is_blurry,
    is_severely_corrupted,
    sharpen_frame,
    deblur_frame,
    enhance_frame,
    resize_with_pad,
    open_video_capture,
)

__all__ = [
    "MediaPipeFaceDetector",
    "FaceBox",
    "get_default_detector",
    "FaceAligner",
    "AlignmentConfig",
    "FrameExtractor",
    "FrameExtractionConfig",
    "extract_frames",
    "FrameSelector",
    "FrameSelectionConfig",
    "SelectedFrame",
    "RingBuffer",
    "LSHIndex",
    "SceneGraph",
    "dhash",
    "hamming_distance",
    "feature_vector",
    "select_frames_from_video",
    "FrameStore",
    "StoreConfig",
    "VideoPreprocessor",
    "PipelineConfig",
    "ProcessedFrame",
    "VideoResult",
    "process_video",
    "process_directory",
    "PathLike",
    "ensure_dir",
    "get_logger",
    "list_videos",
    "variance_of_laplacian",
    "is_blurry",
    "is_severely_corrupted",
    "sharpen_frame",
    "deblur_frame",
    "enhance_frame",
    "resize_with_pad",
    "open_video_capture",
]
