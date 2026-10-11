"""Video Processing, Frame Sampling, Temporal Aggregation, and Liveness.

Implements Section 3.4 and Section 8 of DESIGN.md:
- sampler.py: Frame sampling (at fixed FPS) and face alignment.
- aggregate.py: Top-k frame artifact aggregation and temporal jump penalty.
- liveness.py: Blink and challenge-response liveness checks.
"""

from video.sampler import sample_frames_from_bytes, extract_face_crops
from video.aggregate import aggregate_video_frame_scores, decide_video_verdict
from video.liveness import verify_liveness_challenge

__all__ = [
    "sample_frames_from_bytes",
    "extract_face_crops",
    "aggregate_video_frame_scores",
    "decide_video_verdict",
    "verify_liveness_challenge",
]
