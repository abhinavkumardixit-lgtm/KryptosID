"""Temporal Score Aggregation and Decision Logic for Videos.

Implements Section 8 of DESIGN.md:
- Top-k frame score aggregation (k = 10% of frames).
- Temporal consistency penalty: penalizes large score fluctuations between adjacent frames.
- Decision layer: genuine, manipulated, or inconclusive with rule tracing.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def aggregate_video_frame_scores(
    frame_scores: List[float],
    top_k_ratio: float = 0.10,
    jump_penalty_weight: float = 0.15,
) -> Tuple[float, float, float]:
    """Aggregate per-frame calibrated scores over time (Section 8.3).
    
    Args:
        frame_scores: List of calibrated probabilities [0, 1] for each frame.
        top_k_ratio: Fraction of highest-scoring frames to average (default 10%).
        jump_penalty_weight: Weight for temporal inconsistency penalty.
        
    Returns:
        Tuple of (aggregated_score, top_k_mean, jump_penalty).
    """
    if not frame_scores:
        return 0.5, 0.5, 0.0

    scores = np.asarray(frame_scores, dtype=float)
    n = len(scores)

    # 1. Top-k mean
    k = max(1, int(np.ceil(n * top_k_ratio)))
    top_k_scores = np.sort(scores)[-k:]
    top_k_mean = float(np.mean(top_k_scores))

    # 2. Temporal consistency jump penalty
    if n > 1:
        consecutive_jumps = np.abs(np.diff(scores))
        jump_penalty = float(np.mean(consecutive_jumps))
    else:
        jump_penalty = 0.0

    # 3. Fused aggregate score
    fused_score = float(np.clip(top_k_mean + jump_penalty_weight * jump_penalty, 0.0, 1.0))
    return fused_score, top_k_mean, jump_penalty


def decide_video_verdict(
    aggregated_score: float,
    threshold: float,
    liveness_passed: bool = True,
    face_count: int = 1,
    min_faces_required: int = 3,
    inconclusive_margin: float = 0.05,
) -> Tuple[str, str]:
    """Determine final video verdict and rule fired according to Section 8.5 of DESIGN.md.
    
    Verdicts: 'genuine', 'manipulated', or 'inconclusive'.
    
    Returns:
        Tuple of (decision, rule_fired).
    """
    # Check minimum faces condition
    if face_count < min_faces_required:
        return "inconclusive", f"insufficient_faces_detected (count={face_count} < required={min_faces_required})"

    high_threshold = threshold + inconclusive_margin
    low_threshold = threshold - inconclusive_margin

    if not liveness_passed:
        if aggregated_score >= threshold:
            return "manipulated", "liveness_failed_with_high_artifact_score"
        else:
            return "inconclusive", "liveness_challenge_failed_low_artifact_score"

    if aggregated_score > high_threshold:
        return "manipulated", f"aggregated_score_exceeds_threshold ({aggregated_score:.4f} > {high_threshold:.4f})"
    elif aggregated_score < low_threshold:
        return "genuine", f"aggregated_score_below_threshold ({aggregated_score:.4f} < {low_threshold:.4f})"
    else:
        return "inconclusive", f"score_within_inconclusive_margin ([{low_threshold:.4f}, {high_threshold:.4f}])"
