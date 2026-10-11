"""Audio-Visual Lip-Sync Temporal Consistency Engine.

Detects deepfake talking heads (Wav2Lip, SadTalker, LivePortrait) and voice-swap
forgeries by analyzing temporal cross-correlation between:
1. Speech Acoustic Energy Envelope (RMS acoustic volume over time).
2. Mouth Kinematics (Mouth Aspect Ratio / lip vertical opening over sampled video frames).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def compute_audio_rms_envelope(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    window_sec: float = 0.05,  # 50 ms window
    hop_sec: float = 0.025,     # 25 ms hop (40 Hz envelope)
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RMS energy envelope of audio signal over time.

    Returns:
        Tuple of (timestamps_sec, rms_envelope).
    """
    if len(waveform) == 0:
        return np.array([]), np.array([])

    win_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)

    rms_vals = []
    times = []

    for start in range(0, len(waveform) - win_samples, hop_samples):
        chunk = waveform[start : start + win_samples]
        rms = np.sqrt(np.mean(chunk**2) + 1e-10)
        rms_vals.append(rms)
        times.append((start + win_samples / 2.0) / float(sample_rate))

    if not rms_vals:
        return np.array([0.0]), np.array([0.0])

    rms_arr = np.array(rms_vals, dtype=np.float32)
    # Normalize envelope to [0, 1]
    peak = np.max(rms_arr)
    if peak > 1e-6:
        rms_arr = rms_arr / peak

    return np.array(times, dtype=np.float32), rms_arr


def estimate_mouth_opening_metric(face_crop_rgb: np.ndarray) -> float:
    """Estimate relative vertical mouth opening from face crop image.

    Analyzes lower-third region of face crop:
    open mouth creates intra-oral cavity opening with high active contrast and vertical span.
    """
    if face_crop_rgb is None or face_crop_rgb.size == 0:
        return 0.0

    h, w, _ = face_crop_rgb.shape
    y1, y2 = int(0.60 * h), int(0.92 * h)
    x1, x2 = int(0.25 * w), int(0.75 * w)
    mouth_roi = face_crop_rgb[y1:y2, x1:x2]

    if mouth_roi.size == 0:
        return 0.0

    # Convert to grayscale luminance
    gray = np.dot(mouth_roi[..., :3], [0.299, 0.587, 0.114])
    # Active mouth cavity aperture: area of high-contrast intra-oral pixels and vertical gradient
    grad_y = np.abs(np.diff(gray, axis=0))
    active_span = float(np.sum(gray > 40)) + float(np.sum(grad_y))
    return active_span


def verify_audiovisual_sync(
    face_crops: List[np.ndarray],
    waveform: Optional[np.ndarray],
    video_fps: float = 4.0,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Measure temporal correlation between speech volume envelope and mouth kinematics.

    Args:
        face_crops: Sequence of sampled face crop frames (e.g. at 4.0 fps).
        waveform: Decoded audio waveform (16 kHz mono).
        video_fps: Frame sampling rate of face crops.
        sample_rate: Audio sampling rate.

    Returns:
        Forensic synchronization dictionary with correlation score, lag, and verdict.
    """
    if waveform is None or len(waveform) < int(sample_rate * 0.5) or len(face_crops) < 6:
        return {
            "sync_decision": "inconclusive",
            "correlation_score": 0.50,
            "max_cross_corr": 0.50,
            "optimal_lag_ms": 0.0,
            "is_synchronized": True,
            "rule_fired": "insufficient_audio_or_frames_for_lip_sync",
        }

    # 1. Compute acoustic RMS envelope over time
    audio_times, audio_rms = compute_audio_rms_envelope(waveform, sample_rate=sample_rate)

    # 2. Compute mouth aperture over video frames
    mouth_openings = np.array([estimate_mouth_opening_metric(f) for f in face_crops], dtype=np.float32)
    # Normalize mouth kinematics
    m_range = np.ptp(mouth_openings)
    if m_range > 1e-4:
        mouth_kinematics = (mouth_openings - np.min(mouth_openings)) / m_range
    else:
        mouth_kinematics = np.zeros_like(mouth_openings)

    # 3. Resample audio envelope to video frame timestamps
    num_frames = len(face_crops)
    frame_times = np.linspace(0, num_frames / video_fps, num_frames, endpoint=False)
    audio_resampled = np.interp(frame_times, audio_times, audio_rms).astype(np.float32)

    # 4. Standardize signals (zero mean, unit variance)
    def standardize(x: np.ndarray) -> np.ndarray:
        s = np.std(x)
        return (x - np.mean(x)) / (s + 1e-7)

    norm_audio = standardize(audio_resampled)
    norm_mouth = standardize(mouth_kinematics)

    # 5. Pearson correlation at zero lag
    pearson_r = float(np.mean(norm_audio * norm_mouth))

    # 6. Cross-correlation over temporal search window [-3 frames, +3 frames]
    # At 4 fps, 3 frames = 750 ms
    max_shift = min(3, len(norm_audio) // 3)
    shifts = range(-max_shift, max_shift + 1)
    xcorr_vals = []
    for sh in shifts:
        if sh < 0:
            a_sub = norm_audio[:sh]
            m_sub = norm_mouth[-sh:]
        elif sh > 0:
            a_sub = norm_audio[sh:]
            m_sub = norm_mouth[:-sh]
        else:
            a_sub = norm_audio
            m_sub = norm_mouth
        xcorr_vals.append((sh, float(np.mean(a_sub * m_sub))))

    best_shift, best_corr = max(xcorr_vals, key=lambda p: p[1])
    lag_ms = float((best_shift / video_fps) * 1000.0)

    # 7. Decision rule
    # In genuine speech, correlation is typically positive and peak lag is within +-150ms.
    # Desync or synthetic lip-sync has negative/low correlation or severe lag.
    is_sync = (best_corr >= 0.22) and (abs(lag_ms) <= 300.0)
    desync_prob = float(np.clip(1.0 - max(0.0, best_corr), 0.05, 0.95))

    if not is_sync:
        decision = "manipulated"
        rule = f"lip_sync_mismatch (peak_corr={best_corr:.3f} < 0.22 or lag={lag_ms:.0f}ms > 300ms)"
    else:
        decision = "genuine"
        rule = f"lip_sync_coherent (peak_corr={best_corr:.3f}, lag={lag_ms:.0f}ms)"

    return {
        "sync_decision": decision,
        "desync_probability": round(desync_prob, 4),
        "zero_lag_correlation": round(pearson_r, 4),
        "max_cross_corr": round(best_corr, 4),
        "optimal_lag_ms": round(lag_ms, 1),
        "is_synchronized": is_sync,
        "rule_fired": rule,
    }
