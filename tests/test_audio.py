"""Tests for Audio Forensics & Synthetic Speech Detection.

Verifies:
1. Audio extraction from raw bytes (soundfile and PyAV).
2. Acoustic forensic feature extraction (LFCC, Jitter, Shimmer, HFER, Spectral Roll-Off).
3. Synthetic vocoder detection (Bona Fide vs Neural Vocoder discrimination).
4. Audio-Visual Lip-Sync Temporal Cross-Correlation verification.
"""

import io
import numpy as np
import pytest

from audio.extract import (
    extract_audio_from_bytes,
    generate_synthetic_test_audio,
    is_supported_audio_media,
)
from audio.features import (
    compute_lfcc,
    compute_spectrogram,
    extract_acoustic_forensic_features,
    extract_pitch_and_perturbation,
)
from audio.detector import AudioForensicDetector, get_audio_detector
from audio.av_sync import verify_audiovisual_sync, compute_audio_rms_envelope


def test_audio_extension_support():
    """Verify supported audio and video media extensions."""
    assert is_supported_audio_media("test.wav") is True
    assert is_supported_audio_media("interview.mp3") is True
    assert is_supported_audio_media("voice.ogg") is True
    assert is_supported_audio_media("clip.flac") is True
    assert is_supported_audio_media("video.mp4") is True
    assert is_supported_audio_media("doc.pdf") is False
    assert is_supported_audio_media("photo.jpg") is False


def test_synthetic_test_audio_generation():
    """Verify bona fide and fake test audio waveform generation."""
    bona_audio, sr = generate_synthetic_test_audio(duration_sec=1.5, sample_rate=16000, is_fake=False)
    assert sr == 16000
    assert len(bona_audio) == 24000
    assert np.max(np.abs(bona_audio)) <= 1.01

    fake_audio, sr = generate_synthetic_test_audio(duration_sec=1.5, sample_rate=16000, is_fake=True)
    assert sr == 16000
    assert len(fake_audio) == 24000
    assert np.max(np.abs(fake_audio)) <= 1.01


def test_acoustic_features_extraction():
    """Verify LFCC, pitch jitter, and spectral indicators calculation."""
    audio, sr = generate_synthetic_test_audio(duration_sec=2.0, sample_rate=16000, is_fake=False)
    features = extract_acoustic_forensic_features(audio, sample_rate=sr)

    assert "mean_f0_hz" in features
    assert 80.0 <= features["mean_f0_hz"] <= 250.0
    assert "pitch_jitter_pct" in features
    assert features["pitch_jitter_pct"] > 0.0
    assert "pitch_shimmer_pct" in features
    assert "hf_energy_ratio" in features
    assert "spectral_rolloff_95_hz" in features
    assert "spectral_flatness" in features
    assert len(features["lfcc_mean_c1_c5"]) == 5


def test_vocoder_detector_discrimination():
    """Verify that AudioForensicDetector cleanly separates bona fide human speech from neural vocoders."""
    detector = AudioForensicDetector()

    # 1. Bona fide simulation (has natural micro-tremor, continuous high frequencies)
    bona_audio, sr = generate_synthetic_test_audio(duration_sec=3.0, sample_rate=16000, is_fake=False)
    bona_res = detector.analyze_waveform(bona_audio, sample_rate=sr)

    assert bona_res["decision"] == "genuine"
    assert bona_res["score"] < 0.45
    assert bona_res["has_speech_content"] is True

    # 2. Synthetic vocoder simulation (brickwall 6.8 kHz void, quantized flat pitch)
    fake_audio, sr = generate_synthetic_test_audio(duration_sec=3.0, sample_rate=16000, is_fake=True)
    fake_res = detector.analyze_waveform(fake_audio, sample_rate=sr)

    assert fake_res["decision"] == "manipulated"
    assert fake_res["score"] > 0.55
    assert len(fake_res["anomalies_detected"]) > 0


def test_audiovisual_lip_sync():
    """Verify temporal lip-sync cross-correlation detection."""
    # Generate audio for 3 seconds
    audio, sr = generate_synthetic_test_audio(duration_sec=3.0, sample_rate=16000, is_fake=False)

    # 1. Synchronized video: mouth opening varies with audio envelope
    times, rms = compute_audio_rms_envelope(audio, sample_rate=sr)
    num_frames = 12  # 3 seconds at 4 fps
    frame_times = np.linspace(0, 3.0, num_frames, endpoint=False)
    env_sampled = np.interp(frame_times, times, rms)

    sync_crops = []
    for val in env_sampled:
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        # Simulate mouth opening vertical contrast in lower third
        aperture = int(np.clip(val * 25, 2, 25))
        crop[65 : 65 + aperture, 35:65] = 220
        sync_crops.append(crop)

    sync_result = verify_audiovisual_sync(sync_crops, audio, video_fps=4.0)
    assert sync_result["sync_decision"] == "genuine"
    assert sync_result["is_synchronized"] is True
    assert sync_result["max_cross_corr"] > 0.30

    # 2. Desynchronized video: completely inverted / random mouth motion
    desync_crops = []
    for _ in range(num_frames):
        crop = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
        desync_crops.append(crop)

    desync_result = verify_audiovisual_sync(desync_crops, audio, video_fps=4.0)
    # Random noise mouth motion will fail synchronization threshold or have high desync prob
    assert desync_result["desync_probability"] > 0.40
