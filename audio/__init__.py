"""KryptosID Audio Forensic Analysis & Synthetic Speech Detection Module.

Provides:
- Robust multi-format audio track demuxing from video (MP4, MOV, WebM, AVI) and audio (WAV, MP3, FLAC, OGG).
- Acoustic deepfake & neural vocoder artifact detection (LFCC, high-frequency spectral void, pitch jitter/shimmer).
- Audio-Visual Lip-Sync Temporal Consistency Verification for Video KYC.
"""

from audio.extract import extract_audio_from_bytes, is_supported_audio_media
from audio.features import extract_acoustic_forensic_features
from audio.detector import AudioForensicDetector, get_audio_detector
from audio.av_sync import verify_audiovisual_sync

__all__ = [
    "extract_audio_from_bytes",
    "is_supported_audio_media",
    "extract_acoustic_forensic_features",
    "AudioForensicDetector",
    "get_audio_detector",
    "verify_audiovisual_sync",
]
