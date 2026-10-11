"""Audio Forensic Deepfake & Synthetic Speech Detector.

Analyzes speech recordings for telltale signs of neural vocoder synthesis
(HiFi-GAN, MelGAN, WaveGrad, ElevenLabs, XTTS, StyleTTS, VITS) and voice cloning:
- High-frequency spectral truncation / energy voids (>6.5 kHz).
- Unnatural pitch contour (hyper-smooth flat pitch or erratic splice discontinuities).
- Linear Frequency Cepstral Coefficient (LFCC) variance anomalies.
- Transposed convolution periodic phase ripples (spectral flatness deviations).
"""

from typing import Dict, Any, Optional, Tuple, List
import io
import time
import numpy as np

from audio.extract import extract_audio_from_bytes, generate_synthetic_test_audio
from audio.features import extract_acoustic_forensic_features


class AudioForensicDetector:
    """Forensic acoustic classifier and explanation engine for synthetic speech."""

    def __init__(
        self,
        decision_threshold: float = 0.50,
        inconclusive_margin: float = 0.05,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.inconclusive_margin = inconclusive_margin

    def score_features(self, features: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
        """Compute calibrated synthetic speech probability and anomaly indicators.

        Returns:
            Tuple of (synthetic_probability, detected_anomalies, sub_scores).
        """
        anomalies = []
        sub_scores = {}

        # 1. High-Frequency Void / Roll-off Analysis (>6.5 kHz)
        # Natural speech maintains vocal tract fricative energy up to 8 kHz.
        # Vocoders based on 80-bin mel spectrograms abruptly cut off near 6.8-7.2 kHz.
        hf_ratio = features.get("hf_energy_ratio", 0.001)
        rolloff_95 = features.get("spectral_rolloff_95_hz", 7000.0)

        hf_score = 0.0
        if hf_ratio < 0.0003 or rolloff_95 < 6850.0:
            hf_score = 0.88
            anomalies.append(
                f"High-frequency spectral void detected (HFER={hf_ratio:.5f}, 95% roll-off at {rolloff_95:.0f} Hz < 6850 Hz)"
            )
        elif hf_ratio < 0.0008:
            hf_score = 0.65
            anomalies.append(f"Subnormal high-frequency energy ratio (HFER={hf_ratio:.5f})")
        else:
            hf_score = 0.15

        sub_scores["hf_spectral_void_score"] = round(hf_score, 4)

        # 2. Pitch Micro-Tremor (Jitter RAP) Analysis
        # Human vocal cords exhibit natural micro-perturbation: 0.5% - 1.8%.
        # Neural TTS models produce unnaturally flat pitch (< 0.25%) or erratic frame splices (> 3.5%).
        jitter_pct = features.get("pitch_jitter_pct", 1.0)
        shimmer_pct = features.get("pitch_shimmer_pct", 3.0)

        jitter_score = 0.0
        if jitter_pct < 0.25:
            jitter_score = 0.85
            anomalies.append(
                f"Unnatural pitch smoothness / robotic lack of micro-tremor (Jitter={jitter_pct:.3f}% < 0.25%)"
            )
        elif jitter_pct > 3.8:
            jitter_score = 0.75
            anomalies.append(
                f"Unnatural pitch discontinuity / phase splicing artifact (Jitter={jitter_pct:.3f}% > 3.8%)"
            )
        elif 0.5 <= jitter_pct <= 2.0:
            jitter_score = 0.12  # Consistent with bona fide human phonation
        else:
            jitter_score = 0.35

        sub_scores["pitch_microtremor_score"] = round(jitter_score, 4)

        # 3. Spectral Flatness (Wiener Entropy)
        # Periodic upsampling in vocoders creates periodic ripples in power spectrum.
        flatness = features.get("spectral_flatness", 0.005)
        flatness_score = 0.0
        if flatness < 0.0008:
            flatness_score = 0.78
            anomalies.append(f"Abnormally low spectral Wiener entropy (flatness={flatness:.6f})")
        elif flatness > 0.045:
            flatness_score = 0.70
            anomalies.append(f"Excessive spectral noise injection / vocoder diffusion artifact (flatness={flatness:.5f})")
        else:
            flatness_score = 0.18

        sub_scores["spectral_flatness_score"] = round(flatness_score, 4)

        # 4. LFCC Variance across Linear Filterbanks
        # Neural vocoders show elevated variance in linear high-frequency filterbanks.
        lfcc_var = features.get("lfcc_variance_mean", 0.1)
        lfcc_score = 0.0
        if lfcc_var > 0.45:
            lfcc_score = 0.80
            anomalies.append(f"Elevated LFCC filterbank variance inconsistency (var={lfcc_var:.4f})")
        elif lfcc_var < 0.01:
            lfcc_score = 0.75
            anomalies.append(f"Suppressed cepstral dynamics (LFCC var={lfcc_var:.4f})")
        else:
            lfcc_score = 0.20

        sub_scores["lfcc_artifact_score"] = round(lfcc_score, 4)

        # Weighted forensic acoustic probability
        # High-frequency cutoff and pitch jitter are the most discriminating physical indicators
        weights = {
            "hf": 0.40,
            "jitter": 0.30,
            "flatness": 0.15,
            "lfcc": 0.15,
        }
        raw_prob = (
            weights["hf"] * hf_score
            + weights["jitter"] * jitter_score
            + weights["flatness"] * flatness_score
            + weights["lfcc"] * lfcc_score
        )

        calibrated_prob = float(np.clip(raw_prob, 0.01, 0.99))
        return calibrated_prob, anomalies, sub_scores

    def analyze_waveform(
        self,
        signal: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Analyze 1D audio waveform and return complete forensic assessment."""
        features = extract_acoustic_forensic_features(signal, sample_rate=sample_rate)
        score, anomalies, sub_scores = self.score_features(features)

        high_t = self.decision_threshold + self.inconclusive_margin
        low_t = self.decision_threshold - self.inconclusive_margin

        if score > high_t:
            decision = "manipulated"
            verdict_text = "Synthetic Voice / Neural Vocoder Detected"
        elif score < low_t:
            decision = "genuine"
            verdict_text = "Bona Fide Human Voice Verified"
        else:
            decision = "inconclusive"
            verdict_text = "Inconclusive Acoustic Evidence"

        return {
            "decision": decision,
            "verdict_text": verdict_text,
            "score": round(score, 4),
            "threshold": self.decision_threshold,
            "acoustic_features": features,
            "sub_scores": sub_scores,
            "anomalies_detected": anomalies,
            "has_speech_content": bool(features.get("duration_sec", 0) >= 0.2),
        }

    def analyze_bytes(self, file_bytes: bytes) -> Dict[str, Any]:
        """Demux audio from container bytes and run acoustic forensic analysis."""
        waveform, sr = extract_audio_from_bytes(file_bytes)
        if waveform is None or len(waveform) == 0:
            return {
                "decision": "inconclusive",
                "verdict_text": "No valid audio track found in media container",
                "score": 0.50,
                "threshold": self.decision_threshold,
                "acoustic_features": {},
                "sub_scores": {},
                "anomalies_detected": ["No audio stream present or undecodable audio format"],
                "has_speech_content": False,
            }
        return self.analyze_waveform(waveform, sample_rate=sr)


_GLOBAL_AUDIO_DETECTOR: Optional[AudioForensicDetector] = None


def get_audio_detector() -> AudioForensicDetector:
    """Singleton accessor for AudioForensicDetector."""
    global _GLOBAL_AUDIO_DETECTOR
    if _GLOBAL_AUDIO_DETECTOR is None:
        _GLOBAL_AUDIO_DETECTOR = AudioForensicDetector()
    return _GLOBAL_AUDIO_DETECTOR
