"""Acoustic Forensic Feature Extraction.

Extracts discriminating features for synthetic vocoder and deepfake voice detection:
- LFCC (Linear Frequency Cepstral Coefficients) - ASVspoof challenge standard.
- High-Frequency Spectral Cutoff & Void Ratio (HFER) above 6.5 kHz.
- Pitch Jitter & Amplitude Shimmer micro-perturbations.
- Spectral Roll-Off, Spectral Centroid, and Spectral Flatness (Wiener Entropy).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import get_window
from scipy.fft import dct


def compute_spectrogram(
    signal: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) magnitude and power spectrograms.

    Args:
        signal: 1D float32 audio waveform.
        n_fft: FFT window size (default 512, ~32ms at 16kHz).
        hop_length: Frame hop size (default 160, ~10ms at 16kHz).
        window: Window function.

    Returns:
        Tuple of (magnitude_spectrogram [F, T], power_spectrogram [F, T]).
    """
    if len(signal) < n_fft:
        pad_len = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_len), mode="constant")

    win = get_window(window, n_fft)
    num_frames = 1 + (len(signal) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, n_fft),
        strides=(signal.strides[0] * hop_length, signal.strides[0]),
    )
    windowed = frames * win
    stft = np.fft.rfft(windowed, n=n_fft, axis=1)  # [num_frames, (n_fft//2 + 1)]
    magnitude = np.abs(stft).T                     # [(n_fft//2 + 1), num_frames]
    power = (magnitude ** 2) / float(n_fft)
    return magnitude, power


def compute_lfcc(
    power_spec: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    num_filters: int = 20,
    num_ceps: int = 14,
) -> np.ndarray:
    """Compute Linear Frequency Cepstral Coefficients (LFCC).

    Unlike MFCCs which compress high frequencies onto the nonlinear Mel scale,
    LFCCs use linearly spaced triangular filterbanks across the entire Nyquist band.
    This preserves telltale high-frequency artifacts left by neural vocoders.

    Returns:
        Array of shape [num_ceps, num_frames].
    """
    num_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2.0, num_bins)

    # Linear filterbank centers
    f_min = 0.0
    f_max = sample_rate / 2.0
    filter_centers = np.linspace(f_min, f_max, num_filters + 2)

    fbanks = np.zeros((num_filters, num_bins))
    for m in range(1, num_filters + 1):
        left = filter_centers[m - 1]
        center = filter_centers[m]
        right = filter_centers[m + 1]

        for k in range(num_bins):
            f = freqs[k]
            if left <= f <= center:
                fbanks[m - 1, k] = (f - left) / max(1e-6, center - left)
            elif center <= f <= right:
                fbanks[m - 1, k] = (right - f) / max(1e-6, right - center)

    # Filterbank energies
    fb_energies = np.dot(fbanks, power_spec) + 1e-10  # [num_filters, num_frames]
    log_fb = np.log(fb_energies)

    # DCT-II across filterbank dimension
    lfcc = dct(log_fb, type=2, axis=0, norm="ortho")[:num_ceps, :]
    return lfcc


def extract_pitch_and_perturbation(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> Tuple[float, float, float]:
    """Estimate fundamental frequency (F0), Jitter (RAP), and Shimmer.

    Natural vocal folds exhibit micro-tremor perturbations:
      - Jitter: cycle-to-cycle F0 variation (natural ~0.5% - 1.8%).
      - Shimmer: cycle-to-cycle peak amplitude variation (natural ~1.5% - 5.0%).
    TTS / neural vocoders often produce flat, synthetic pitch tracks with jitter < 0.2%
    or erratic phase discontinuities.

    Returns:
        Tuple of (mean_f0, jitter_pct, shimmer_pct).
    """
    frame_len = int(frame_ms * sample_rate / 1000.0)
    hop_len = int(hop_ms * sample_rate / 1000.0)

    min_f0 = 70.0
    max_f0 = 400.0
    min_lag = int(sample_rate / max_f0)
    max_lag = int(sample_rate / min_f0)

    periods = []
    amplitudes = []

    for start in range(0, len(signal) - frame_len, hop_len):
        frame = signal[start : start + frame_len]
        energy = np.sum(frame**2)
        if energy < 1e-3:
            continue  # Silence / unvoiced

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        if max_lag >= len(corr):
            continue

        search_corr = corr[min_lag:max_lag]
        if len(search_corr) == 0:
            continue

        peak_idx = np.argmax(search_corr) + min_lag
        peak_val = corr[peak_idx]

        # Voicing threshold: normalized autocorrelation > 0.35
        if peak_val / max(1e-8, corr[0]) > 0.35:
            periods.append(peak_idx)
            amplitudes.append(np.max(np.abs(frame)))

    if len(periods) < 5:
        # Default fallback if insufficient voiced speech detected
        return 130.0, 0.008, 0.025

    periods = np.array(periods, dtype=float)
    amplitudes = np.array(amplitudes, dtype=float)

    mean_period = np.mean(periods)
    mean_f0 = float(sample_rate / max(1e-4, mean_period))

    # Relative Average Perturbation (Jitter)
    period_diffs = np.abs(np.diff(periods))
    jitter_pct = float(np.mean(period_diffs) / max(1e-4, mean_period))

    # Amplitude Perturbation (Shimmer)
    amp_diffs = np.abs(np.diff(amplitudes))
    shimmer_pct = float(np.mean(amp_diffs) / max(1e-4, np.mean(amplitudes)))

    return mean_f0, jitter_pct, shimmer_pct


def compute_spectral_indicators(
    power_spec: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
) -> Dict[str, float]:
    """Compute spectral roll-off, spectral centroid, high-frequency energy ratio, and Wiener flatness."""
    num_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2.0, num_bins)

    # 1. Total energy per frequency bin across time
    freq_energy = np.sum(power_spec, axis=1) + 1e-12
    total_energy = np.sum(freq_energy)

    # 2. High-Frequency Energy Ratio (HFER) above 6.5 kHz
    cutoff_freq = 6500.0
    hf_mask = freqs >= cutoff_freq
    hf_energy = np.sum(freq_energy[hf_mask])
    hf_energy_ratio = float(hf_energy / total_energy)

    # 3. Spectral Centroid
    spectral_centroid = float(np.sum(freqs * freq_energy) / total_energy)

    # 4. Spectral Roll-Off (85% and 95% energy threshold)
    cum_energy = np.cumsum(freq_energy)
    idx_85 = np.searchsorted(cum_energy, 0.85 * total_energy)
    idx_95 = np.searchsorted(cum_energy, 0.95 * total_energy)
    roll_off_85 = float(freqs[min(idx_85, num_bins - 1)])
    roll_off_95 = float(freqs[min(idx_95, num_bins - 1)])

    # 5. Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
    # Computed per frame and averaged
    epsilon = 1e-12
    log_power = np.log(power_spec + epsilon)
    geom_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(power_spec, axis=0) + epsilon
    flatness_per_frame = geom_mean / arith_mean
    mean_flatness = float(np.mean(flatness_per_frame))

    return {
        "hf_energy_ratio": round(hf_energy_ratio, 5),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "spectral_rolloff_85_hz": round(roll_off_85, 1),
        "spectral_rolloff_95_hz": round(roll_off_95, 1),
        "spectral_flatness": round(mean_flatness, 5),
    }


def extract_acoustic_forensic_features(
    signal: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Comprehensive acoustic feature extractor for forensic voice verification."""
    if len(signal) < 1600:  # < 100 ms
        # Pad short signal
        signal = np.pad(signal, (0, 1600 - len(signal)), mode="constant")

    n_fft = 512
    hop_length = 160
    mag_spec, power_spec = compute_spectrogram(signal, n_fft=n_fft, hop_length=hop_length)

    # LFCC features
    lfcc = compute_lfcc(power_spec, sample_rate=sample_rate, n_fft=n_fft)
    lfcc_mean = np.mean(lfcc, axis=1)
    lfcc_var = np.var(lfcc, axis=1)

    # Pitch & Micro-perturbations
    f0, jitter, shimmer = extract_pitch_and_perturbation(signal, sample_rate=sample_rate)

    # Spectral indicators
    spec_metrics = compute_spectral_indicators(power_spec, sample_rate=sample_rate, n_fft=n_fft)

    return {
        "duration_sec": round(float(len(signal) / sample_rate), 2),
        "sample_rate": sample_rate,
        "mean_f0_hz": round(f0, 1),
        "pitch_jitter_pct": round(jitter * 100.0, 3),
        "pitch_shimmer_pct": round(shimmer * 100.0, 3),
        **spec_metrics,
        "lfcc_mean_c1_c5": [round(float(v), 3) for v in lfcc_mean[:5]],
        "lfcc_variance_mean": round(float(np.mean(lfcc_var)), 4),
    }
