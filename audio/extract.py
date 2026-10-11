"""Audio Demuxing and Extraction Engine.

Extracts, decodes, and resamples audio streams from standalone audio files
(WAV, MP3, FLAC, OGG, M4A) and video containers (MP4, MOV, WebM, AVI)
into standardized 16 kHz mono float32 waveforms.
"""

from typing import Optional, Tuple
import io
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def is_supported_audio_media(filename_or_ext: str) -> bool:
    """Check if file extension corresponds to a supported audio track or format."""
    ext = "." + filename_or_ext.lower().split(".")[-1] if "." in filename_or_ext else filename_or_ext.lower()
    return ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS


def _resample_linear(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Fast 1D linear resampling fallback if resampler is unavailable."""
    if orig_sr == target_sr or len(signal) == 0:
        return signal
    duration = len(signal) / float(orig_sr)
    target_len = int(round(duration * target_sr))
    orig_times = np.linspace(0, duration, len(signal), endpoint=False)
    target_times = np.linspace(0, duration, target_len, endpoint=False)
    return np.interp(target_times, orig_times, signal).astype(np.float32)


def extract_audio_from_bytes(
    file_bytes: bytes,
    target_sr: int = 16000,
) -> Tuple[Optional[np.ndarray], int]:
    """Extract audio track from raw bytes (audio or video container) into 16 kHz mono float32.

    Args:
        file_bytes: Raw binary bytes of audio or video container.
        target_sr: Desired output sample rate in Hz (default 16000).

    Returns:
        Tuple of (waveform_1d_float32, sample_rate).
        If no audio stream exists or container cannot be decoded, returns (None, 0).
    """
    if not file_bytes or len(file_bytes) < 32:
        return None, 0

    # 1. Attempt decoding with soundfile (fastest for WAV / FLAC / OGG)
    if HAS_SOUNDFILE:
        try:
            with sf.SoundFile(io.BytesIO(file_bytes)) as f:
                raw_data = f.read(dtype="float32")
                orig_sr = f.samplerate
                # Downmix multi-channel to mono
                if raw_data.ndim > 1:
                    mono = np.mean(raw_data, axis=1)
                else:
                    mono = raw_data
                if orig_sr != target_sr:
                    mono = _resample_linear(mono, orig_sr, target_sr)
                # Normalize peak to [-1.0, 1.0] if non-silent
                max_val = np.max(np.abs(mono))
                if max_val > 1e-6:
                    mono = mono / max_val
                return mono.astype(np.float32), target_sr
        except Exception:
            pass  # Fall through to PyAV for video containers or compressed audio

    # 2. Attempt demuxing & decoding with PyAV (handles MP4, MOV, WebM, MP3, etc.)
    if HAS_AV:
        try:
            container = av.open(io.BytesIO(file_bytes))
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return None, 0

            audio_stream = audio_streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)

            pcm_chunks = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    frame.pts = None
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames:
                        for rf in resampled_frames:
                            arr = rf.to_ndarray()
                            pcm_chunks.append(arr.flatten())

            # Flush any remaining resampled frames
            flushed = resampler.resample(None)
            if flushed:
                for rf in flushed:
                    pcm_chunks.append(rf.to_ndarray().flatten())

            if not pcm_chunks:
                return None, 0

            mono = np.concatenate(pcm_chunks).astype(np.float32)
            max_val = np.max(np.abs(mono))
            if max_val > 1e-6:
                mono = mono / max_val
            return mono, target_sr
        except Exception:
            pass

    return None, 0


def generate_synthetic_test_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    is_fake: bool = False,
) -> Tuple[np.ndarray, int]:
    """Generate realistic test speech waveform for unit testing and CI pipelines.

    Bona fide simulation:
      - Harmonic glottal flow (fundamental 130 Hz with natural pitch jitter ~1.1% & shimmer ~3.2%).
      - Vocal tract formants (F1=500, F2=1500, F3=2500, F4=3500 Hz).
      - Unvoiced fricative high-frequency breath noise extending cleanly to Nyquist (8 kHz).

    Synthetic vocoder simulation:
      - Perfectly quantized flat pitch (zero natural jitter).
      - Steep brickwall frequency cutoff at 6.8 kHz (typical 80-bin Mel vocoder void).
      - Transposed convolution periodic spectral phase ripple.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    f0_base = 130.0  # Typical male/neutral vocal fundamental frequency

    if not is_fake:
        # Natural human speech: micro-tremor jitter and amplitude shimmer
        jitter = 0.012 * np.sin(2 * np.pi * 5.5 * t) + 0.006 * np.random.randn(num_samples)
        f0 = f0_base * (1.0 + jitter)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate

        # Natural harmonic series with decaying glottal pulse amplitudes
        harmonics = np.zeros(num_samples)
        for h in range(1, 15):
            amp = (1.0 / h) * (1.0 + 0.03 * np.sin(2 * np.pi * 3.0 * t))
            harmonics += amp * np.sin(h * phase)

        # Formant filters (vocal tract resonance)
        formant_boost = (
            0.5 * np.sin(2 * np.pi * 500 * t)
            + 0.3 * np.sin(2 * np.pi * 1500 * t)
            + 0.2 * np.sin(2 * np.pi * 2500 * t)
        )
        # Wideband fricative breath noise extending to 8 kHz
        fricative = 0.08 * np.random.randn(num_samples)
        audio = harmonics + formant_boost + fricative
    else:
        # Deepfake vocoder: quantized flat pitch (zero micro-jitter)
        phase = 2 * np.pi * f0_base * t
        harmonics = np.zeros(num_samples)
        for h in range(1, 12):
            harmonics += (1.0 / h) * np.sin(h * phase)

        # Neural vocoder artifact: 6.8 kHz brickwall cutoff (zero high-frequency breath)
        # Periodic checkerboard phase ripple from transposed convolution
        vocoder_ripple = 0.05 * np.sin(2 * np.pi * 1800 * t) * np.cos(2 * np.pi * 400 * t)
        audio = harmonics + vocoder_ripple

        # FFT brickwall filter to simulate vocoder frequency truncation (> 6800 Hz)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
        fft[freqs > 6800] = 0.0
        audio = np.fft.irfft(fft, n=num_samples)

    # Normalize to [-1.0, 1.0]
    audio = audio / np.max(np.abs(audio) + 1e-7)
    return audio.astype(np.float32), sample_rate
