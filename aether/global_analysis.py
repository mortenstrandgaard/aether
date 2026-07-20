"""
Global track-level analysis (pipeline step 2).

- BPM via librosa.beat.beat_track
- Key via Essentia KeyExtractor (if available) or librosa chroma + KS templates
- RMS energy curve and spectral centroid over time
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np
from scipy.stats import pearsonr

from aether.config import (
    MAJOR_PROFILE,
    MINOR_PROFILE,
    NOTE_NAMES,
    AnalysisSettings,
)


def _essentia_key(y: np.ndarray, sr: int) -> Optional[str]:
    """Best-effort Essentia KeyExtractor. Returns None if unavailable."""
    try:
        import essentia.standard as es  # type: ignore
    except Exception:
        return None

    try:
        audio = y.astype(np.float32)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        key, scale, _strength = es.KeyExtractor()(audio)
        if key and scale:
            return f"{key} {scale}"
    except Exception:
        return None
    return None


def _librosa_key(y: np.ndarray, sr: int, hop_length: int = 512) -> str:
    """Chroma CQT + Krumhansl–Schmuckler template matching."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    total = float(np.sum(chroma_mean))
    if total > 0:
        chroma_mean = chroma_mean / (total + 1e-12)

    best_corr = -np.inf
    best_key = "C major"

    major = np.asarray(MAJOR_PROFILE, dtype=float)
    minor = np.asarray(MINOR_PROFILE, dtype=float)
    major = major / (np.sum(major) + 1e-12)
    minor = minor / (np.sum(minor) + 1e-12)

    for i, name in enumerate(NOTE_NAMES):
        maj_rot = np.roll(major, i)
        min_rot = np.roll(minor, i)
        r_maj, _ = pearsonr(chroma_mean, maj_rot)
        r_min, _ = pearsonr(chroma_mean, min_rot)
        # Guard NaN from flat chroma
        r_maj = float(r_maj) if np.isfinite(r_maj) else -1.0
        r_min = float(r_min) if np.isfinite(r_min) else -1.0
        if r_maj > best_corr:
            best_corr = r_maj
            best_key = f"{name} major"
        if r_min > best_corr:
            best_corr = r_min
            best_key = f"{name} minor"

    return best_key


def detect_bpm(y: np.ndarray, sr: int, hop_length: int = 512) -> float:
    """Estimate global BPM. Returns rounded float."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
    return round(float(tempo), 2)


def detect_key(y: np.ndarray, sr: int, hop_length: int = 512) -> str:
    """Estimate musical key; prefer Essentia when installed."""
    key = _essentia_key(y, sr)
    if key:
        return key
    return _librosa_key(y, sr, hop_length=hop_length)


def energy_curve(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """RMS energy over time → (times_sec, rms)."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times.astype(float), rms.astype(float)


def spectral_centroid_curve(
    y: np.ndarray, sr: int, hop_length: int = 512, n_fft: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral centroid over time → (times_sec, centroid_hz)."""
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
    )[0]
    times = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length)
    return times.astype(float), cent.astype(float)


def analyze_global(
    y: np.ndarray,
    sr: int,
    settings: Optional[AnalysisSettings] = None,
) -> dict[str, Any]:
    """Run all global track analysis and return a serializable dict."""
    settings = settings or AnalysisSettings()
    hop = settings.hop_length
    n_fft = settings.n_fft

    bpm = detect_bpm(y, sr, hop_length=hop)
    key = detect_key(y, sr, hop_length=hop)
    t_rms, rms = energy_curve(y, sr, hop_length=hop)
    t_cent, cent = spectral_centroid_curve(y, sr, hop_length=hop, n_fft=n_fft)

    return {
        "bpm": bpm,
        "key": key,
        "duration": float(len(y) / sr),
        "rms_times": t_rms.tolist(),
        "rms": rms.tolist(),
        "centroid_times": t_cent.tolist(),
        "centroid": cent.tolist(),
        "rms_mean": float(np.mean(rms)),
        "centroid_mean": float(np.mean(cent)),
    }
