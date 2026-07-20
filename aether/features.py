"""
Per-event feature extraction (pipeline step 4–5).

For every detected segment extract:
  - Pitch (YIN / piptrack) + note name
  - Spectral: centroid, rolloff, flatness, contrast, bandwidth
  - Timbre: 13 MFCCs + delta means
  - Waveform estimate (rule-based spectral shape)
  - Envelope: attack / decay
  - Effects proxies: reverb tail, distortion, harmonic richness
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from aether.config import AnalysisSettings, N_MFCC
from aether.notes import hz_to_note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    if i1 <= i0:
        return np.zeros(1, dtype=np.float32)
    return y[i0:i1]


def _safe_fft_hop(seg_len: int, n_fft: int, hop_length: int) -> tuple[int, int]:
    """Pick n_fft / hop that fit short segments without librosa warnings."""
    n_fft_use = int(min(n_fft, 2 ** int(np.floor(np.log2(max(seg_len, 4))))))
    n_fft_use = max(n_fft_use, 256)
    if n_fft_use > seg_len:
        n_fft_use = max(64, int(2 ** int(np.floor(np.log2(max(seg_len, 4))))))
    hop_use = min(hop_length, max(n_fft_use // 4, 32))
    return n_fft_use, hop_use


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def extract_pitch(
    seg: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> tuple[Optional[float], list[float], Optional[str]]:
    """Average pitch (Hz), short contour, nearest note name via YIN (+ piptrack fallback)."""
    if len(seg) < int(0.02 * sr):
        return None, [], None

    try:
        f0 = librosa.yin(seg, fmin=fmin, fmax=fmax, sr=sr)
        valid = f0[np.isfinite(f0) & (f0 > 0)]
        if valid.size == 0:
            pitches, mags = librosa.piptrack(y=seg, sr=sr, fmin=fmin, fmax=fmax)
            pitch_vals: list[float] = []
            for t in range(pitches.shape[1]):
                idx = int(np.argmax(mags[:, t]))
                p = float(pitches[idx, t])
                if p > 0:
                    pitch_vals.append(p)
            if not pitch_vals:
                return None, [], None
            avg = float(np.median(pitch_vals))
            return avg, pitch_vals[:64], hz_to_note(avg)

        avg = float(np.median(valid))
        contour = [
            float(x) if np.isfinite(x) and x > 0 else 0.0 for x in f0[:128]
        ]
        return avg, contour, hz_to_note(avg)
    except Exception:
        return None, [], None


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

def extract_spectral(
    seg: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, Any]:
    empty = {
        "spectral_centroid": 0.0,
        "spectral_rolloff": 0.0,
        "spectral_flatness": 0.0,
        "spectral_contrast": [],
        "spectral_bandwidth": 0.0,
    }
    if len(seg) < 4:
        return empty

    n_fft_use, hop_use = _safe_fft_hop(len(seg), n_fft, hop_length)
    try:
        cent = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=seg, sr=sr, n_fft=n_fft_use, hop_length=hop_use
                )
            )
        )
        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=seg, sr=sr, n_fft=n_fft_use, hop_length=hop_use
                )
            )
        )
        flat = float(
            np.mean(
                librosa.feature.spectral_flatness(
                    y=seg, n_fft=n_fft_use, hop_length=hop_use
                )
            )
        )
        contrast = librosa.feature.spectral_contrast(
            y=seg, sr=sr, n_fft=n_fft_use, hop_length=hop_use
        )
        contrast_mean = [float(x) for x in np.mean(contrast, axis=1)]
        bw = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=seg, sr=sr, n_fft=n_fft_use, hop_length=hop_use
                )
            )
        )
    except Exception:
        return empty

    return {
        "spectral_centroid": round(cent, 2),
        "spectral_rolloff": round(rolloff, 2),
        "spectral_flatness": round(flat, 6),
        "spectral_contrast": [round(c, 3) for c in contrast_mean],
        "spectral_bandwidth": round(bw, 2),
    }


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

def extract_mfcc(
    seg: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, Any]:
    zeros = [0.0] * n_mfcc
    if len(seg) < 4:
        return {"mfcc_mean": zeros, "mfcc_std": zeros, "mfcc_delta_mean": zeros}

    n_fft_use, hop_use = _safe_fft_hop(len(seg), n_fft, hop_length)
    mfcc = librosa.feature.mfcc(
        y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft_use, hop_length=hop_use
    )

    # delta() requires width < n_frames
    n_frames = mfcc.shape[1]
    if n_frames >= 9:
        delta = librosa.feature.delta(mfcc)
        delta_mean = [round(float(x), 5) for x in np.mean(delta, axis=1)]
    elif n_frames >= 3:
        width = n_frames if n_frames % 2 == 1 else n_frames - 1
        width = max(3, width)
        try:
            delta = librosa.feature.delta(mfcc, width=width, mode="nearest")
            delta_mean = [round(float(x), 5) for x in np.mean(delta, axis=1)]
        except Exception:
            delta_mean = zeros
    else:
        delta_mean = zeros

    return {
        "mfcc_mean": [round(float(x), 5) for x in np.mean(mfcc, axis=1)],
        "mfcc_std": [round(float(x), 5) for x in np.std(mfcc, axis=1)],
        "mfcc_delta_mean": delta_mean,
    }


# ---------------------------------------------------------------------------
# Waveform estimate (rule-based)
# ---------------------------------------------------------------------------

def estimate_waveform_type(
    seg: np.ndarray,
    sr: int,
    n_fft: int = 4096,
) -> str:
    """
    Rule-based waveform class from spectral shape / harmonic structure.

    Returns one of: sine, triangle, sawtooth, square, noise, complex.
    """
    if len(seg) < int(0.01 * sr):
        return "complex"

    n = max(n_fft, 2 ** int(np.ceil(np.log2(max(len(seg), 4)))))
    windowed = seg * np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(windowed, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    if spectrum.sum() < 1e-12:
        return "complex"

    # Spectral flatness → noise
    power = spectrum ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    flatness = geo / arith
    if flatness > 0.4:
        return "noise"

    peak_idx, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.05, distance=3)
    if len(peak_idx) < 2:
        return "sine"

    peak_freqs = freqs[peak_idx]
    peak_amps = spectrum[peak_idx]
    order = np.argsort(peak_amps)[::-1]
    peak_freqs = peak_freqs[order]
    f0 = float(peak_freqs[0])
    if f0 < 20:
        return "complex"

    def amp_at(harm: int) -> float:
        target = harm * f0
        if target >= freqs[-1]:
            return 0.0
        i = int(np.argmin(np.abs(freqs - target)))
        lo, hi = max(0, i - 2), min(len(spectrum), i + 3)
        return float(np.max(spectrum[lo:hi]))

    a1 = amp_at(1) + 1e-12
    ratios = [amp_at(h) / a1 for h in range(1, 8)]

    odd = sum(ratios[i] for i in range(0, 7, 2))
    even = sum(ratios[i] for i in range(1, 7, 2))
    odd_ratio = odd / (odd + even + 1e-12)
    higher = sum(ratios[1:])

    if higher < 0.15:
        return "sine"

    hs = np.array([2, 3, 4, 5], dtype=float)
    amps = np.array([ratios[1], ratios[2], ratios[3], ratios[4]], dtype=float)
    mask = amps > 1e-6
    if mask.sum() >= 2:
        slope = float(np.polyfit(np.log(hs[mask]), np.log(amps[mask]), 1)[0])
    else:
        slope = -1.0

    if odd_ratio > 0.75 and higher > 0.2:
        return "square"
    if slope < -1.6:
        return "triangle"
    if -1.6 <= slope <= -0.5:
        return "sawtooth"
    return "complex"


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def extract_envelope(
    seg: np.ndarray,
    sr: int,
    hop_length: int = 256,
) -> dict[str, Any]:
    """Approximate attack / decay times from RMS envelope."""
    if len(seg) < hop_length:
        return {"attack_time": 0.0, "decay_time": 0.0, "peak_time": 0.0}

    rms = librosa.feature.rms(y=seg, hop_length=hop_length)[0]
    if rms.size == 0 or float(np.max(rms)) <= 0:
        return {"attack_time": 0.0, "decay_time": 0.0, "peak_time": 0.0}

    rms_n = rms / np.max(rms)
    peak_frame = int(np.argmax(rms_n))
    peak_time = float(librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length))

    pre = rms_n[: peak_frame + 1]
    attack = peak_time
    if pre.size > 1:
        try:
            t10 = int(np.where(pre >= 0.1)[0][0])
            t90 = int(np.where(pre >= 0.9)[0][0])
            attack = float(
                librosa.frames_to_time(max(0, t90 - t10), sr=sr, hop_length=hop_length)
            )
        except IndexError:
            attack = peak_time

    post = rms_n[peak_frame:]
    decay = 0.0
    if post.size > 1:
        below = np.where(post <= 0.1)[0]
        if below.size:
            decay = float(librosa.frames_to_time(int(below[0]), sr=sr, hop_length=hop_length))
        else:
            decay = float(
                librosa.frames_to_time(len(post) - 1, sr=sr, hop_length=hop_length)
            )

    return {
        "attack_time": round(max(0.0, attack), 4),
        "decay_time": round(max(0.0, decay), 4),
        "peak_time": round(peak_time, 4),
    }


# ---------------------------------------------------------------------------
# Effects proxies
# ---------------------------------------------------------------------------

def estimate_effects(
    seg: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, Any]:
    """
    Rule-based effects estimates:
      - reverb_tail: late vs early post-peak energy
      - distortion: high-band + flatness proxy
      - harmonic_richness: spectral contrast mid/high bands
    """
    empty = {"reverb_tail": 0.0, "distortion": 0.0, "harmonic_richness": 0.0}
    if len(seg) < 256:
        return empty

    n_fft_use, hop_use = _safe_fft_hop(len(seg), n_fft, hop_length)
    rms = librosa.feature.rms(y=seg, hop_length=hop_use)[0]
    peak = int(np.argmax(rms)) if rms.size else 0
    post = rms[peak:] if peak < len(rms) else rms

    if post.size < 3:
        reverb = 0.0
    else:
        n = len(post)
        first = float(np.mean(post[: max(1, n // 3)]) + 1e-12)
        last = float(np.mean(post[-(max(1, n // 3)) :]))
        reverb = float(np.clip(last / first, 0.0, 1.0))

    try:
        flat = float(
            np.mean(
                librosa.feature.spectral_flatness(
                    y=seg, n_fft=n_fft_use, hop_length=hop_use
                )
            )
        )
        contrast = librosa.feature.spectral_contrast(
            y=seg, sr=sr, n_fft=n_fft_use, hop_length=hop_use
        )
        richness = (
            float(np.mean(contrast[2:]))
            if contrast.shape[0] > 2
            else float(np.mean(contrast))
        )
        richness_n = float(np.clip(richness / 40.0, 0.0, 1.0))
        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=seg,
                    sr=sr,
                    n_fft=n_fft_use,
                    hop_length=hop_use,
                    roll_percent=0.95,
                )
            )
        )
    except Exception:
        return {**empty, "reverb_tail": round(reverb, 4)}

    nyq = sr / 2.0
    high_share = float(np.clip(rolloff / nyq, 0.0, 1.0))
    distortion = float(np.clip(0.5 * high_share + 0.5 * flat * 2.0, 0.0, 1.0))

    return {
        "reverb_tail": round(reverb, 4),
        "distortion": round(distortion, 4),
        "harmonic_richness": round(richness_n, 4),
    }


# ---------------------------------------------------------------------------
# Spectral flux (motion / riser cue)
# ---------------------------------------------------------------------------

def extract_spectral_flux(
    seg: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Mean positive spectral flux — higher = more spectral change over time.
    Used as a classical cue for risers / evolving textures.
    """
    if len(seg) < hop_length * 3:
        return 0.0
    n_fft_use, hop_use = _safe_fft_hop(len(seg), n_fft, hop_length)
    try:
        S = np.abs(librosa.stft(seg, n_fft=n_fft_use, hop_length=hop_use))
        if S.shape[1] < 2:
            return 0.0
        # Half-wave rectified frame-to-frame diff, normalized
        diff = np.diff(S, axis=1)
        flux = np.maximum(0.0, diff).sum(axis=0)
        # Normalize by spectral energy
        energy = S[:, 1:].sum(axis=0) + 1e-12
        flux_n = flux / energy
        return round(float(np.mean(flux_n)), 5)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_event_features(
    y: np.ndarray,
    sr: int,
    event: dict[str, Any],
    settings: Optional[AnalysisSettings] = None,
) -> dict[str, Any]:
    """Enrich a single event dict with all feature categories."""
    settings = settings or AnalysisSettings()
    start = float(event["start_time"])
    end = float(event["end_time"])
    seg = _slice(y, sr, start, end)

    pitch_hz, contour, note = extract_pitch(seg, sr)
    spectral = extract_spectral(
        seg, sr, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    mfcc = extract_mfcc(
        seg,
        sr,
        n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    )
    waveform = estimate_waveform_type(seg, sr)
    envelope = extract_envelope(seg, sr)
    effects = estimate_effects(
        seg, sr, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    flux = extract_spectral_flux(
        seg, sr, n_fft=settings.n_fft, hop_length=settings.hop_length
    )

    out = dict(event)
    out.update(
        {
            "pitch_hz": round(pitch_hz, 2) if pitch_hz is not None else None,
            "pitch_contour": contour,
            "note": note,
            "waveform_estimate": waveform,
            "spectral_flux": flux,
            "similarity_score_to_library": None,
            **spectral,
            **mfcc,
            **envelope,
            **effects,
        }
    )
    return out


def extract_all_event_features(
    y: np.ndarray,
    sr: int,
    events: list[dict[str, Any]],
    settings: Optional[AnalysisSettings] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[dict[str, Any]]:
    """Extract features for every event; optional progress in [0, 1]."""
    results: list[dict[str, Any]] = []
    n = len(events)
    for i, ev in enumerate(events):
        results.append(extract_event_features(y, sr, ev, settings=settings))
        if progress_callback is not None:
            progress_callback((i + 1) / max(n, 1))
    return results
