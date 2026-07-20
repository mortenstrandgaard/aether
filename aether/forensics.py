"""
AETHER Forensics — classical DSP comparison of two (or many) sounds / speech clips.

Produces:
  - Overall match score 0–100%
  - Reliability 0–100% (how trustworthy the test conditions are)
  - Per-dimension scores + written report
  - 1-vs-N ranking (reference vs many candidates)

Works on ordinary speech (tale), singing, and general audio — Voice mode
optimises weights for speech-like material.

IMPORTANT LIMITATIONS (always disclosed):
  - NOT court-grade speaker identification / biometrics.
  - Same words not required; different text compares *voice colour*, not content.
  - Mic/room/codec strongly affect scores.
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np
from numpy.linalg import norm

from aether.config import AnalysisSettings, N_MFCC
from aether.features import (
    estimate_effects,
    extract_envelope,
    extract_mfcc,
    extract_pitch,
    extract_spectral,
    extract_spectral_flux,
)
from aether.loader import load_audio
from aether.similarity import dtw_mfcc_distance


DEFAULT_WEIGHTS = {
    "timbre_mfcc": 0.32,
    "pitch": 0.22,
    "spectral": 0.20,
    "envelope": 0.12,
    "energy_dynamics": 0.08,
    "temporal_align": 0.06,
}

VOICE_WEIGHTS = {
    "timbre_mfcc": 0.36,
    "pitch": 0.28,
    "spectral": 0.14,
    "envelope": 0.08,
    "energy_dynamics": 0.06,
    "temporal_align": 0.08,
}


# ---------------------------------------------------------------------------
# Preprocess (trim region, normalize, speech-oriented cleanup)
# ---------------------------------------------------------------------------

def preprocess_audio(
    y: np.ndarray,
    sr: int,
    *,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    peak_normalize: bool = True,
    rms_normalize: bool = True,
    target_rms: float = 0.1,
    trim_silence: bool = True,
    highpass_hz: float = 0.0,
    max_seconds: float = 45.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prepare a clip for fair comparison. Returns (y_out, meta).
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size == 0:
        raise ValueError("Empty audio")

    n = len(y)
    i0 = max(0, int(start_sec * sr))
    i1 = n if end_sec is None else min(n, int(end_sec * sr))
    if i1 <= i0:
        raise ValueError("Invalid region: end must be after start")
    y = y[i0:i1]

    meta: dict[str, Any] = {
        "region_start": round(i0 / sr, 3),
        "region_end": round(i1 / sr, 3),
        "raw_duration": round((i1 - i0) / sr, 3),
    }

    if highpass_hz and highpass_hz > 0:
        try:
            from scipy.signal import butter, filtfilt

            nyq = sr / 2.0
            wn = min(highpass_hz / nyq, 0.99)
            if 0 < wn < 1:
                b, a = butter(2, wn, btype="high")
                y = filtfilt(b, a, y).astype(np.float32)
                meta["highpass_hz"] = highpass_hz
        except Exception:
            meta["highpass_hz"] = None

    if trim_silence:
        try:
            yt, idx = librosa.effects.trim(y, top_db=32.0)
            if yt.size > int(0.15 * sr):
                y = yt
                meta["silence_trim"] = True
        except Exception:
            meta["silence_trim"] = False

    if len(y) > int(max_seconds * sr):
        y = y[: int(max_seconds * sr)]
        meta["capped_seconds"] = max_seconds

    if peak_normalize:
        peak = float(np.max(np.abs(y))) + 1e-12
        y = y / peak
        meta["peak_normalize"] = True

    if rms_normalize:
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        y = y * (target_rms / rms)
        # soft clip
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        meta["rms_normalize"] = True
        meta["target_rms"] = target_rms

    meta["final_duration"] = round(len(y) / sr, 3)
    meta["n_samples"] = int(len(y))
    return y.astype(np.float32), meta


def estimate_reliability(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    *,
    preprocess_a: Optional[dict] = None,
    preprocess_b: Optional[dict] = None,
) -> dict[str, Any]:
    """
    How trustworthy is this comparison given clip quality / length?
    Independent of match score — short/noisy clips → low reliability.
    """
    reasons: list[str] = []
    score = 1.0

    da = _safe(profile_a.get("duration"), 0.0)
    db = _safe(profile_b.get("duration"), 0.0)
    dmin = min(da, db)

    if dmin < 1.0:
        score *= 0.35
        reasons.append(f"Very short clip(s) ({dmin:.1f}s) — scores unstable.")
    elif dmin < 2.0:
        score *= 0.55
        reasons.append(f"Short clip(s) ({dmin:.1f}s) — prefer ≥2–3s of speech.")
    elif dmin < 3.0:
        score *= 0.75
        reasons.append("Clip length OK but thin; ≥3–5s is better for speech.")
    else:
        reasons.append(f"Length OK (shortest clip {dmin:.1f}s).")

    # Pitch stability / presence (speech should have some voicing)
    va = _safe(profile_a.get("voiced_ratio"))
    vb = _safe(profile_b.get("voiced_ratio"))
    if va < 0.15 or vb < 0.15:
        score *= 0.7
        reasons.append("Low voicing proxy — music, noise, or whisper may dominate.")
    elif min(va, vb) >= 0.35:
        reasons.append("Voicing present in both clips.")

    # Extreme flatness → noisy / not speech-like
    fa = _safe(profile_a.get("spectral_flatness"))
    fb = _safe(profile_b.get("spectral_flatness"))
    if fa > 0.35 or fb > 0.35:
        score *= 0.75
        reasons.append("High spectral flatness (noisy / wideband) reduces confidence.")

    # Duration mismatch
    if da > 0 and db > 0:
        ratio = max(da, db) / min(da, db)
        if ratio > 4:
            score *= 0.8
            reasons.append("Very different lengths — comparing unequal amounts of material.")

    # Pitch missing on both
    if not profile_a.get("pitch_hz") and not profile_b.get("pitch_hz"):
        score *= 0.85
        reasons.append("No stable pitch detected — unpitched material or heavy processing.")

    pct = round(100.0 * float(np.clip(score, 0.05, 1.0)), 1)
    if pct >= 75:
        label = "High — conditions look reasonable for a rough acoustic compare"
    elif pct >= 50:
        label = "Medium — usable with caution"
    elif pct >= 30:
        label = "Low — treat the match % as a weak hint only"
    else:
        label = "Very low — re-record longer/cleaner clips if possible"

    return {
        "reliability_pct": pct,
        "label": label,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _sim_from_cosine(c: float) -> float:
    return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))


def _sim_from_rel_diff(a: float, b: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(np.exp(-abs(a - b) / scale), 0.0, 1.0))


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def extract_forensic_profile(
    y: np.ndarray,
    sr: int,
    settings: Optional[AnalysisSettings] = None,
    label: str = "sample",
    max_seconds: float = 45.0,
    *,
    mode: str = "sound",
    already_preprocessed: bool = False,
    preprocess_meta: Optional[dict] = None,
) -> dict[str, Any]:
    """Build forensic feature profile from mono waveform."""
    settings = settings or AnalysisSettings()
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size == 0:
        raise ValueError(f"Empty audio for '{label}'")

    if not already_preprocessed:
        y, preprocess_meta = preprocess_audio(
            y,
            sr,
            peak_normalize=True,
            rms_normalize=True,
            trim_silence=True,
            highpass_hz=80.0 if mode == "voice" else 0.0,
            max_seconds=max_seconds,
        )
    preprocess_meta = preprocess_meta or {}

    # Pitch range: speech-friendly in voice mode
    fmin, fmax = (70.0, 450.0) if mode == "voice" else (50.0, 2000.0)

    spectral = extract_spectral(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    mfcc = extract_mfcc(
        y, sr, n_mfcc=settings.n_mfcc, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    envelope = extract_envelope(y, sr)
    effects = estimate_effects(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    flux = extract_spectral_flux(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    pitch_hz, contour, note = extract_pitch(y, sr, fmin=fmin, fmax=fmax)

    pitch_vals = [p for p in contour if p and p > 0]
    if pitch_hz and pitch_hz > 0 and not pitch_vals:
        pitch_vals = [pitch_hz]

    if pitch_vals:
        p_arr = np.asarray(pitch_vals, dtype=float)
        pitch_mean = float(np.median(p_arr))
        pitch_std = float(np.std(p_arr))
        pitch_min = float(np.min(p_arr))
        pitch_max = float(np.max(p_arr))
        pitch_range = pitch_max - pitch_min
    else:
        pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0

    # Downsample contour for UI plots (time axis in seconds)
    hop_pitch = max(1, len(contour) // 128) if contour else 1
    pitch_times = []
    pitch_plot = []
    if contour:
        # yin hop ≈ frame length; approximate with librosa default spacing
        # Use linspace over duration for display
        dur = len(y) / sr
        n_c = len(contour)
        for i in range(0, n_c, max(1, n_c // 128)):
            pitch_times.append(round(dur * i / max(n_c - 1, 1), 3))
            pitch_plot.append(round(float(contour[i]), 2) if contour[i] and contour[i] > 0 else None)

    n_fft_use = min(settings.n_fft, max(256, 2 ** int(np.floor(np.log2(max(len(y), 4))))))
    hop = min(settings.hop_length, max(n_fft_use // 4, 64))
    mfcc_mat = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=settings.n_mfcc, n_fft=n_fft_use, hop_length=hop
    ).T

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_cv = float(rms_std / (rms_mean + 1e-12))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop)))

    try:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft_use, hop_length=hop, n_mels=40
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mel_mean = [float(x) for x in np.mean(log_mel, axis=1)]
    except Exception:
        mel_mean = [0.0] * 40

    flat = _safe(spectral.get("spectral_flatness"))
    voiced_ratio = 0.0
    if pitch_vals and contour:
        voiced_ratio = float(np.clip(len(pitch_vals) / max(len(contour), 1), 0.0, 1.0))
    if flat < 0.12 and pitch_mean > 0:
        voiced_ratio = max(voiced_ratio, 0.5)

    duration = float(len(y) / sr)

    # Keep waveform for viz (cap for memory)
    max_plot = int(min(len(y), 15 * sr))
    y_plot = y[:max_plot].copy()

    return {
        "label": label,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "pitch_hz": round(pitch_mean, 2) if pitch_mean > 0 else None,
        "pitch_std": round(pitch_std, 2),
        "pitch_min": round(pitch_min, 2),
        "pitch_max": round(pitch_max, 2),
        "pitch_range": round(pitch_range, 2),
        "pitch_times": pitch_times,
        "pitch_contour_plot": pitch_plot,
        "note": note,
        "zcr": round(zcr, 5),
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "rms_cv": round(rms_cv, 4),
        "spectral_flux": flux,
        "voiced_ratio": round(voiced_ratio, 3),
        "mel_mean": mel_mean,
        "mfcc_matrix": mfcc_mat.astype(np.float32),
        "y_plot": y_plot,
        "preprocess": preprocess_meta,
        **spectral,
        **mfcc,
        **envelope,
        **effects,
    }


def extract_forensic_profile_from_path(
    path: str,
    settings: Optional[AnalysisSettings] = None,
    label: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    y, sr, meta = load_audio(path, settings=settings)
    return extract_forensic_profile(
        y, sr, settings=settings, label=label or meta.get("filename") or path, **kwargs
    )


# ---------------------------------------------------------------------------
# Dimension scores
# ---------------------------------------------------------------------------

def _timbre_score(a: dict, b: dict) -> float:
    va = np.concatenate(
        [
            np.asarray(a.get("mfcc_mean") or [0.0] * N_MFCC, dtype=float),
            np.asarray(a.get("mfcc_delta_mean") or [0.0] * N_MFCC, dtype=float),
            np.asarray(a.get("mel_mean") or [0.0] * 40, dtype=float) / 80.0,
        ]
    )
    vb = np.concatenate(
        [
            np.asarray(b.get("mfcc_mean") or [0.0] * N_MFCC, dtype=float),
            np.asarray(b.get("mfcc_delta_mean") or [0.0] * N_MFCC, dtype=float),
            np.asarray(b.get("mel_mean") or [0.0] * 40, dtype=float) / 80.0,
        ]
    )
    return _sim_from_cosine(_cosine(va, vb))


def _pitch_score(a: dict, b: dict) -> float:
    pa, pb = a.get("pitch_hz"), b.get("pitch_hz")
    if not pa or not pb or pa <= 0 or pb <= 0:
        if (not pa or pa <= 0) and (not pb or pb <= 0):
            return 0.65
        return 0.25

    cents = 1200.0 * abs(np.log2(pa / pb))
    mean_sim = float(np.clip(np.exp(-cents / 350.0), 0.0, 1.0))
    std_sim = _sim_from_rel_diff(_safe(a.get("pitch_std")), _safe(b.get("pitch_std")), 40.0)
    range_sim = _sim_from_rel_diff(
        _safe(a.get("pitch_range")), _safe(b.get("pitch_range")), 80.0
    )

    # Contour shape: compare normalized valid pitch sequences (speech intonation)
    ca = [p for p in (a.get("pitch_contour_plot") or []) if p]
    cb = [p for p in (b.get("pitch_contour_plot") or []) if p]
    contour_sim = 0.5
    if len(ca) >= 4 and len(cb) >= 4:
        # resample to 32 points, convert to cents relative to median
        def norm_c(seq):
            arr = np.asarray(seq, dtype=float)
            idx = np.linspace(0, len(arr) - 1, 32).astype(int)
            arr = arr[idx]
            med = np.median(arr)
            return 1200.0 * np.log2(arr / (med + 1e-9))

        try:
            na, nb = norm_c(ca), norm_c(cb)
            dist = float(np.mean(np.abs(na - nb)))
            contour_sim = float(np.clip(np.exp(-dist / 200.0), 0.0, 1.0))
        except Exception:
            contour_sim = 0.5

    return float(0.40 * mean_sim + 0.20 * std_sim + 0.15 * range_sim + 0.25 * contour_sim)


def _spectral_score(a: dict, b: dict) -> float:
    parts = [
        _sim_from_rel_diff(
            _safe(a.get("spectral_centroid")), _safe(b.get("spectral_centroid")), 1500.0
        ),
        _sim_from_rel_diff(
            _safe(a.get("spectral_rolloff")), _safe(b.get("spectral_rolloff")), 2500.0
        ),
        _sim_from_rel_diff(
            _safe(a.get("spectral_flatness")), _safe(b.get("spectral_flatness")), 0.15
        ),
        _sim_from_rel_diff(
            _safe(a.get("spectral_bandwidth")), _safe(b.get("spectral_bandwidth")), 1500.0
        ),
        _sim_from_rel_diff(_safe(a.get("zcr")), _safe(b.get("zcr")), 0.08),
    ]
    ca = np.asarray(a.get("spectral_contrast") or [], dtype=float)
    cb = np.asarray(b.get("spectral_contrast") or [], dtype=float)
    if ca.size and cb.size and ca.size == cb.size:
        parts.append(_sim_from_cosine(_cosine(ca, cb)))
    return float(np.mean(parts))


def _envelope_score(a: dict, b: dict) -> float:
    return float(
        np.mean(
            [
                _sim_from_rel_diff(
                    _safe(a.get("attack_time")), _safe(b.get("attack_time")), 0.08
                ),
                _sim_from_rel_diff(
                    _safe(a.get("decay_time")), _safe(b.get("decay_time")), 0.25
                ),
                _sim_from_rel_diff(
                    _safe(a.get("duration"), 1.0), _safe(b.get("duration"), 1.0), 2.0
                ),
            ]
        )
    )


def _energy_score(a: dict, b: dict) -> float:
    return float(
        np.mean(
            [
                _sim_from_rel_diff(_safe(a.get("rms_cv")), _safe(b.get("rms_cv")), 0.35),
                _sim_from_rel_diff(
                    _safe(a.get("spectral_flux")), _safe(b.get("spectral_flux")), 0.12
                ),
                _sim_from_rel_diff(
                    _safe(a.get("reverb_tail")), _safe(b.get("reverb_tail")), 0.25
                ),
            ]
        )
    )


def _temporal_score(a: dict, b: dict) -> float:
    ma, mb = a.get("mfcc_matrix"), b.get("mfcc_matrix")
    if ma is None or mb is None:
        return 0.5
    dist = dtw_mfcc_distance(np.asarray(ma), np.asarray(mb))
    return float(1.0 / (1.0 + dist))


def _verdict(score_pct: float, mode: str) -> str:
    if mode == "voice":
        if score_pct >= 85:
            return "Strong acoustic similarity (same source plausible — not proven)"
        if score_pct >= 70:
            return "Moderate–high similarity (overlapping voice-like traits)"
        if score_pct >= 55:
            return "Mixed / inconclusive similarity"
        if score_pct >= 40:
            return "Weak similarity (notable differences)"
        return "Low similarity (likely different acoustic sources or conditions)"
    if score_pct >= 85:
        return "Very high match — timbre/pitch/shape align closely"
    if score_pct >= 70:
        return "Strong match — good candidate as related or similar sound"
    if score_pct >= 55:
        return "Moderate match — some shared traits, clear differences remain"
    if score_pct >= 40:
        return "Weak match — mostly different character"
    return "Low match — dissimilar acoustic profiles"


def _dimension_comment(name: str, pct: float, a: dict, b: dict) -> str:
    if name == "timbre_mfcc":
        if pct >= 80:
            return "MFCC/mel envelopes align closely — similar timbre / vocal colour."
        if pct >= 55:
            return "Partial timbre overlap; some colouration or mic difference likely."
        return "Timbre profiles diverge — different body, instrument, or voice colour."
    if name == "pitch":
        pa, pb = a.get("pitch_hz"), b.get("pitch_hz")
        if not pa or not pb:
            return "Pitch comparison limited (one or both clips lack stable pitch)."
        if pct >= 80:
            return f"Pitch centres close ({pa:.0f} Hz vs {pb:.0f} Hz); intonation shape related."
        if pct >= 55:
            return f"Pitch related but shifted ({pa:.0f} Hz vs {pb:.0f} Hz) or different melody."
        return f"Pitch centres differ substantially ({pa:.0f} Hz vs {pb:.0f} Hz)."
    if name == "spectral":
        if pct >= 80:
            return "Brightness, noise-floor and bandwidth are highly consistent."
        if pct >= 55:
            return "Spectral balance partly shared; EQ, distance or room may differ."
        return "Spectral shape differs (brighter/darker or noisier vs cleaner)."
    if name == "envelope":
        if pct >= 80:
            return "Attack/decay timing is similar (same articulation style)."
        if pct >= 55:
            return "Envelope roughly related; phrasing or clip length differs."
        return "Dynamic shape differs (pluck vs sustain, short vs long, etc.)."
    if name == "energy_dynamics":
        if pct >= 80:
            return "Loudness variation and motion over time look consistent."
        return "Energy behaviour differs (steadier vs more dynamic, or different room tail)."
    if name == "temporal_align":
        if pct >= 80:
            return "MFCC sequence alignment (DTW) is strong — evolution over time matches."
        if pct >= 55:
            return (
                "Moderate temporal alignment. For speech: different words lower this "
                "even if the same person — timbre/pitch matter more."
            )
        return (
            "Time evolution does not align (different words, timing, or material). "
            "Normal for speech with different text."
        )
    return ""


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_analysis_report(result: dict[str, Any], language: str = "en") -> str:
    a = result["profile_a"]
    b = result["profile_b"]
    score = result["score_pct"]
    dims = result["dimensions"]
    mode = result.get("mode", "sound")
    verdict = result["verdict"]
    rel = result.get("reliability") or {}

    if language == "da":
        return _report_da(a, b, score, dims, mode, verdict, result, rel)
    return _report_en(a, b, score, dims, mode, verdict, result, rel)


def _report_en(a, b, score, dims, mode, verdict, result, rel) -> str:
    lines = [
        "AETHER FORENSICS REPORT",
        "=" * 48,
        f"Mode: {mode.upper()}",
        f"Sample A: {a.get('label', 'A')}  ({a.get('duration', 0):.2f}s)",
        f"Sample B: {b.get('label', 'B')}  ({b.get('duration', 0):.2f}s)",
        f"Overall match score: {score:.1f}%",
        f"Reliability: {rel.get('reliability_pct', '—')}% — {rel.get('label', '')}",
        f"Verdict: {verdict}",
        "",
        "SPEECH / TALE NOTE",
        "-" * 48,
        "Ordinary speech works in VOICE mode. You do NOT need the same words.",
        "Different sentences still compare vocal colour (timbre, pitch range).",
        "DTW/temporal score often drops when text differs — that is expected.",
        "Best: ≥3s dry speech per clip, similar phone/mic, little background music.",
        "",
        "DISCLAIMER",
        "-" * 48,
        "Classical DSP only. NOT court-grade speaker ID. Guidance only, not legal proof.",
        "",
        "RELIABILITY REASONS",
        "-" * 48,
    ]
    for r in rel.get("reasons") or []:
        lines.append(f"  • {r}")
    lines += ["", "DIMENSION BREAKDOWN", "-" * 48]
    for key, label in [
        ("timbre_mfcc", "Timbre (MFCC / mel)"),
        ("pitch", "Pitch / intonation"),
        ("spectral", "Spectral shape"),
        ("envelope", "Envelope / articulation"),
        ("energy_dynamics", "Energy dynamics"),
        ("temporal_align", "Temporal alignment (DTW)"),
    ]:
        d = dims.get(key) or {}
        lines.append(f"  {label:28s}  {d.get('pct', 0):5.1f}%")
        if d.get("comment"):
            lines.append(f"      → {d['comment']}")
    lines += [
        "",
        "KEY MEASUREMENTS",
        "-" * 48,
        f"  Pitch mean     A: {a.get('pitch_hz') or '—'} Hz    B: {b.get('pitch_hz') or '—'} Hz",
        f"  Centroid       A: {a.get('spectral_centroid', 0):.0f} Hz   B: {b.get('spectral_centroid', 0):.0f} Hz",
        f"  Flatness       A: {a.get('spectral_flatness', 0):.4f}    B: {b.get('spectral_flatness', 0):.4f}",
        f"  Voicing proxy  A: {100 * _safe(a.get('voiced_ratio')):.0f}%      B: {100 * _safe(b.get('voiced_ratio')):.0f}%",
        "",
        "Generated by AETHER Forensics v2 (classical DSP, zero ML).",
    ]
    return "\n".join(lines)


def _report_da(a, b, score, dims, mode, verdict, result, rel) -> str:
    lines = [
        "AETHER FORENSICS-RAPPORT",
        "=" * 48,
        f"Tilstand: {mode.upper()}",
        f"Sample A: {a.get('label', 'A')}  ({a.get('duration', 0):.2f}s)",
        f"Sample B: {b.get('label', 'B')}  ({b.get('duration', 0):.2f}s)",
        f"Samlet match-score: {score:.1f}%",
        f"Pålidelighed: {rel.get('reliability_pct', '—')}% — {rel.get('label', '')}",
        f"Konklusion: {verdict}",
        "",
        "OM ALMINDELIG TALE",
        "-" * 48,
        "VOICE-mode virker på almindelig tale. I behøver IKKE sige de samme ord.",
        "Forskellige sætninger sammenligner stadig stemmeklang (timbre, pitch-område).",
        "DTW/tids-score falder ofte når teksten er forskellig — det er forventeligt.",
        "Bedst: ≥3 sek tør tale pr. klip, lignende telefon/mic, lidt baggrundsstøj.",
        "",
        "ANSVARSFRASKRIVELSE",
        "-" * 48,
        "Kun klassisk DSP. IKKE retsmedicinsk stemme-ID. Vejledning — ikke bevis.",
        "",
        "PÅLIDELIGHED",
        "-" * 48,
    ]
    for r in rel.get("reasons") or []:
        lines.append(f"  • {r}")
    lines += ["", "DIMENSIONER", "-" * 48]
    labels_da = {
        "timbre_mfcc": "Klang (MFCC / mel)",
        "pitch": "Pitch / intonation",
        "spectral": "Spektral form",
        "envelope": "Envelope",
        "energy_dynamics": "Energi-dynamik",
        "temporal_align": "Tids-alignment (DTW)",
    }
    for key, label in labels_da.items():
        d = dims.get(key) or {}
        lines.append(f"  {label:28s}  {d.get('pct', 0):5.1f}%")
        if d.get("comment"):
            lines.append(f"      → {d['comment']}")
    lines.append("")
    lines.append("Genereret af AETHER Forensics v2 (klassisk DSP, ingen ML).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _light_profile(p: dict) -> dict:
    skip = {"mfcc_matrix"}
    return {k: v for k, v in p.items() if k not in skip}


def compare_profiles(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    *,
    mode: str = "sound",
    weights: Optional[dict[str, float]] = None,
    language: str = "en",
) -> dict[str, Any]:
    mode = mode if mode in ("sound", "voice") else "sound"
    w = dict(VOICE_WEIGHTS if mode == "voice" else DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}

    raw = {
        "timbre_mfcc": _timbre_score(profile_a, profile_b),
        "pitch": _pitch_score(profile_a, profile_b),
        "spectral": _spectral_score(profile_a, profile_b),
        "envelope": _envelope_score(profile_a, profile_b),
        "energy_dynamics": _energy_score(profile_a, profile_b),
        "temporal_align": _temporal_score(profile_a, profile_b),
    }

    dims: dict[str, Any] = {}
    overall = 0.0
    for k, sim in raw.items():
        pct = round(100.0 * sim, 1)
        dims[k] = {
            "score": round(sim, 4),
            "pct": pct,
            "weight": round(w.get(k, 0), 3),
            "comment": _dimension_comment(k, pct, profile_a, profile_b),
        }
        overall += w.get(k, 0) * sim

    score_pct = round(100.0 * float(np.clip(overall, 0.0, 1.0)), 1)
    reliability = estimate_reliability(profile_a, profile_b)

    result = {
        "score_pct": score_pct,
        "score_01": round(float(overall), 4),
        "verdict": _verdict(score_pct, mode),
        "mode": mode,
        "weights": w,
        "dimensions": dims,
        "reliability": reliability,
        "profile_a": _light_profile(profile_a),
        "profile_b": _light_profile(profile_b),
    }
    result["report"] = write_analysis_report(result, language=language)
    return result


def compare_audio_arrays(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int = 44100,
    *,
    label_a: str = "A",
    label_b: str = "B",
    mode: str = "sound",
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
    start_a: float = 0.0,
    end_a: Optional[float] = None,
    start_b: float = 0.0,
    end_b: Optional[float] = None,
    rms_normalize: bool = True,
    highpass_speech: bool = True,
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    hp = 80.0 if (mode == "voice" and highpass_speech) else 0.0

    ya, meta_a = preprocess_audio(
        y_a, sr, start_sec=start_a, end_sec=end_a,
        rms_normalize=rms_normalize, highpass_hz=hp,
    )
    yb, meta_b = preprocess_audio(
        y_b, sr, start_sec=start_b, end_sec=end_b,
        rms_normalize=rms_normalize, highpass_hz=hp,
    )
    pa = extract_forensic_profile(
        ya, sr, settings=settings, label=label_a, mode=mode,
        already_preprocessed=True, preprocess_meta=meta_a,
    )
    pb = extract_forensic_profile(
        yb, sr, settings=settings, label=label_b, mode=mode,
        already_preprocessed=True, preprocess_meta=meta_b,
    )
    return compare_profiles(pa, pb, mode=mode, language=language)


def compare_files(
    path_a: str,
    path_b: str,
    *,
    mode: str = "sound",
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
    start_a: float = 0.0,
    end_a: Optional[float] = None,
    start_b: float = 0.0,
    end_b: Optional[float] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    ya, sr_a, meta_a = load_audio(path_a, settings=settings)
    yb, sr_b, meta_b = load_audio(path_b, settings=settings)
    # Resample B to A's rate if needed
    if sr_a != sr_b:
        yb = librosa.resample(yb, orig_sr=sr_b, target_sr=sr_a)
    return compare_audio_arrays(
        ya, yb, sr_a,
        label_a=meta_a.get("filename") or path_a,
        label_b=meta_b.get("filename") or path_b,
        mode=mode, settings=settings, language=language,
        start_a=start_a, end_a=end_a, start_b=start_b, end_b=end_b,
        **kwargs,
    )


def compare_reference_vs_many(
    y_ref: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    sr: int = 44100,
    *,
    mode: str = "voice",
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
    label_ref: str = "Reference",
    start_ref: float = 0.0,
    end_ref: Optional[float] = None,
) -> dict[str, Any]:
    """
    Rank many candidates against one reference.
    candidates: list of (label, waveform)
    """
    settings = settings or AnalysisSettings()
    rankings = []
    for label, y_c in candidates:
        try:
            r = compare_audio_arrays(
                y_ref, y_c, sr,
                label_a=label_ref, label_b=label,
                mode=mode, settings=settings, language=language,
                start_a=start_ref, end_a=end_ref,
            )
            rankings.append(
                {
                    "label": label,
                    "score_pct": r["score_pct"],
                    "reliability_pct": r["reliability"]["reliability_pct"],
                    "verdict": r["verdict"],
                    "dimensions": {k: v["pct"] for k, v in r["dimensions"].items()},
                    "result": r,
                }
            )
        except Exception as exc:
            rankings.append(
                {
                    "label": label,
                    "score_pct": 0.0,
                    "reliability_pct": 0.0,
                    "verdict": f"Error: {exc}",
                    "dimensions": {},
                    "result": None,
                }
            )
    rankings.sort(key=lambda x: x["score_pct"], reverse=True)
    return {
        "reference": label_ref,
        "mode": mode,
        "rankings": rankings,
        "best": rankings[0] if rankings else None,
    }
