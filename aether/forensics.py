"""
AETHER Forensics — classical DSP comparison of two sounds / voice clips.

Produces:
  - Overall match score 0–100%
  - Per-dimension scores (timbre, pitch, spectral, envelope, rhythm/energy)
  - A written, rule-based analysis report (no AI / no LLM)

IMPORTANT LIMITATIONS (always disclosed in the report):
  - This is NOT court-grade speaker identification.
  - Not biometric authentication.
  - Classical features only (MFCC, pitch stats, spectral shape, envelope, etc.).
  - Same mic/room/codec can inflate similarity; different conditions can deflate it.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# Default weights for overall score (sum ≈ 1.0)
DEFAULT_WEIGHTS = {
    "timbre_mfcc": 0.32,
    "pitch": 0.22,
    "spectral": 0.20,
    "envelope": 0.12,
    "energy_dynamics": 0.08,
    "temporal_align": 0.06,  # DTW on MFCC
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _sim_from_cosine(c: float) -> float:
    """Map cosine [-1,1] → [0,1]."""
    return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))


def _sim_from_rel_diff(a: float, b: float, scale: float) -> float:
    """1 when equal; decays with relative absolute difference / scale."""
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


def extract_forensic_profile(
    y: np.ndarray,
    sr: int,
    settings: Optional[AnalysisSettings] = None,
    label: str = "sample",
    max_seconds: float = 30.0,
) -> dict[str, Any]:
    """
    Build a forensic feature profile from a mono waveform.
    Caps length for speed; peak-normalizes.
    """
    settings = settings or AnalysisSettings()
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size == 0:
        raise ValueError(f"Empty audio for '{label}'")

    # Trim silence lightly
    try:
        yt, _ = librosa.effects.trim(y, top_db=35.0)
        if yt.size > int(0.1 * sr):
            y = yt
    except Exception:
        pass

    if len(y) > int(max_seconds * sr):
        y = y[: int(max_seconds * sr)]

    peak = float(np.max(np.abs(y))) + 1e-12
    y = y / peak

    spectral = extract_spectral(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    mfcc = extract_mfcc(
        y, sr, n_mfcc=settings.n_mfcc, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    envelope = extract_envelope(y, sr)
    effects = estimate_effects(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    flux = extract_spectral_flux(y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length)
    pitch_hz, contour, note = extract_pitch(y, sr, fmin=50.0, fmax=500.0)

    # Pitch statistics from contour / yin
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

    # Full MFCC matrix for DTW
    n_fft_use = min(settings.n_fft, max(256, 2 ** int(np.floor(np.log2(max(len(y), 4))))))
    hop = min(settings.hop_length, max(n_fft_use // 4, 64))
    mfcc_mat = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=settings.n_mfcc, n_fft=n_fft_use, hop_length=hop
    ).T

    # Energy dynamics
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_cv = float(rms_std / (rms_mean + 1e-12))

    # Zero-crossing rate (voice / noise cue)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop)))

    # Rough spectral envelope (log-mel mean) for “voice colour”
    try:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft_use, hop_length=hop, n_mels=40
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mel_mean = [float(x) for x in np.mean(log_mel, axis=1)]
    except Exception:
        mel_mean = [0.0] * 40

    # Voicing proxy: harmonic-ish if low flatness + has pitch
    flat = _safe(spectral.get("spectral_flatness"))
    voiced_ratio = 0.0
    if pitch_vals:
        voiced_ratio = float(np.clip(len(pitch_vals) / max(len(contour), 1), 0.0, 1.0))
    if flat < 0.1 and pitch_mean > 0:
        voiced_ratio = max(voiced_ratio, 0.55)

    duration = float(len(y) / sr)

    return {
        "label": label,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "pitch_hz": round(pitch_mean, 2) if pitch_mean > 0 else None,
        "pitch_std": round(pitch_std, 2),
        "pitch_min": round(pitch_min, 2),
        "pitch_max": round(pitch_max, 2),
        "pitch_range": round(pitch_range, 2),
        "note": note,
        "zcr": round(zcr, 5),
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "rms_cv": round(rms_cv, 4),
        "spectral_flux": flux,
        "voiced_ratio": round(voiced_ratio, 3),
        "mel_mean": mel_mean,
        "mfcc_matrix": mfcc_mat.astype(np.float32),
        **spectral,
        **mfcc,
        **envelope,
        **effects,
    }


def extract_forensic_profile_from_path(
    path: str,
    settings: Optional[AnalysisSettings] = None,
    label: Optional[str] = None,
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    y, sr, meta = load_audio(path, settings=settings)
    return extract_forensic_profile(
        y, sr, settings=settings, label=label or meta.get("filename") or path
    )


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
        # Both unpitched → neutral-high if both lack pitch; else low
        if (not pa or pa <= 0) and (not pb or pb <= 0):
            return 0.65  # both noise/unpitched-ish
        return 0.25

    # Compare in cents (musical distance)
    cents = 1200.0 * abs(np.log2(pa / pb))
    mean_sim = float(np.clip(np.exp(-cents / 350.0), 0.0, 1.0))  # ~350 cents half-life

    std_sim = _sim_from_rel_diff(
        _safe(a.get("pitch_std")), _safe(b.get("pitch_std")), scale=40.0
    )
    range_sim = _sim_from_rel_diff(
        _safe(a.get("pitch_range")), _safe(b.get("pitch_range")), scale=80.0
    )
    return float(0.55 * mean_sim + 0.25 * std_sim + 0.20 * range_sim)


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
                    _safe(a.get("duration"), 1.0),
                    _safe(b.get("duration"), 1.0),
                    2.0,
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
    ma = a.get("mfcc_matrix")
    mb = b.get("mfcc_matrix")
    if ma is None or mb is None:
        return 0.5
    dist = dtw_mfcc_distance(np.asarray(ma), np.asarray(mb))
    return float(1.0 / (1.0 + dist))


def _verdict(score_pct: float, mode: str) -> str:
    """Human label — conservative language for voice mode."""
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
    # general sound
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
            return f"Pitch centres close ({pa:.0f} Hz vs {pb:.0f} Hz) with similar variation."
        if pct >= 55:
            return f"Pitch related but shifted ({pa:.0f} Hz vs {pb:.0f} Hz) or different intonation."
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
            return "Moderate temporal alignment; similar material, different pacing."
        return "Time evolution of timbre does not align well under DTW."
    return ""


def write_analysis_report(
    result: dict[str, Any],
    language: str = "en",
) -> str:
    """
    Rule-based written forensic report (English or Danish).
    No generative AI — pure templates + measured numbers.
    """
    a = result["profile_a"]
    b = result["profile_b"]
    score = result["score_pct"]
    dims = result["dimensions"]
    mode = result.get("mode", "sound")
    verdict = result["verdict"]

    if language == "da":
        return _report_da(a, b, score, dims, mode, verdict, result)
    return _report_en(a, b, score, dims, mode, verdict, result)


def _report_en(a, b, score, dims, mode, verdict, result) -> str:
    lines = []
    lines.append("AETHER FORENSICS REPORT")
    lines.append("=" * 48)
    lines.append(f"Mode: {mode.upper()}")
    lines.append(f"Sample A: {a.get('label', 'A')}  ({a.get('duration', 0):.2f}s)")
    lines.append(f"Sample B: {b.get('label', 'B')}  ({b.get('duration', 0):.2f}s)")
    lines.append(f"Overall match score: {score:.1f}%")
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("-" * 48)
    lines.append(
        "Classical DSP comparison only (MFCC, pitch, spectral shape, envelope, DTW). "
        "NOT a court-grade biometric speaker ID system. "
        "Same room/mic/codec can raise scores; different phones/rooms can lower them. "
        "Use as investigative / creative guidance, not legal proof."
    )
    lines.append("")
    lines.append("DIMENSION BREAKDOWN")
    lines.append("-" * 48)
    for key, label in [
        ("timbre_mfcc", "Timbre (MFCC / mel)"),
        ("pitch", "Pitch / intonation"),
        ("spectral", "Spectral shape"),
        ("envelope", "Envelope / articulation"),
        ("energy_dynamics", "Energy dynamics"),
        ("temporal_align", "Temporal alignment (DTW)"),
    ]:
        d = dims.get(key) or {}
        pct = d.get("pct", 0)
        lines.append(f"  {label:28s}  {pct:5.1f}%")
        comment = d.get("comment") or ""
        if comment:
            lines.append(f"      → {comment}")
    lines.append("")
    lines.append("KEY MEASUREMENTS")
    lines.append("-" * 48)
    lines.append(
        f"  Pitch mean     A: {a.get('pitch_hz') or '—'} Hz    B: {b.get('pitch_hz') or '—'} Hz"
    )
    lines.append(
        f"  Centroid       A: {a.get('spectral_centroid', 0):.0f} Hz   "
        f"B: {b.get('spectral_centroid', 0):.0f} Hz"
    )
    lines.append(
        f"  Flatness       A: {a.get('spectral_flatness', 0):.4f}    "
        f"B: {b.get('spectral_flatness', 0):.4f}"
    )
    lines.append(
        f"  Attack         A: {a.get('attack_time', 0):.3f}s     "
        f"B: {b.get('attack_time', 0):.3f}s"
    )
    lines.append(
        f"  Voicing proxy  A: {100 * _safe(a.get('voiced_ratio')):.0f}%      "
        f"B: {100 * _safe(b.get('voiced_ratio')):.0f}%"
    )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 48)
    # Top strengths / weaknesses
    ranked = sorted(
        ((k, dims[k]["pct"]) for k in dims),
        key=lambda x: x[1],
        reverse=True,
    )
    strong = [k for k, p in ranked if p >= 70][:3]
    weak = [k for k, p in ranked if p < 55][:3]
    name_map = {
        "timbre_mfcc": "timbre",
        "pitch": "pitch",
        "spectral": "spectral shape",
        "envelope": "envelope",
        "energy_dynamics": "energy dynamics",
        "temporal_align": "temporal alignment",
    }
    if strong:
        lines.append(
            "Strongest agreement: " + ", ".join(name_map.get(k, k) for k in strong) + "."
        )
    if weak:
        lines.append(
            "Largest disagreements: " + ", ".join(name_map.get(k, k) for k in weak) + "."
        )
    if mode == "voice":
        lines.append(
            "Voice mode emphasises pitch centre/variation and MFCC timbre. "
            "For better results use dry speech (little music), similar loudness, "
            "and clips longer than ~2–3 seconds."
        )
    else:
        lines.append(
            "Sound mode balances timbre and spectral shape for drums, synths, FX, etc."
        )
    lines.append("")
    lines.append(f"Weights used: {result.get('weights', {})}")
    lines.append("Generated by AETHER Forensics (classical DSP, zero ML).")
    return "\n".join(lines)


def _report_da(a, b, score, dims, mode, verdict, result) -> str:
    # Map verdict stay in English structure but Danish body
    lines = []
    lines.append("AETHER FORENSICS-RAPPORT")
    lines.append("=" * 48)
    lines.append(f"Tilstand: {mode.upper()}")
    lines.append(f"Sample A: {a.get('label', 'A')}  ({a.get('duration', 0):.2f}s)")
    lines.append(f"Sample B: {b.get('label', 'B')}  ({b.get('duration', 0):.2f}s)")
    lines.append(f"Samlet match-score: {score:.1f}%")
    lines.append(f"Konklusion: {verdict}")
    lines.append("")
    lines.append("ANSVARSFRASKRIVELSE")
    lines.append("-" * 48)
    lines.append(
        "Kun klassisk DSP-sammenligning (MFCC, pitch, spektrum, envelope, DTW). "
        "IKKE retsmedicinsk biometrisk stemme-ID. "
        "Samme rum/mikrofon kan hæve scoren; forskellige telefoner/rum kan sænke den. "
        "Brug som vejledning — ikke som juridisk bevis."
    )
    lines.append("")
    lines.append("DIMENSIONER")
    lines.append("-" * 48)
    labels_da = {
        "timbre_mfcc": "Klang (MFCC / mel)",
        "pitch": "Pitch / intonation",
        "spectral": "Spektral form",
        "envelope": "Envelope / artikulation",
        "energy_dynamics": "Energi-dynamik",
        "temporal_align": "Tids-alignment (DTW)",
    }
    for key, label in labels_da.items():
        d = dims.get(key) or {}
        pct = d.get("pct", 0)
        lines.append(f"  {label:28s}  {pct:5.1f}%")
        if d.get("comment"):
            lines.append(f"      → {d['comment']}")
    lines.append("")
    lines.append("NØGLETAL")
    lines.append("-" * 48)
    lines.append(
        f"  Pitch mean     A: {a.get('pitch_hz') or '—'} Hz    B: {b.get('pitch_hz') or '—'} Hz"
    )
    lines.append(
        f"  Centroid       A: {a.get('spectral_centroid', 0):.0f} Hz   "
        f"B: {b.get('spectral_centroid', 0):.0f} Hz"
    )
    lines.append("")
    lines.append("Fortolkning følger de målte dimensioner ovenfor. "
                 "Genereret af AETHER Forensics (klassisk DSP, ingen ML).")
    return "\n".join(lines)


def compare_profiles(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    *,
    mode: str = "sound",
    weights: Optional[dict[str, float]] = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Compare two forensic profiles.

    mode:
      - "sound"  — balanced for general audio / samples
      - "voice"  — emphasise pitch + MFCC timbre (speech/vox)
    """
    mode = mode if mode in ("sound", "voice") else "sound"
    w = dict(DEFAULT_WEIGHTS)
    if mode == "voice":
        w = {
            "timbre_mfcc": 0.36,
            "pitch": 0.28,
            "spectral": 0.14,
            "envelope": 0.08,
            "energy_dynamics": 0.06,
            "temporal_align": 0.08,
        }
    if weights:
        w.update(weights)

    # Normalize weights
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
    verdict = _verdict(score_pct, mode)

    # Strip heavy matrices from returned profiles for JSON/UI
    def light(p: dict) -> dict:
        return {k: v for k, v in p.items() if k != "mfcc_matrix"}

    result = {
        "score_pct": score_pct,
        "score_01": round(float(overall), 4),
        "verdict": verdict,
        "mode": mode,
        "weights": w,
        "dimensions": dims,
        "profile_a": light(profile_a),
        "profile_b": light(profile_b),
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
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    pa = extract_forensic_profile(y_a, sr, settings=settings, label=label_a)
    pb = extract_forensic_profile(y_b, sr, settings=settings, label=label_b)
    return compare_profiles(pa, pb, mode=mode, language=language)


def compare_files(
    path_a: str,
    path_b: str,
    *,
    mode: str = "sound",
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    pa = extract_forensic_profile_from_path(path_a, settings=settings)
    pb = extract_forensic_profile_from_path(path_b, settings=settings)
    return compare_profiles(pa, pb, mode=mode, language=language)
