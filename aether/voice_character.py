"""
AETHER Voice Characteristics — classical DSP description of voice + prosody.

Measures acoustic *characteristics* of speech/voice material only.
Does NOT infer personality, psychology, identity truth, or clinical diagnoses.

Two layers:
  1. Voice quality  — how the instrument sounds (pitch centre, colour, voicing)
  2. Prosody        — how the voice is used over time (melody, rate, pauses, dynamics)

All labels are descriptive acoustic categories with explicit disclaimers.
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from aether.config import AnalysisSettings
from aether.forensics import preprocess_audio
from aether.loader import load_audio


def _lpc(seq: np.ndarray, order: int) -> np.ndarray:
    """LPC coefficients — prefer librosa (stable across SciPy versions)."""
    try:
        return librosa.lpc(seq.astype(np.float64), order=order)
    except Exception:
        # Fallback: pure numpy autocorrelation LPC (Levinson-ish via solve)
        x = seq.astype(np.float64)
        x = x - np.mean(x)
        r = np.correlate(x, x, mode="full")
        r = r[len(x) - 1 : len(x) + order]
        R = np.array([r[abs(i - j)] for i in range(order) for j in range(order)]).reshape(order, order)
        try:
            a = np.linalg.solve(R, -r[1 : order + 1])
            return np.concatenate([[1.0], a])
        except np.linalg.LinAlgError:
            return np.array([1.0] + [0.0] * order)


# ---------------------------------------------------------------------------
# Low-level measurements
# ---------------------------------------------------------------------------

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


def _yin_contour(
    y: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 450.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, f0_hz) with 0 for unvoiced."""
    if len(y) < int(0.05 * sr):
        return np.array([0.0]), np.array([0.0])
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)
    # Approximate frame times for yin (hop = frame_length/4 default in librosa)
    hop = 512
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    if len(times) != len(f0):
        times = np.linspace(0, len(y) / sr, len(f0))
    return times.astype(float), f0.astype(float)


def _formant_proxies(y: np.ndarray, sr: int, order: Optional[int] = None) -> dict[str, Any]:
    """
    LPC-based formant frequency proxies (F1–F3-ish).
    Classical speech analysis approach — approximate, not lab-grade.
    """
    if len(y) < sr // 10:
        return {"f1_hz": None, "f2_hz": None, "f3_hz": None, "method": "lpc", "n_found": 0}

    # Pre-emphasis + window
    y = librosa.effects.preemphasis(y)
    # Use a stable mid chunk
    mid = len(y) // 2
    win = min(len(y), int(0.05 * sr))
    seg = y[max(0, mid - win // 2) : mid + win // 2]
    if len(seg) < 64:
        seg = y

    if order is None:
        order = int(2 + sr / 1000)  # common rule of thumb
        order = max(10, min(order, 24))

    try:
        a = _lpc(seg.astype(float), order)
        # Roots of LPC polynomial
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angs = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted(angs * (sr / (2 * np.pi)))
        # Keep speech-relevant band
        formants = [f for f in freqs if 90 < f < 4000]
        # Bandwidth-ish filter: prefer poles closer to unit circle
        f1 = formants[0] if len(formants) > 0 else None
        f2 = formants[1] if len(formants) > 1 else None
        f3 = formants[2] if len(formants) > 2 else None
        return {
            "f1_hz": round(f1, 1) if f1 else None,
            "f2_hz": round(f2, 1) if f2 else None,
            "f3_hz": round(f3, 1) if f3 else None,
            "method": "lpc",
            "n_found": len(formants),
            "order": order,
        }
    except Exception:
        return {"f1_hz": None, "f2_hz": None, "f3_hz": None, "method": "lpc", "n_found": 0}


def _hnr_proxy(y: np.ndarray, sr: int, f0_mean: float) -> Optional[float]:
    """
    Simple harmonic-to-noise style proxy from autocorrelation at lag ~1/f0.
    Higher → more periodic / harmonic; lower → noisier / breathier.
    Returns value roughly in a useful 0–30 dB-ish mapped scale, or None.
    """
    if not f0_mean or f0_mean < 60:
        return None
    lag = int(round(sr / f0_mean))
    if lag < 2 or lag >= len(y) // 2:
        return None
    y0 = y - np.mean(y)
    # Normalized autocorrelation at pitch lag
    c0 = float(np.dot(y0, y0) + 1e-12)
    c_lag = float(np.dot(y0[:-lag], y0[lag:]))
    r = c_lag / c0
    r = float(np.clip(r, 1e-6, 0.999))
    # Map correlation to dB-like HNR proxy
    hnr = 10.0 * np.log10(r / (1.0 - r))
    return round(float(hnr), 2)


def _syllable_rate(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Energy-envelope peak rate as a classical speech-rate proxy (not true phoneme rate).
    """
    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size < 4 or float(np.max(rms)) <= 0:
        return {"peaks_per_sec": 0.0, "n_peaks": 0}
    rms_n = rms / np.max(rms)
    # Peaks above 0.25 with spacing ~50ms
    min_dist = max(1, int(0.05 * sr / hop))
    peaks, _ = find_peaks(rms_n, height=0.25, distance=min_dist)
    dur = len(y) / sr
    rate = float(len(peaks) / max(dur, 1e-6))
    return {"peaks_per_sec": round(rate, 2), "n_peaks": int(len(peaks))}


def _pause_stats(y: np.ndarray, sr: int, top_db: float = 30.0) -> dict[str, float]:
    """Fraction of time below energy threshold (pause-like)."""
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size == 0 or float(np.max(rms)) <= 0:
        return {"pause_fraction": 0.0, "speech_fraction": 1.0}
    # Threshold relative to peak
    thr = float(np.max(rms)) * (10 ** (-top_db / 20.0))
    silent = rms < thr
    pause_frac = float(np.mean(silent))
    return {
        "pause_fraction": round(pause_frac, 3),
        "speech_fraction": round(1.0 - pause_frac, 3),
    }


# ---------------------------------------------------------------------------
# Descriptive labels (acoustic only — never personality)
# ---------------------------------------------------------------------------

def _pitch_register_label(f0: Optional[float]) -> str:
    if f0 is None or f0 <= 0:
        return "unpitched / unstable"
    if f0 < 100:
        return "low pitch centre"
    if f0 < 150:
        return "low–mid pitch centre"
    if f0 < 200:
        return "mid pitch centre"
    if f0 < 260:
        return "mid–high pitch centre"
    return "high pitch centre"


def _pitch_variability_label(std_hz: float, range_st: float) -> str:
    if range_st < 3 and std_hz < 12:
        return "narrow pitch range (relatively steady melody)"
    if range_st < 8 and std_hz < 25:
        return "moderate pitch range"
    if range_st < 14:
        return "wide pitch range (more melodic variation)"
    return "very wide pitch excursions"


def _rate_label(peaks_per_sec: float) -> str:
    if peaks_per_sec < 1.5:
        return "slow energy-peak rate (slow or sparse articulation proxy)"
    if peaks_per_sec < 3.0:
        return "moderate energy-peak rate"
    if peaks_per_sec < 4.5:
        return "fast energy-peak rate"
    return "very fast energy-peak rate"


def _pause_label(pause_frac: float) -> str:
    if pause_frac < 0.12:
        return "few pauses / dense delivery"
    if pause_frac < 0.28:
        return "balanced pause structure"
    if pause_frac < 0.45:
        return "noticeable pausing"
    return "high pause fraction (sparse or hesitant delivery — acoustic only)"


def _colour_label(centroid: float, flatness: float, hnr: Optional[float]) -> str:
    parts = []
    if centroid < 800:
        parts.append("darker spectral balance")
    elif centroid < 1800:
        parts.append("balanced spectral balance")
    else:
        parts.append("brighter spectral balance")
    if flatness > 0.25:
        parts.append("noisier / less tonal colour")
    elif flatness < 0.08:
        parts.append("more tonal / harmonic colour")
    if hnr is not None:
        if hnr > 12:
            parts.append("strong periodicity (clearer harmonic structure)")
        elif hnr < 3:
            parts.append("weaker periodicity (breathier / noisier tendency)")
    return "; ".join(parts) if parts else "colour indeterminate"


def _voicing_label(voiced_ratio: float) -> str:
    if voiced_ratio < 0.2:
        return "mostly unvoiced / noisy material"
    if voiced_ratio < 0.45:
        return "mixed voicing"
    if voiced_ratio < 0.7:
        return "predominantly voiced"
    return "strongly voiced throughout"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_voice_character(
    y: np.ndarray,
    sr: int,
    *,
    label: str = "sample",
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> dict[str, Any]:
    """
    Full voice quality + prosody characteristic report for one clip.
    """
    settings = settings or AnalysisSettings()
    y = np.asarray(y, dtype=np.float32).ravel()
    y, prep = preprocess_audio(
        y,
        sr,
        start_sec=start_sec,
        end_sec=end_sec,
        peak_normalize=True,
        rms_normalize=True,
        trim_silence=True,
        highpass_hz=80.0,
        max_seconds=60.0,
    )
    duration = len(y) / sr

    times, f0 = _yin_contour(y, sr)
    voiced = f0[f0 > 0]
    if voiced.size:
        f0_mean = float(np.median(voiced))
        f0_std = float(np.std(voiced))
        f0_min = float(np.min(voiced))
        f0_max = float(np.max(voiced))
        # range in semitones
        range_st = float(12.0 * np.log2((f0_max + 1e-9) / (f0_min + 1e-9)))
        voiced_ratio = float(voiced.size / max(len(f0), 1))
    else:
        f0_mean = f0_std = f0_min = f0_max = range_st = 0.0
        voiced_ratio = 0.0

    # Plot-friendly contour (cap points)
    step = max(1, len(f0) // 160)
    pitch_times = [round(float(t), 3) for t in times[::step]]
    pitch_vals = [
        round(float(v), 2) if v > 0 else None for v in f0[::step]
    ]

    formants = _formant_proxies(y, sr)
    hnr = _hnr_proxy(y, sr, f0_mean if f0_mean > 0 else 0.0)
    rate = _syllable_rate(y, sr)
    pauses = _pause_stats(y, sr)

    # Spectral colour
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = librosa.feature.rms(y=y)[0]
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-12)) if rms.size else 0.0

    # Labels
    voice_quality = {
        "pitch_register": _pitch_register_label(f0_mean if f0_mean > 0 else None),
        "spectral_colour": _colour_label(cent, flat, hnr),
        "voicing": _voicing_label(voiced_ratio),
    }
    prosody = {
        "pitch_variability": _pitch_variability_label(f0_std, range_st),
        "delivery_rate": _rate_label(rate["peaks_per_sec"]),
        "pausing": _pause_label(pauses["pause_fraction"]),
        "dynamics": (
            "steady intensity"
            if rms_cv < 0.35
            else ("moderate intensity variation" if rms_cv < 0.7 else "high intensity variation")
        ),
    }

    measurements = {
        "duration_sec": round(duration, 3),
        "f0_mean_hz": round(f0_mean, 2) if f0_mean > 0 else None,
        "f0_std_hz": round(f0_std, 2),
        "f0_min_hz": round(f0_min, 2) if f0_mean > 0 else None,
        "f0_max_hz": round(f0_max, 2) if f0_mean > 0 else None,
        "f0_range_semitones": round(range_st, 2),
        "voiced_ratio": round(voiced_ratio, 3),
        "hnr_proxy_db": hnr,
        "spectral_centroid_hz": round(cent, 1),
        "spectral_rolloff_hz": round(rolloff, 1),
        "spectral_flatness": round(flat, 5),
        "zcr": round(zcr, 5),
        "rms_cv": round(rms_cv, 4),
        "energy_peaks_per_sec": rate["peaks_per_sec"],
        "n_energy_peaks": rate["n_peaks"],
        "pause_fraction": pauses["pause_fraction"],
        "speech_fraction": pauses["speech_fraction"],
        "formants": formants,
    }

    reliability = _character_reliability(duration, voiced_ratio, f0_mean)

    # Keep short plot buffer
    y_plot = y[: int(min(len(y), 12 * sr))].copy()

    result = {
        "label": label,
        "sample_rate": sr,
        "preprocess": prep,
        "voice_quality": voice_quality,
        "prosody": prosody,
        "measurements": measurements,
        "reliability": reliability,
        "pitch_times": pitch_times,
        "pitch_contour_plot": pitch_vals,
        "y_plot": y_plot,
        "disclaimer": (
            "Acoustic characteristics only. Not personality, not clinical diagnosis, "
            "not legal speaker identification. Formants and rates are classical proxies."
        ),
    }
    result["report"] = write_character_report(result, language=language)
    return result


def _character_reliability(duration: float, voiced_ratio: float, f0_mean: float) -> dict[str, Any]:
    score = 1.0
    reasons = []
    if duration < 2.0:
        score *= 0.4
        reasons.append("Clip shorter than 2s — characteristics unstable.")
    elif duration < 4.0:
        score *= 0.7
        reasons.append("Short clip — prefer ≥4–8s of continuous speech.")
    else:
        reasons.append(f"Duration OK ({duration:.1f}s).")
    if voiced_ratio < 0.25:
        score *= 0.55
        reasons.append("Low voicing — music/noise/whisper reduces pitch reliability.")
    elif f0_mean <= 0:
        score *= 0.5
        reasons.append("No stable f0 — pitch-based descriptors weak.")
    else:
        reasons.append("Usable voiced content for pitch descriptors.")
    pct = round(100 * float(np.clip(score, 0.05, 1.0)), 1)
    if pct >= 70:
        label = "High enough for a rough acoustic character card"
    elif pct >= 45:
        label = "Medium — interpret cautiously"
    else:
        label = "Low — re-record longer/cleaner speech if possible"
    return {"reliability_pct": pct, "label": label, "reasons": reasons}


def analyze_voice_character_file(
    path: str,
    *,
    label: Optional[str] = None,
    settings: Optional[AnalysisSettings] = None,
    language: str = "en",
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> dict[str, Any]:
    settings = settings or AnalysisSettings()
    y, sr, meta = load_audio(path, settings=settings)
    return analyze_voice_character(
        y,
        sr,
        label=label or meta.get("filename") or path,
        settings=settings,
        language=language,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def compare_voice_characters(
    char_a: dict[str, Any],
    char_b: dict[str, Any],
    language: str = "en",
) -> dict[str, Any]:
    """
    Side-by-side characteristic comparison (descriptive deltas — not identity proof).
    """
    ma = char_a.get("measurements") or {}
    mb = char_b.get("measurements") or {}

    def delta_hz(key: str) -> Optional[float]:
        a, b = ma.get(key), mb.get(key)
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 2)

    def delta_f(key: str) -> Optional[float]:
        a, b = ma.get(key), mb.get(key)
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 4)

    diffs = {
        "f0_mean_hz": delta_hz("f0_mean_hz"),
        "f0_range_semitones": delta_f("f0_range_semitones"),
        "spectral_centroid_hz": delta_hz("spectral_centroid_hz"),
        "energy_peaks_per_sec": delta_f("energy_peaks_per_sec"),
        "pause_fraction": delta_f("pause_fraction"),
        "voiced_ratio": delta_f("voiced_ratio"),
        "hnr_proxy_db": delta_f("hnr_proxy_db"),
    }

    bullets = []
    if diffs["f0_mean_hz"] is not None:
        d = diffs["f0_mean_hz"]
        if abs(d) < 8:
            bullets.append("Pitch centres are close.")
        elif d > 0:
            bullets.append(f"B has higher pitch centre than A (~{d:+.0f} Hz).")
        else:
            bullets.append(f"B has lower pitch centre than A (~{d:+.0f} Hz).")
    if diffs["spectral_centroid_hz"] is not None:
        d = diffs["spectral_centroid_hz"]
        if abs(d) > 250:
            bullets.append(
                "B is spectrally brighter than A."
                if d > 0
                else "B is spectrally darker than A."
            )
    if diffs["energy_peaks_per_sec"] is not None:
        d = diffs["energy_peaks_per_sec"]
        if abs(d) > 0.6:
            bullets.append(
                "B shows a faster energy-peak rate (delivery proxy)."
                if d > 0
                else "B shows a slower energy-peak rate (delivery proxy)."
            )
    if diffs["pause_fraction"] is not None:
        d = diffs["pause_fraction"]
        if abs(d) > 0.08:
            bullets.append(
                "B has more pause-like low-energy time."
                if d > 0
                else "B has less pause-like low-energy time."
            )

    report = _compare_character_report(char_a, char_b, diffs, bullets, language)
    return {
        "label_a": char_a.get("label"),
        "label_b": char_b.get("label"),
        "deltas": diffs,
        "bullets": bullets,
        "report": report,
        "char_a": {k: v for k, v in char_a.items() if k != "y_plot"},
        "char_b": {k: v for k, v in char_b.items() if k != "y_plot"},
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_character_report(result: dict[str, Any], language: str = "en") -> str:
    if language == "da":
        return _report_character_da(result)
    return _report_character_en(result)


def _report_character_en(r: dict[str, Any]) -> str:
    m = r.get("measurements") or {}
    vq = r.get("voice_quality") or {}
    pr = r.get("prosody") or {}
    rel = r.get("reliability") or {}
    f = (m.get("formants") or {})
    lines = [
        "AETHER VOICE CHARACTERISTICS REPORT",
        "=" * 52,
        f"Sample: {r.get('label', '—')}",
        f"Duration: {m.get('duration_sec', 0):.2f}s",
        f"Reliability: {rel.get('reliability_pct', '—')}% — {rel.get('label', '')}",
        "",
        "SCOPE",
        "-" * 52,
        "This report describes ACOUSTIC CHARACTERISTICS of the recording only.",
        "It does NOT describe personality, psychology, intent, or medical status.",
        "It is NOT a legal speaker-identification product.",
        "Formant values and speech-rate proxies are classical estimates.",
        "",
        "1. VOICE QUALITY  (how the voice instrument sounds)",
        "-" * 52,
        f"  Pitch register:     {vq.get('pitch_register', '—')}",
        f"  f0 mean:            {m.get('f0_mean_hz') or '—'} Hz",
        f"  f0 range:           {m.get('f0_min_hz') or '—'} – {m.get('f0_max_hz') or '—'} Hz"
        f"  ({m.get('f0_range_semitones', 0):.1f} st)",
        f"  f0 std:             {m.get('f0_std_hz', 0)} Hz",
        f"  Voicing:            {vq.get('voicing', '—')}  (ratio {m.get('voiced_ratio', 0):.2f})",
        f"  Spectral colour:    {vq.get('spectral_colour', '—')}",
        f"  Centroid:           {m.get('spectral_centroid_hz', 0):.0f} Hz",
        f"  Flatness:           {m.get('spectral_flatness', 0):.4f}",
        f"  HNR proxy:          {m.get('hnr_proxy_db') if m.get('hnr_proxy_db') is not None else '—'} dB",
        f"  Formant proxies:    F1={f.get('f1_hz') or '—'}  F2={f.get('f2_hz') or '—'}  F3={f.get('f3_hz') or '—'} Hz",
        "",
        "2. PROSODY  (how the voice is used over time / stemmeføring)",
        "-" * 52,
        f"  Pitch variability:  {pr.get('pitch_variability', '—')}",
        f"  Delivery rate:      {pr.get('delivery_rate', '—')}",
        f"  Energy peaks/sec:   {m.get('energy_peaks_per_sec', 0)}  (syllable-rate proxy)",
        f"  Pausing:            {pr.get('pausing', '—')}",
        f"  Pause fraction:     {m.get('pause_fraction', 0):.2f}",
        f"  Dynamics:           {pr.get('dynamics', '—')}  (RMS CV {m.get('rms_cv', 0):.2f})",
        "",
        "RELIABILITY NOTES",
        "-" * 52,
    ]
    for reason in rel.get("reasons") or []:
        lines.append(f"  • {reason}")
    lines += [
        "",
        "Generated by AETHER Voice Characteristics (classical DSP, zero ML).",
        r.get("disclaimer", ""),
    ]
    return "\n".join(lines)


def _report_character_da(r: dict[str, Any]) -> str:
    m = r.get("measurements") or {}
    vq = r.get("voice_quality") or {}
    pr = r.get("prosody") or {}
    rel = r.get("reliability") or {}
    f = (m.get("formants") or {})
    lines = [
        "AETHER STEMME-KARAKTERISTIK RAPPORT",
        "=" * 52,
        f"Sample: {r.get('label', '—')}",
        f"Varighed: {m.get('duration_sec', 0):.2f}s",
        f"Pålidelighed: {rel.get('reliability_pct', '—')}% — {rel.get('label', '')}",
        "",
        "AFGRÆNSNING",
        "-" * 52,
        "Denne rapport beskriver KUN AKUSTISKE KARAKTERISTIKA i optagelsen.",
        "Den laver IKKE psykologiske profiler, intention-analyse eller helbredsvurdering.",
        "Den er IKKE retsmedicinsk person-identifikation.",
        "Formanter og talehastighed er klassiske proxies (estimater).",
        "",
        "1. STEMMEKVALITET  (hvordan stemme-instrumentet lyder)",
        "-" * 52,
        f"  Pitch-register:     {vq.get('pitch_register', '—')}",
        f"  f0 mean:            {m.get('f0_mean_hz') or '—'} Hz",
        f"  f0-område:          {m.get('f0_min_hz') or '—'} – {m.get('f0_max_hz') or '—'} Hz"
        f"  ({m.get('f0_range_semitones', 0):.1f} st)",
        f"  f0 std:             {m.get('f0_std_hz', 0)} Hz",
        f"  Voicing:            {vq.get('voicing', '—')}  (ratio {m.get('voiced_ratio', 0):.2f})",
        f"  Spektral farve:     {vq.get('spectral_colour', '—')}",
        f"  Centroid:           {m.get('spectral_centroid_hz', 0):.0f} Hz",
        f"  Flatness:           {m.get('spectral_flatness', 0):.4f}",
        f"  HNR-proxy:          {m.get('hnr_proxy_db') if m.get('hnr_proxy_db') is not None else '—'} dB",
        f"  Formant-proxies:    F1={f.get('f1_hz') or '—'}  F2={f.get('f2_hz') or '—'}  F3={f.get('f3_hz') or '—'} Hz",
        "",
        "2. PROSODI / STEMMEFØRING  (hvordan stemmen bruges over tid)",
        "-" * 52,
        f"  Pitch-variabilitet: {pr.get('pitch_variability', '—')}",
        f"  Leverings-rate:     {pr.get('delivery_rate', '—')}",
        f"  Energi-peaks/sek:   {m.get('energy_peaks_per_sec', 0)}  (stavelses-rate proxy)",
        f"  Pauser:             {pr.get('pausing', '—')}",
        f"  Pause-andel:        {m.get('pause_fraction', 0):.2f}",
        f"  Dynamik:            {pr.get('dynamics', '—')}  (RMS CV {m.get('rms_cv', 0):.2f})",
        "",
        "PÅLIDELIGHED",
        "-" * 52,
    ]
    for reason in rel.get("reasons") or []:
        lines.append(f"  • {reason}")
    lines += [
        "",
        "Genereret af AETHER Stemme-karakteristik (klassisk DSP, ingen ML).",
        "Kun akustik — ingen psykologiske konklusioner.",
    ]
    return "\n".join(lines)


def _compare_character_report(a, b, diffs, bullets, language: str) -> str:
    if language == "da":
        lines = [
            "AETHER KARAKTERISTIK-SAMMENLIGNING",
            "=" * 52,
            f"A: {a.get('label')}    B: {b.get('label')}",
            "",
            "Beskriver akustiske forskelle — ikke personlighed, ikke juridisk bevis.",
            "",
            "DELTA (B − A)",
            "-" * 52,
        ]
        for k, v in diffs.items():
            lines.append(f"  {k:24s}  {v if v is not None else '—'}")
        lines += ["", "OPSUMMERING", "-" * 52]
        for b_line in bullets or ["Ingen store forskelle markeret af reglerne."]:
            lines.append(f"  • {b_line}")
        return "\n".join(lines)

    lines = [
        "AETHER CHARACTERISTICS COMPARISON",
        "=" * 52,
        f"A: {a.get('label')}    B: {b.get('label')}",
        "",
        "Acoustic differences only — not personality, not legal proof.",
        "",
        "DELTA (B − A)",
        "-" * 52,
    ]
    for k, v in diffs.items():
        lines.append(f"  {k:24s}  {v if v is not None else '—'}")
    lines += ["", "SUMMARY", "-" * 52]
    for b_line in bullets or ["No large rule-flagged differences."]:
        lines.append(f"  • {b_line}")
    return "\n".join(lines)
