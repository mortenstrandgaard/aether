"""
Pitch-lane forensics — horizontal note grid + classical autotune / pitch-correction heuristics.

Pure DSP. Does NOT prove plugin brand; reports acoustic evidence that pitch
is quantized / corrected vs continuous natural intonation.
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np

from aether.notes import hz_to_midi, midi_to_hz, hz_to_note


def extract_pitch_lane(
    y: np.ndarray,
    sr: int,
    *,
    fmin: float = 70.0,
    fmax: float = 500.0,
    hop_length: int = 256,
) -> dict[str, Any]:
    """Continuous pitch track for visualization + heuristics."""
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size < int(0.1 * sr):
        return {
            "times": [],
            "f0": [],
            "voiced_mask": [],
            "duration": 0.0,
            "f0_mean": None,
        }

    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    voiced = f0 > 0
    voiced_f0 = f0[voiced]

    return {
        "times": times.astype(float).tolist(),
        "f0": f0.astype(float).tolist(),
        "voiced_mask": voiced.astype(bool).tolist(),
        "duration": float(len(y) / sr),
        "f0_mean": float(np.median(voiced_f0)) if voiced_f0.size else None,
        "hop_length": hop_length,
        "sample_rate": sr,
    }


def note_grid_hz(
    f0_min: float,
    f0_max: float,
    pad_semitones: int = 2,
) -> list[dict[str, Any]]:
    """Equal-temperament horizontal guide lines covering the pitch range."""
    if f0_min <= 0 or f0_max <= 0 or f0_max < f0_min:
        f0_min, f0_max = 80.0, 400.0
    midi_lo = int(np.floor(hz_to_midi(f0_min) or 40)) - pad_semitones
    midi_hi = int(np.ceil(hz_to_midi(f0_max) or 72)) + pad_semitones
    midi_lo = max(24, midi_lo)
    midi_hi = min(96, midi_hi)
    lines = []
    for m in range(midi_lo, midi_hi + 1):
        hz = midi_to_hz(m)
        name = hz_to_note(hz) or str(m)
        lines.append({"midi": m, "hz": round(hz, 2), "note": name})
    return lines


def _cents_to_nearest_note(hz: float) -> float:
    midi = hz_to_midi(hz)
    if midi is None:
        return 50.0
    nearest = round(midi)
    return float(100.0 * (midi - nearest))  # -50..+50


def analyze_pitch_correction(
    lane: dict[str, Any],
    *,
    quantize_cents: float = 12.0,
) -> dict[str, Any]:
    """
    Classical heuristics for pitch correction / autotune-like behaviour.

    Signals (not proof of a specific product):
      - high fraction of frames near exact equal-tempered pitch centres
      - long plateaus of nearly constant f0 (held notes)
      - large step-like pitch jumps between plateaus
      - low micro-variation (jitter) on plateaus vs natural vibrato
    """
    f0 = np.asarray(lane.get("f0") or [], dtype=float)
    times = np.asarray(lane.get("times") or [], dtype=float)
    if f0.size < 8:
        return {
            "correction_likelihood_pct": 0.0,
            "label": "Insufficient pitch data",
            "evidence": [],
            "metrics": {},
        }

    voiced = f0 > 0
    vf0 = f0[voiced]
    if vf0.size < 8:
        return {
            "correction_likelihood_pct": 0.0,
            "label": "Too little voiced pitch",
            "evidence": ["Few voiced frames — cannot assess correction."],
            "metrics": {},
        }

    cents = np.array([_cents_to_nearest_note(h) for h in vf0])
    abs_cents = np.abs(cents)
    near_grid = float(np.mean(abs_cents <= quantize_cents))
    median_abs_cents = float(np.median(abs_cents))

    # Plateaus: successive frames with very small Hz change
    dv = np.abs(np.diff(vf0))
    # relative step in cents
    step_cents = np.abs(1200.0 * np.log2((vf0[1:] + 1e-9) / (vf0[:-1] + 1e-9)))
    small_step = step_cents < 8.0  # almost flat
    big_jump = step_cents > 70.0  # > ~0.7 semitone in one hop

    plateau_frac = float(np.mean(small_step)) if small_step.size else 0.0
    jump_frac = float(np.mean(big_jump)) if big_jump.size else 0.0

    # Vibrato / micro-variation: std of cents deviation on "held" regions
    # Use rolling windows of ~80ms
    hop_t = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    win = max(3, int(0.08 / max(hop_t, 1e-4)))
    micro_stds = []
    idx = np.where(voiced)[0]
    for i in range(0, len(idx) - win, win):
        seg = f0[idx[i : i + win]]
        seg = seg[seg > 0]
        if len(seg) >= win // 2:
            # std in cents around mean
            m = np.mean(seg)
            micro_stds.append(float(np.std(1200.0 * np.log2(seg / (m + 1e-9)))))
    micro_med = float(np.median(micro_stds)) if micro_stds else 0.0

    # Score 0–100 likelihood of *strong* pitch quantisation / correction
    score = 0.0
    evidence: list[str] = []

    # Near grid
    if near_grid >= 0.72:
        score += 35
        evidence.append(
            f"{100 * near_grid:.0f}% of voiced frames sit within ±{quantize_cents:.0f} cents of equal-tempered notes (strong snap-to-grid)."
        )
    elif near_grid >= 0.55:
        score += 20
        evidence.append(
            f"{100 * near_grid:.0f}% of frames near note centres — moderate quantisation hint."
        )
    else:
        evidence.append(
            f"Only {100 * near_grid:.0f}% of frames near exact notes — more continuous intonation."
        )

    if median_abs_cents <= 8:
        score += 15
        evidence.append(f"Median detune from nearest note is only {median_abs_cents:.1f} cents (very tight).")
    elif median_abs_cents >= 25:
        evidence.append(f"Median detune {median_abs_cents:.1f} cents — freer intonation.")

    if plateau_frac >= 0.55 and micro_med < 12:
        score += 25
        evidence.append(
            f"Long flat pitch plateaus ({100 * plateau_frac:.0f}% tiny steps) with low micro-variation "
            f"({micro_med:.1f} cents) — held, corrected notes rather than free vibrato."
        )
    elif micro_med >= 20:
        evidence.append(
            f"Higher micro-variation ({micro_med:.1f} cents) — more natural vibrato/instability."
        )
    else:
        score += 8

    if jump_frac >= 0.04 and plateau_frac >= 0.4:
        score += 20
        evidence.append(
            f"Noticeable step-like pitch jumps ({100 * jump_frac:.1f}% of transitions) between plateaus — "
            "classic hard-tune transition pattern."
        )
    elif jump_frac < 0.015:
        evidence.append("Few large stepwise pitch jumps — glides/portamento more common.")

    score = float(np.clip(score, 0, 100))
    if score >= 70:
        label = "Strong acoustic signs of pitch quantisation / correction"
    elif score >= 45:
        label = "Moderate signs — possible correction or very precise singing"
    elif score >= 25:
        label = "Weak / mixed signs — not conclusive"
    else:
        label = "Little quantisation evidence — continuous pitch behaviour"

    return {
        "correction_likelihood_pct": round(score, 1),
        "label": label,
        "evidence": evidence,
        "metrics": {
            "near_grid_fraction": round(near_grid, 3),
            "median_abs_cents": round(median_abs_cents, 2),
            "plateau_fraction": round(plateau_frac, 3),
            "jump_fraction": round(jump_frac, 4),
            "micro_variation_cents": round(micro_med, 2),
            "quantize_threshold_cents": quantize_cents,
            "n_voiced_frames": int(vf0.size),
        },
        "disclaimer": (
            "Heuristic acoustic analysis only. Cannot name a plugin (Auto-Tune, Melodyne, …). "
            "Very accurate singers and heavy correction can look similar. Not legal proof."
        ),
    }


def build_pitch_forensics(
    y: np.ndarray,
    sr: int,
    *,
    label: str = "clip",
    fmin: float = 70.0,
    fmax: float = 500.0,
) -> dict[str, Any]:
    lane = extract_pitch_lane(y, sr, fmin=fmin, fmax=fmax)
    f0 = np.asarray(lane["f0"], dtype=float)
    voiced = f0[f0 > 0]
    if voiced.size:
        grid = note_grid_hz(float(np.min(voiced)), float(np.max(voiced)))
    else:
        grid = note_grid_hz(100.0, 400.0)
    correction = analyze_pitch_correction(lane)
    return {
        "label": label,
        "lane": lane,
        "note_grid": grid,
        "correction": correction,
    }
