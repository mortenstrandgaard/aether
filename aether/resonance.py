"""
Resonance Score between events — pure feature similarity (no AI).

Combines:
  - pitch-class proximity (circle of fifths / chromatic distance)
  - spectral shape (centroid, flatness, MFCC means via cosine)
  - envelope profile similarity (attack, decay, duration)
  - timing relationship (same phase-ish distance / proximity bonus)

Returns 0–100 scores, a full matrix, and ranked pairs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from aether.config import NOTE_NAMES
from aether.sound_dna import pitch_class_from_event


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pitch_class_index(pc: str) -> Optional[int]:
    if not pc or pc == "—":
        return None
    # Normalize flats
    flat_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    pc = flat_map.get(pc, pc)
    if pc not in NOTE_NAMES:
        return None
    return NOTE_NAMES.index(pc)


def pitch_class_score(e1: dict[str, Any], e2: dict[str, Any]) -> float:
    """
    1.0 = same pitch class, decreases with chromatic distance.
    Missing pitch → neutral 0.5.
    """
    i1 = _pitch_class_index(pitch_class_from_event(e1))
    i2 = _pitch_class_index(pitch_class_from_event(e2))
    if i1 is None or i2 is None:
        return 0.5
    dist = min((i1 - i2) % 12, (i2 - i1) % 12)
    # Unison=1.0, minor 2nd≈0.83, tritone≈0.5, … floor at 0
    return float(max(0.0, 1.0 - dist / 12.0 * 1.0))


def spectral_score(e1: dict[str, Any], e2: dict[str, Any]) -> float:
    """Similarity of spectral / MFCC fingerprints → [0, 1]."""
    def vec(e: dict) -> np.ndarray:
        mfcc = e.get("mfcc_mean") or [0.0] * 13
        extras = [
            float(e.get("spectral_centroid") or 0.0) / 8000.0,
            float(e.get("spectral_rolloff") or 0.0) / 12000.0,
            float(e.get("spectral_flatness") or 0.0),
            float(e.get("spectral_bandwidth") or 0.0) / 8000.0,
            float(e.get("harmonic_ratio") or 0.5),
        ]
        return np.concatenate([np.asarray(mfcc, dtype=float), extras])

    cos = _cosine(vec(e1), vec(e2))
    # map cosine [-1,1] → [0,1]
    return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


def envelope_score(e1: dict[str, Any], e2: dict[str, Any]) -> float:
    """Compare attack, decay, duration on a log-ish scale → [0, 1]."""

    def env_vec(e: dict) -> np.ndarray:
        a = max(1e-4, float(e.get("attack_time") or 0.01))
        d = max(1e-4, float(e.get("decay_time") or 0.1))
        dur = max(1e-3, float(e.get("duration") or 0.2))
        return np.array([np.log10(a), np.log10(d), np.log10(dur)], dtype=float)

    v1, v2 = env_vec(e1), env_vec(e2)
    dist = float(np.linalg.norm(v1 - v2))
    # dist ~0 same, ~2 quite different
    return float(np.clip(np.exp(-dist), 0.0, 1.0))


def timing_score(e1: dict[str, Any], e2: dict[str, Any], bpm: Optional[float] = None) -> float:
    """
    Prefer events close in time OR separated by musical intervals (beat/bar).
    """
    t1 = float(e1.get("start_time") or 0.0)
    t2 = float(e2.get("start_time") or 0.0)
    dt = abs(t1 - t2)
    if dt < 1e-6:
        return 1.0

    # Proximity: close events "resonate" in arrangement sense
    prox = float(np.exp(-dt / 2.0))  # ~0.6 at 1s, ~0.14 at 4s

    beat_bonus = 0.0
    if bpm and bpm > 0:
        beat = 60.0 / bpm
        # How close dt is to an integer number of beats
        n = round(dt / beat)
        if n >= 1:
            err = abs(dt - n * beat) / beat
            beat_bonus = float(np.exp(-err * 4.0)) * 0.5  # up to +0.5

    return float(np.clip(0.55 * prox + 0.45 * max(prox, beat_bonus), 0.0, 1.0))


def resonance_pair_score(
    e1: dict[str, Any],
    e2: dict[str, Any],
    bpm: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Weighted resonance 0–100 between two events.
    Default weights: pitch 0.25, spectral 0.35, envelope 0.25, timing 0.15.
    """
    w = weights or {
        "pitch": 0.25,
        "spectral": 0.35,
        "envelope": 0.25,
        "timing": 0.15,
    }
    p = pitch_class_score(e1, e2)
    s = spectral_score(e1, e2)
    env = envelope_score(e1, e2)
    t = timing_score(e1, e2, bpm=bpm)

    # Clamp pitch score (earlier formula can go slightly negative)
    p = float(np.clip(p, 0.0, 1.0))

    total = (
        w["pitch"] * p
        + w["spectral"] * s
        + w["envelope"] * env
        + w["timing"] * t
    )
    score_100 = float(np.clip(100.0 * total, 0.0, 100.0))
    return {
        "score": round(score_100, 1),
        "pitch": round(100 * p, 1),
        "spectral": round(100 * s, 1),
        "envelope": round(100 * env, 1),
        "timing": round(100 * t, 1),
    }


def compute_resonance(
    events: list[dict[str, Any]],
    bpm: Optional[float] = None,
    top_pairs: int = 12,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """
    Full resonance map for a track.

    Returns:
      {
        "ids": [...],
        "matrix": NxN list of scores 0–100 (diagonal 100),
        "pairs": [ {a, b, score, ...}, ... ] sorted descending (i < j),
        "strongest": top pair or None,
      }
    """
    n = len(events)
    ids = [e.get("id", f"A-{i+1:03d}") for i, e in enumerate(events)]
    matrix = np.zeros((n, n), dtype=float)

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        matrix[i, i] = 100.0
        for j in range(i + 1, n):
            detail = resonance_pair_score(events[i], events[j], bpm=bpm)
            sc = detail["score"]
            matrix[i, j] = sc
            matrix[j, i] = sc
            if sc >= min_score:
                pairs.append(
                    {
                        "a": ids[i],
                        "b": ids[j],
                        "a_class": events[i].get("sound_class"),
                        "b_class": events[j].get("sound_class"),
                        "a_icon": events[i].get("sound_class_icon", ""),
                        "b_icon": events[j].get("sound_class_icon", ""),
                        **detail,
                    }
                )

    pairs.sort(key=lambda x: x["score"], reverse=True)
    top = pairs[:top_pairs]

    return {
        "ids": ids,
        "matrix": matrix.round(1).tolist(),
        "pairs": top,
        "all_pairs": pairs,
        "strongest": top[0] if top else None,
        "n_events": n,
    }


def resonance_for_event(
    event_id: str,
    resonance: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Best partners for a single event from a precomputed resonance map."""
    if not resonance:
        return []
    hits = [
        p
        for p in (resonance.get("all_pairs") or resonance.get("pairs") or [])
        if p["a"] == event_id or p["b"] == event_id
    ]
    hits.sort(key=lambda x: x["score"], reverse=True)
    out = []
    for p in hits[:top_k]:
        partner = p["b"] if p["a"] == event_id else p["a"]
        partner_class = p["b_class"] if p["a"] == event_id else p["a_class"]
        partner_icon = p["b_icon"] if p["a"] == event_id else p["a_icon"]
        out.append(
            {
                "partner": partner,
                "partner_class": partner_class,
                "partner_icon": partner_icon,
                "score": p["score"],
                "pitch": p.get("pitch"),
                "spectral": p.get("spectral"),
                "envelope": p.get("envelope"),
                "timing": p.get("timing"),
            }
        )
    return out
