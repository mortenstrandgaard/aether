"""
Event detection — the heart of AETHER (pipeline step 3).

1. HPSS (harmonic–percussive source separation)
2. Onset detection on both components (aubio if available, else librosa)
3. Merge close onsets
4. Build segments (min 80 ms, max 4 s by default)
5. Assign IDs A-001, A-002, … and classify harmonic / percussive / mixed
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np

from aether.config import AnalysisSettings


def hpss(y: np.ndarray, margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic–Percussive Source Separation via librosa median filtering."""
    y_harm, y_perc = librosa.effects.hpss(y, margin=margin)
    return y_harm.astype(np.float32), y_perc.astype(np.float32)


def _onsets_aubio(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Onset times via aubio (optional dependency).
    Returns None if aubio is not installed or fails.
    """
    try:
        import aubio  # type: ignore
    except Exception:
        return None

    try:
        hop = 512
        win = 1024
        # aubio expects float32 mono
        src = y.astype(np.float32)
        onset_o = aubio.onset("default", win, hop, sr)
        times: list[float] = []
        for i in range(0, len(src) - hop, hop):
            frame = src[i : i + hop]
            if len(frame) < hop:
                frame = np.pad(frame, (0, hop - len(frame)))
            if onset_o(frame):
                times.append(float(onset_o.get_last_s()))
        return np.asarray(times, dtype=float) if times else np.asarray([], dtype=float)
    except Exception:
        return None


def _onsets_librosa(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.07,
) -> np.ndarray:
    """Onset times via librosa onset strength + peak picking."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)

    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        delta=threshold,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return times.astype(float)


def detect_onsets(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.07,
    backend: str = "auto",
) -> np.ndarray:
    """
    Return onset times (seconds).

    backend:
      - "aubio"  → aubio only (falls back to librosa on failure)
      - "librosa" → librosa only
      - "auto"   → try aubio, then librosa
    """
    if backend in ("auto", "aubio"):
        aub = _onsets_aubio(y, sr)
        if aub is not None and len(aub) > 0:
            return aub
        if backend == "aubio":
            # Explicit aubio request but empty/fail → still use librosa
            pass
    return _onsets_librosa(y, sr, hop_length=hop_length, threshold=threshold)


def merge_onsets(onsets: np.ndarray, gap_sec: float) -> np.ndarray:
    """Merge onsets closer than gap_sec, keeping the first of each cluster."""
    if len(onsets) == 0:
        return onsets
    onsets = np.sort(onsets.astype(float))
    merged = [float(onsets[0])]
    for t in onsets[1:]:
        if float(t) - merged[-1] >= gap_sec:
            merged.append(float(t))
    return np.asarray(merged, dtype=float)


def build_segments(
    onsets: np.ndarray,
    duration: float,
    min_sec: float,
    max_sec: float,
) -> list[tuple[float, float]]:
    """
    Build [start, end) segments from onsets.

    End = next onset or start+max_sec, clamped to duration.
    Segments shorter than min_sec are dropped.
    """
    if len(onsets) == 0:
        end = min(duration, max_sec)
        return [(0.0, end)] if end >= min_sec else []

    segments: list[tuple[float, float]] = []
    for i, start in enumerate(onsets):
        start = float(start)
        if i + 1 < len(onsets):
            end = float(onsets[i + 1])
        else:
            end = float(duration)
        end = min(end, start + max_sec, duration)
        if end - start >= min_sec:
            segments.append((start, end))
    return segments


def classify_event_type(
    start: float,
    end: float,
    y_harm: np.ndarray,
    y_perc: np.ndarray,
    sr: int,
) -> tuple[str, float]:
    """
    Classify event as harmonic / percussive / mixed from HPSS energy ratio.

    Returns (type, harmonic_ratio) where ratio = harm / (harm + perc).
    """
    i0 = max(0, int(start * sr))
    i1 = min(len(y_harm), int(end * sr))
    if i1 <= i0:
        return "mixed", 0.5

    h = float(np.sum(y_harm[i0:i1] ** 2))
    p = float(np.sum(y_perc[i0:i1] ** 2))
    total = h + p + 1e-12
    ratio = h / total

    if ratio > 0.65:
        etype = "harmonic"
    elif ratio < 0.35:
        etype = "percussive"
    else:
        etype = "mixed"
    return etype, float(ratio)


def detect_events(
    y: np.ndarray,
    sr: int,
    settings: Optional[AnalysisSettings] = None,
) -> dict[str, Any]:
    """
    Full event detection pipeline.

    Returns dict with:
      - y_harmonic, y_percussive
      - onsets (list of seconds)
      - events (list of timing/type dicts; features filled later)
    """
    settings = settings or AnalysisSettings()
    hop = settings.hop_length
    min_sec = settings.min_event_ms / 1000.0
    max_sec = settings.max_event_ms / 1000.0
    merge_gap = settings.merge_gap_ms / 1000.0
    duration = float(len(y) / sr)
    backend = getattr(settings, "onset_backend", "auto")

    y_harm, y_perc = hpss(y, margin=settings.hpss_margin)

    on_h = detect_onsets(
        y_harm, sr, hop_length=hop, threshold=settings.onset_threshold, backend=backend
    )
    on_p = detect_onsets(
        y_perc, sr, hop_length=hop, threshold=settings.onset_threshold, backend=backend
    )
    all_onsets = merge_onsets(np.concatenate([on_h, on_p]), gap_sec=merge_gap)
    segments = build_segments(all_onsets, duration, min_sec=min_sec, max_sec=max_sec)

    events: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(segments, start=1):
        etype, h_ratio = classify_event_type(start, end, y_harm, y_perc, sr)
        events.append(
            {
                "id": f"A-{idx:03d}",
                "start_time": round(start, 4),
                "end_time": round(end, 4),
                "duration": round(end - start, 4),
                "type": etype,
                "harmonic_ratio": round(h_ratio, 4),
            }
        )

    return {
        "y_harmonic": y_harm,
        "y_percussive": y_perc,
        "onsets": all_onsets.tolist(),
        "events": events,
    }
