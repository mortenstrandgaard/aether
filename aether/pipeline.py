"""
Full AETHER analysis pipeline orchestration.

  User source (file path or URL)
        ↓
  yt-dlp (if URL) → WAV
        ↓
  Load + normalize (+ optional silence trim)
        ↓
  Global analysis (BPM, key, energy, centroid)
        ↓
  HPSS + onset detection → events A-001…
        ↓
  Per-event feature extraction + note mapping
        ↓
  Sound DNA + sound-class + Serum/Vital preset dump
        ↓
  Cross-event Resonance Map
        ↓
  Result dict (optionally with audio buffers for UI/export)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from aether.config import AnalysisSettings, RESONANCE_TOP_PAIRS
from aether.event_detection import detect_events
from aether.features import extract_all_event_features
from aether.global_analysis import analyze_global
from aether.loader import download_url_to_wav, is_url, load_audio
from aether.resonance import compute_resonance
from aether.sound_dna import enrich_all_events

ProgressCb = Optional[Callable[[str, float], None]]


def run_analysis(
    source: str,
    settings: Optional[AnalysisSettings] = None,
    progress: ProgressCb = None,
    keep_audio: bool = True,
) -> dict[str, Any]:
    """
    Run the complete analysis pipeline on a file path or URL.

    progress(stage_label, fraction_0_to_1) is called at major stages.
    When keep_audio=True, result includes y / y_harmonic / y_percussive for
    playback, spectrogram, and WAV export (not written to JSON export).
    """
    settings = settings or AnalysisSettings()

    def report(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, float(np.clip(frac, 0.0, 1.0)))

    report("Loading audio…", 0.02)
    path = source
    original_source = source

    if is_url(source):
        report("Downloading from URL…", 0.05)
        path = download_url_to_wav(source, sample_rate=settings.sample_rate)

    y, sr, meta = load_audio(path, settings=settings)

    report("Global analysis (BPM, key)…", 0.12)
    global_info = analyze_global(y, sr, settings=settings)

    report("HPSS + event detection…", 0.28)
    det = detect_events(y, sr, settings=settings)

    report("Extracting per-event features…", 0.42)

    def feat_progress(p: float) -> None:
        report("Extracting per-event features…", 0.42 + 0.35 * p)

    events = extract_all_event_features(
        y, sr, det["events"], settings=settings, progress_callback=feat_progress
    )

    report("Sound DNA + classification + presets…", 0.82)
    events = enrich_all_events(
        events,
        bpm=global_info.get("bpm"),
        track_key=global_info.get("key"),
    )

    report("Computing resonance map…", 0.92)
    resonance = compute_resonance(
        events,
        bpm=global_info.get("bpm"),
        top_pairs=RESONANCE_TOP_PAIRS,
    )

    report("Finalizing…", 0.98)
    result: dict[str, Any] = {
        "track_id": meta["track_id"],
        "filename": meta["filename"],
        "source": original_source,
        "duration": global_info["duration"],
        "bpm": global_info["bpm"],
        "key": global_info["key"],
        "rms_mean": global_info["rms_mean"],
        "centroid_mean": global_info["centroid_mean"],
        "rms_times": global_info["rms_times"],
        "rms": global_info["rms"],
        "centroid_times": global_info["centroid_times"],
        "centroid": global_info["centroid"],
        "onsets": det["onsets"],
        "events": events,
        "n_events": len(events),
        "resonance": resonance,
        "settings": settings.to_dict(),
        "sr": sr,
    }

    if keep_audio:
        result["y"] = y
        result["y_harmonic"] = det["y_harmonic"]
        result["y_percussive"] = det["y_percussive"]

    report("Done", 1.0)
    return result


def analysis_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Lightweight summary for UI headers / logging."""
    return {
        "filename": analysis.get("filename"),
        "duration": analysis.get("duration"),
        "bpm": analysis.get("bpm"),
        "key": analysis.get("key"),
        "n_events": analysis.get("n_events") or len(analysis.get("events") or []),
        "track_id": analysis.get("track_id"),
        "strongest_resonance": (analysis.get("resonance") or {}).get("strongest"),
    }
