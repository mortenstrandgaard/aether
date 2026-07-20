"""
Session save / load — re-open an analysis without re-running the full pipeline.

Session pack (ZIP):
  manifest.json   — version, selected event, paths, meta
  analysis.json   — full JSON-serializable analysis (no raw audio arrays)
  audio.wav       — optional mono analysis buffer (if available at save time)

Loading restores events, DNA, class, presets, resonance, global metrics.
Playback / spectrogram work when audio.wav is present or source file is reloaded.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import soundfile as sf

from aether import __version__
from aether.export_utils import analysis_to_jsonable
from aether.loader import load_audio

SESSION_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
ANALYSIS_NAME = "analysis.json"
AUDIO_NAME = "audio.wav"


def save_session_zip(
    analysis: dict[str, Any],
    *,
    selected_event_id: Optional[str] = None,
    library_path: Optional[str] = None,
    include_audio: bool = True,
) -> bytes:
    """
    Serialize analysis (+ optional audio) to a .aether.zip session pack.
    """
    payload = analysis_to_jsonable(analysis)
    manifest = {
        "format": "aether_session",
        "format_version": SESSION_FORMAT_VERSION,
        "aether_version": __version__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "selected_event_id": selected_event_id,
        "library_path": library_path,
        "filename": analysis.get("filename"),
        "track_id": analysis.get("track_id"),
        "source": analysis.get("source"),
        "has_audio": False,
        "sample_rate": analysis.get("sr"),
        "n_events": analysis.get("n_events") or len(analysis.get("events") or []),
        "bpm": analysis.get("bpm"),
        "key": analysis.get("key"),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Optional audio buffer
        y = analysis.get("y")
        sr = analysis.get("sr")
        if include_audio and y is not None and sr is not None:
            audio_buf = io.BytesIO()
            sf.write(audio_buf, np.asarray(y, dtype=np.float32), int(sr), format="WAV", subtype="PCM_16")
            zf.writestr(AUDIO_NAME, audio_buf.getvalue())
            manifest["has_audio"] = True
            manifest["sample_rate"] = int(sr)

            # Also store HPSS if present (optional, larger)
            # Skipped by default to keep sessions smaller — can recompute if needed.

        zf.writestr(ANALYSIS_NAME, json.dumps(payload, indent=2).encode("utf-8"))
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2).encode("utf-8"))

    return buf.getvalue()


def load_session_zip(
    file_bytes: bytes,
    *,
    reload_source_audio: bool = False,
) -> dict[str, Any]:
    """
    Load a session pack.

    Returns:
      {
        "manifest": dict,
        "analysis": dict,   # may include y/sr if audio was packed or reloaded
        "selected_event_id": str | None,
        "library_path": str | None,
        "warnings": list[str],
      }
    """
    warnings: list[str] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
        names = set(zf.namelist())
        if ANALYSIS_NAME not in names:
            raise ValueError(f"Invalid session: missing {ANALYSIS_NAME}")

        analysis = json.loads(zf.read(ANALYSIS_NAME).decode("utf-8"))
        manifest: dict[str, Any] = {}
        if MANIFEST_NAME in names:
            manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        else:
            warnings.append("No manifest.json — using defaults.")
            manifest = {
                "format_version": 0,
                "selected_event_id": None,
                "library_path": None,
                "has_audio": AUDIO_NAME in names,
            }

        # Restore audio from pack
        if AUDIO_NAME in names:
            raw = zf.read(AUDIO_NAME)
            y, sr = sf.read(io.BytesIO(raw), always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = np.mean(y, axis=1)
            analysis["y"] = np.asarray(y, dtype=np.float32)
            analysis["sr"] = int(sr)
        else:
            warnings.append(
                "Session has no audio.wav — spectrogram/playback unavailable "
                "unless you re-upload the original file."
            )

    # Optional: reload from original source path (if still on disk)
    if reload_source_audio and analysis.get("y") is None:
        source = analysis.get("source") or manifest.get("source")
        if source and isinstance(source, str) and not source.lower().startswith("http"):
            try:
                y, sr, _meta = load_audio(source)
                analysis["y"] = y
                analysis["sr"] = sr
                warnings.append(f"Reloaded audio from source path: {source}")
            except Exception as exc:
                warnings.append(f"Could not reload source audio: {exc}")

    # Ensure resonance key exists
    if "resonance" not in analysis:
        warnings.append("Session has no resonance map (older export?).")

    selected = manifest.get("selected_event_id")
    if selected is None and analysis.get("events"):
        selected = analysis["events"][0].get("id")

    return {
        "manifest": manifest,
        "analysis": analysis,
        "selected_event_id": selected,
        "library_path": manifest.get("library_path"),
        "warnings": warnings,
    }


def is_session_filename(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith(".aether.zip") or n.endswith(".aether") or (
        n.endswith(".zip") and "aether" in n
    )
