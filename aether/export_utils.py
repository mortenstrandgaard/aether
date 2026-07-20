"""
Export helpers: JSON analysis, MIDI note map, individual event WAVs, full ZIP.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
import soundfile as sf

from aether.notes import note_to_midi


def analysis_to_jsonable(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Strip non-serializable / heavy UI-only arrays for a clean JSON export.

    Drops: raw audio (y, HPSS), long RMS/centroid curves.
    Keeps: track metadata, settings, all events with features.
    """
    skip_keys = {
        "y",
        "y_harmonic",
        "y_percussive",
        "sr",
        "rms",
        "rms_times",
        "centroid",
        "centroid_times",
    }
    out: dict[str, Any] = {}
    for k, v in analysis.items():
        if k in skip_keys:
            continue
        if k == "events":
            out["events"] = [_event_jsonable(e) for e in v]
        elif k == "resonance":
            # Keep matrix + top pairs; drop full all_pairs if huge
            if isinstance(v, dict):
                res = {
                    "ids": v.get("ids"),
                    "matrix": v.get("matrix"),
                    "pairs": v.get("pairs"),
                    "strongest": v.get("strongest"),
                    "n_events": v.get("n_events"),
                }
                out["resonance"] = res
            else:
                out["resonance"] = v
        else:
            out[k] = v
    return out


def _event_jsonable(event: dict[str, Any]) -> dict[str, Any]:
    e = dict(event)
    contour = e.get("pitch_contour")
    if isinstance(contour, list) and len(contour) > 64:
        e["pitch_contour"] = contour[:64]
    # Drop bulky nested classification score table if present (keep summary fields)
    if isinstance(e.get("classification"), dict):
        cls = dict(e["classification"])
        cls.pop("scores", None)
        e["classification"] = cls
    return e


def export_json_bytes(analysis: dict[str, Any], indent: int = 2) -> bytes:
    """Serialize analysis to UTF-8 JSON bytes."""
    data = analysis_to_jsonable(analysis)
    return json.dumps(data, indent=indent).encode("utf-8")


def export_midi_bytes(
    analysis: dict[str, Any],
    ticks_per_beat: int = 480,
) -> bytes:
    """
    Map events with known notes to a single-track MIDI file.

    Note-on at start_time, note-off at end_time.
    Velocity derived from harmonic_ratio when present.
    """
    try:
        from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
    except ImportError as exc:
        raise RuntimeError("mido is required for MIDI export: pip install mido") from exc

    bpm = float(analysis.get("bpm") or 120.0)
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))
    track.append(
        MetaMessage(
            "track_name",
            name=str(analysis.get("filename") or "AETHER"),
            time=0,
        )
    )

    abs_events: list[tuple[int, str, int, int]] = []  # tick, type, note, velocity
    for ev in analysis.get("events") or []:
        note_name = ev.get("note")
        midi_note = note_to_midi(note_name) if note_name else None
        if midi_note is None:
            continue
        start_tick = int(round(float(ev["start_time"]) * (bpm / 60.0) * ticks_per_beat))
        end_tick = int(round(float(ev["end_time"]) * (bpm / 60.0) * ticks_per_beat))
        if end_tick <= start_tick:
            end_tick = start_tick + max(1, int(0.1 * (bpm / 60.0) * ticks_per_beat))
        vel = 80
        if ev.get("harmonic_ratio") is not None:
            vel = int(np.clip(40 + 80 * float(ev["harmonic_ratio"]), 1, 127))
        abs_events.append((start_tick, "on", midi_note, vel))
        abs_events.append((end_tick, "off", midi_note, 0))

    # Note-offs first at the same tick avoids stuck notes
    abs_events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))
    last = 0
    for tick, kind, note, vel in abs_events:
        delta = max(0, tick - last)
        if kind == "on":
            track.append(Message("note_on", note=note, velocity=vel, time=delta))
        else:
            track.append(Message("note_off", note=note, velocity=0, time=delta))
        last = tick

    track.append(MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def extract_event_audio(
    y: np.ndarray,
    sr: int,
    event: dict[str, Any],
) -> np.ndarray:
    """Slice mono audio for one event."""
    start = float(event["start_time"])
    end = float(event["end_time"])
    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    if i1 <= i0:
        return np.zeros(1, dtype=np.float32)
    return y[i0:i1].astype(np.float32)


def export_event_wav_bytes(
    y: np.ndarray,
    sr: int,
    event: dict[str, Any],
) -> bytes:
    """PCM16 WAV bytes for a single event."""
    seg = extract_event_audio(y, sr, event)
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def export_all_events_zip(
    y: np.ndarray,
    sr: int,
    analysis: dict[str, Any],
) -> bytes:
    """ZIP: analysis.json + events/A-XXX.wav for every event."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis.json", export_json_bytes(analysis))
        for ev in analysis.get("events") or []:
            name = f"events/{ev['id']}.wav"
            zf.writestr(name, export_event_wav_bytes(y, sr, ev))
    return buf.getvalue()
