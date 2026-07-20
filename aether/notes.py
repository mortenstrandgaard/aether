"""
Pitch ↔ musical note utilities (classical, no ML).

Uses equal temperament with A4 = 440 Hz.
"""

from __future__ import annotations

import math
from typing import Optional

from aether.config import NOTE_NAMES


def hz_to_midi(hz: float) -> Optional[float]:
    """Convert frequency (Hz) to continuous MIDI note number."""
    if hz is None or hz <= 0 or math.isnan(hz) or math.isinf(hz):
        return None
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency (Hz)."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_note(hz: float) -> Optional[str]:
    """
    Convert frequency in Hz to nearest note name + octave.

    Example: 369.99 → 'F#4'
    """
    midi = hz_to_midi(hz)
    if midi is None:
        return None
    midi_i = int(round(midi))
    midi_i = max(0, min(127, midi_i))
    name = NOTE_NAMES[midi_i % 12]
    octave = (midi_i // 12) - 1
    return f"{name}{octave}"


def note_to_midi(note: str) -> Optional[int]:
    """
    Parse note names like 'F#4', 'Bb3', 'C5' to MIDI integers.
    Returns None if parsing fails.
    """
    if not note:
        return None
    note = note.strip()
    if len(note) < 2:
        return None

    # Accidental (# or b)
    if len(note) >= 2 and note[1] in ("#", "b"):
        name = note[:2]
        oct_str = note[2:]
    else:
        name = note[0].upper()
        oct_str = note[1:]

    # Normalize flats to sharps
    flat_map = {
        "Db": "C#",
        "Eb": "D#",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
        "db": "C#",
        "eb": "D#",
        "gb": "F#",
        "ab": "G#",
        "bb": "A#",
    }
    name = flat_map.get(name, name if len(name) == 1 else name[0].upper() + name[1:])
    if len(name) == 1:
        name = name.upper()

    if name not in NOTE_NAMES:
        return None
    try:
        octave = int(oct_str)
    except ValueError:
        return None
    return NOTE_NAMES.index(name) + (octave + 1) * 12
