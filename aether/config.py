"""
AETHER configuration: analysis defaults, musical constants, and UI palette.

All user-tunable knobs live in AnalysisSettings so the Streamlit advanced panel
and the pipeline share one source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Core DSP defaults
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
N_MFCC = 13
MIN_EVENT_MS = 80
MAX_EVENT_MS = 4000
DEFAULT_ONSET_THRESHOLD = 0.07
DEFAULT_MERGE_GAP_MS = 50
SIMILARITY_TOP_K = 12

# Preferred onset backend: "auto" tries aubio then librosa
ONSET_BACKEND = "auto"  # "auto" | "librosa" | "aubio"


@dataclass
class AnalysisSettings:
    """User-tunable analysis parameters (Advanced settings in UI)."""

    sample_rate: int = SAMPLE_RATE
    mono: bool = True
    trim_silence: bool = True
    trim_top_db: float = 40.0
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD
    min_event_ms: int = MIN_EVENT_MS
    max_event_ms: int = MAX_EVENT_MS
    merge_gap_ms: float = DEFAULT_MERGE_GAP_MS
    n_mfcc: int = N_MFCC
    hop_length: int = 512
    n_fft: int = 2048
    hpss_margin: float = 1.0
    onset_backend: str = ONSET_BACKEND

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cache_key(self) -> tuple:
        """Stable tuple for @st.cache_data keys."""
        return (
            self.sample_rate,
            self.mono,
            self.trim_silence,
            self.trim_top_db,
            self.onset_threshold,
            self.min_event_ms,
            self.max_event_ms,
            self.merge_gap_ms,
            self.n_mfcc,
            self.hop_length,
            self.n_fft,
            self.hpss_margin,
            self.onset_backend,
        )


# ---------------------------------------------------------------------------
# Musical constants
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl–Schmuckler key profiles (major / minor)
MAJOR_PROFILE = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
]
MINOR_PROFILE = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
]

WAVEFORM_TYPES = ("sine", "triangle", "sawtooth", "square", "noise", "complex")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}

# Resonance map defaults
RESONANCE_TOP_PAIRS = 12
RESONANCE_HIGHLIGHT_THRESHOLD = 70.0  # % — "strong" pairs


# ---------------------------------------------------------------------------
# Dark theme palette — music production / studio vibe
# ---------------------------------------------------------------------------

COLORS = {
    # Surfaces
    "bg": "#0b0b0e",
    "surface": "#141418",
    "surface2": "#1c1c22",
    "surface3": "#25252d",
    "border": "#2e2e38",
    # Typography
    "text": "#ececf1",
    "muted": "#8a8a9a",
    # Brand accents
    "accent": "#7c5cfc",      # violet
    "accent2": "#00d4aa",     # teal
    "accent3": "#4d9fff",     # blue
    # Event types (HPSS)
    "harmonic": "#7c5cfc",
    "percussive": "#ff6b4a",
    "mixed": "#00d4aa",
    # Status
    "warning": "#f5a623",
    "error": "#ff4d6a",
    "success": "#00d4aa",
}

# Sound-class colours (rule-based classification UI)
SOUND_CLASS_COLORS = {
    "Kick": "#ff6b4a",
    "Snare": "#ff8f6b",
    "HiHat": "#ffd166",
    "Perc": "#f5a623",
    "Bass": "#4d9fff",
    "Lead": "#7c5cfc",
    "Pluck": "#c084fc",
    "Pad": "#00d4aa",
    "Riser": "#22d3ee",
    "Hit": "#fb7185",
    "Texture": "#94a3b8",
    "FX": "#e879f9",
    "Atmos": "#67e8f9",
    "Unknown": "#8a8a9a",
}
