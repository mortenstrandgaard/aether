"""
Layer mixer for AETHER — classical HPSS layers only (not neural stems).

Layers:
  - full        : original mono analysis signal
  - harmonic    : HPSS harmonic (melody / sustained / often vox-ish)
  - percussive  : HPSS percussive (drums / attacks)
  - residual    : full - harmonic - percussive (whatever HPSS left out)

Users can mute/solo layers and export the active mix as WAV.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import soundfile as sf


LAYER_KEYS = ("full", "harmonic", "percussive", "residual")

LAYER_LABELS = {
    "full": "Full mix (original)",
    "harmonic": "Harmonic layer (melody / sustained / often vocal-ish)",
    "percussive": "Percussive layer (hits / drums / attacks)",
    "residual": "Residual (full − harmonic − percussive)",
}


def _as_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).ravel()
    return y


def build_layers(analysis: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract aligned layer arrays from an analysis object."""
    y = analysis.get("y")
    yh = analysis.get("y_harmonic")
    yp = analysis.get("y_percussive")
    if y is None:
        raise ValueError("No full audio buffer in analysis (re-run analysis with audio kept).")

    y = _as_mono(y)
    n = len(y)
    if yh is None:
        yh = np.zeros(n, dtype=np.float32)
    else:
        yh = _as_mono(yh)
        if len(yh) < n:
            yh = np.pad(yh, (0, n - len(yh)))
        else:
            yh = yh[:n]
    if yp is None:
        yp = np.zeros(n, dtype=np.float32)
    else:
        yp = _as_mono(yp)
        if len(yp) < n:
            yp = np.pad(yp, (0, n - len(yp)))
        else:
            yp = yp[:n]

    residual = y - yh - yp
    return {
        "full": y.astype(np.float32),
        "harmonic": yh.astype(np.float32),
        "percussive": yp.astype(np.float32),
        "residual": residual.astype(np.float32),
    }


def mix_layers(
    layers: dict[str, np.ndarray],
    *,
    enabled: Optional[dict[str, bool]] = None,
    gains: Optional[dict[str, float]] = None,
    use_full_only: bool = False,
) -> np.ndarray:
    """
    Mix enabled layers.

    If use_full_only and 'full' is enabled, return full (ignores other layers
    to avoid double-counting when user wants the original track).
    """
    enabled = enabled or {k: True for k in LAYER_KEYS}
    gains = gains or {k: 1.0 for k in LAYER_KEYS}

    if use_full_only or (enabled.get("full") and not any(
        enabled.get(k) for k in ("harmonic", "percussive", "residual")
    )):
        y = layers["full"] * float(gains.get("full", 1.0))
        return _peak_safe(y)

    # When mixing parts, ignore 'full' to avoid double count
    parts = []
    for k in ("harmonic", "percussive", "residual"):
        if enabled.get(k):
            parts.append(layers[k] * float(gains.get(k, 1.0)))
    if not parts:
        # Fallback: if only full toggled
        if enabled.get("full"):
            return _peak_safe(layers["full"] * float(gains.get("full", 1.0)))
        return np.zeros(1, dtype=np.float32)

    n = min(len(p) for p in parts)
    mix = np.zeros(n, dtype=np.float32)
    for p in parts:
        mix += p[:n]
    return _peak_safe(mix)


def _peak_safe(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y


def mix_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def slice_layer(
    y: np.ndarray,
    sr: int,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> np.ndarray:
    i0 = max(0, int(start_sec * sr))
    i1 = len(y) if end_sec is None else min(len(y), int(end_sec * sr))
    if i1 <= i0:
        return np.zeros(1, dtype=np.float32)
    return y[i0:i1].copy()
