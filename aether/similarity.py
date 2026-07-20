"""
Sample library similarity engine — pure math, no AI.

Pre-compute MFCC + spectral features for library samples, then rank by:
  1. Cosine similarity on MFCC means + spectral extras
  2. Optional Dynamic Time Warping (DTW) blend on MFCC sequences
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import librosa
import numpy as np
from numpy.linalg import norm

from aether.config import AUDIO_EXTENSIONS, AnalysisSettings, N_MFCC, SIMILARITY_TOP_K
from aether.features import extract_mfcc, extract_spectral
from aether.loader import load_audio


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; returns 0 for zero vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def feature_vector(sample: dict[str, Any]) -> np.ndarray:
    """Fixed-length descriptor for cosine matching."""
    mfcc = sample.get("mfcc_mean") or [0.0] * N_MFCC
    delta = sample.get("mfcc_delta_mean") or [0.0] * N_MFCC
    extras = [
        float(sample.get("spectral_centroid") or 0.0) / 8000.0,
        float(sample.get("spectral_rolloff") or 0.0) / 12000.0,
        float(sample.get("spectral_flatness") or 0.0),
        float(sample.get("spectral_bandwidth") or 0.0) / 8000.0,
    ]
    return np.concatenate(
        [np.asarray(mfcc, dtype=float), np.asarray(delta, dtype=float), extras]
    )


def dtw_mfcc_distance(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """
    Classic DTW on MFCC sequences (frames × coeffs).
    Returns normalized distance (lower = more similar).
    """
    if mfcc_a.ndim == 1:
        mfcc_a = mfcc_a.reshape(1, -1)
    if mfcc_b.ndim == 1:
        mfcc_b = mfcc_b.reshape(1, -1)

    # Cap frames for speed
    max_frames = 64
    if mfcc_a.shape[0] > max_frames:
        idx = np.linspace(0, mfcc_a.shape[0] - 1, max_frames).astype(int)
        mfcc_a = mfcc_a[idx]
    if mfcc_b.shape[0] > max_frames:
        idx = np.linspace(0, mfcc_b.shape[0] - 1, max_frames).astype(int)
        mfcc_b = mfcc_b[idx]

    n, m = mfcc_a.shape[0], mfcc_b.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(mfcc_a[i - 1] - mfcc_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    path_len = n + m
    return float(cost[n, m] / max(path_len, 1))


def analyze_sample_file(
    path: str,
    settings: Optional[AnalysisSettings] = None,
) -> dict[str, Any]:
    """Feature-index one sample file for the library."""
    settings = settings or AnalysisSettings()
    y, sr, meta = load_audio(path, settings=settings)

    # Cap long samples for library index speed
    max_sec = 8.0
    if len(y) > int(max_sec * sr):
        y = y[: int(max_sec * sr)]

    mfcc = extract_mfcc(
        y, sr, n_mfcc=settings.n_mfcc, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    spectral = extract_spectral(
        y, sr, n_fft=settings.n_fft, hop_length=settings.hop_length
    )
    mfcc_mat = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
    ).T  # frames × coeffs

    return {
        "path": path,
        "filename": meta["filename"],
        "duration": meta["duration"],
        **mfcc,
        **spectral,
        "mfcc_matrix": mfcc_mat.astype(np.float32),
    }


def index_library(
    folder: str,
    settings: Optional[AnalysisSettings] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    max_files: int = 500,
) -> list[dict[str, Any]]:
    """Pre-compute features for audio files under a folder (recursive, capped)."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
        if len(files) >= max_files:
            break

    indexed: list[dict[str, Any]] = []
    n = len(files)
    for i, p in enumerate(files):
        try:
            indexed.append(analyze_sample_file(str(p), settings=settings))
        except Exception:
            continue
        if progress_callback is not None:
            progress_callback((i + 1) / max(n, 1))
    return indexed


def find_similar(
    event: dict[str, Any],
    library: list[dict[str, Any]],
    top_k: int = SIMILARITY_TOP_K,
    use_dtw: bool = False,
    event_audio: Optional[np.ndarray] = None,
    sr: int = 44100,
    settings: Optional[AnalysisSettings] = None,
) -> list[dict[str, Any]]:
    """
    Rank library samples by similarity to an event.

    Primary: cosine on MFCC means + spectral extras.
    Optional: blend with DTW on MFCC sequences (0.6 cosine + 0.4 dtw_sim).
    """
    if not library:
        return []

    settings = settings or AnalysisSettings()
    query = feature_vector(event)

    event_mfcc_mat = None
    if use_dtw and event_audio is not None and len(event_audio) > 0:
        event_mfcc_mat = librosa.feature.mfcc(
            y=event_audio,
            sr=sr,
            n_mfcc=settings.n_mfcc,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
        ).T

    scores: list[tuple[float, dict[str, Any]]] = []
    for sample in library:
        cos = cosine_similarity(query, feature_vector(sample))
        score = cos
        if use_dtw and event_mfcc_mat is not None and "mfcc_matrix" in sample:
            dist = dtw_mfcc_distance(event_mfcc_mat, sample["mfcc_matrix"])
            dtw_sim = 1.0 / (1.0 + dist)
            score = 0.6 * cos + 0.4 * dtw_sim
        scores.append((score, sample))

    scores.sort(key=lambda x: x[0], reverse=True)
    results: list[dict[str, Any]] = []
    for score, sample in scores[:top_k]:
        results.append(
            {
                "filename": sample["filename"],
                "path": sample["path"],
                "score": round(float(score), 4),
                "duration": sample.get("duration"),
                "spectral_centroid": sample.get("spectral_centroid"),
            }
        )
    return results


def library_to_serializable(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop non-JSON fields (mfcc_matrix) for display/export."""
    return [{k: v for k, v in s.items() if k != "mfcc_matrix"} for s in library]
