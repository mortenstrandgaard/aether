"""
Audio loading, preprocessing, and optional URL download via yt-dlp.

Pipeline step 1:
  - Accept local path or http(s) URL
  - Load at 44.1 kHz mono
  - Peak-normalize to [-1, 1]
  - Optional silence trim
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf

from aether.config import SAMPLE_RATE, AnalysisSettings


def is_url(source: str) -> bool:
    """Return True if source looks like an http(s) URL."""
    s = (source or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def download_url_to_wav(
    url: str,
    out_dir: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """
    Download audio from a URL (YouTube etc.) with yt-dlp and convert to WAV.

    Requires yt-dlp and typically ffmpeg on PATH.
    Returns absolute path to a mono WAV at `sample_rate`.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for URL downloads. Install with: pip install yt-dlp"
        ) from exc

    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="aether_dl_"))
    work.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work / "download.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    title = "download"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                title = info.get("title") or title
    except Exception as exc:
        raise RuntimeError(f"Failed to download audio from URL: {exc}") from exc

    candidates = list(work.glob("download*.wav")) + list(work.glob("*.wav"))
    if not candidates:
        # Fallback: re-encode whatever audio container yt-dlp left behind
        audio_files = [
            p
            for p in work.iterdir()
            if p.suffix.lower() in {".webm", ".m4a", ".mp3", ".opus", ".ogg", ".wav", ".flac"}
        ]
        if not audio_files:
            raise RuntimeError("yt-dlp finished but no audio file was found.")
        y, _ = librosa.load(str(audio_files[0]), sr=sample_rate, mono=True)
        wav_path = work / "download.wav"
        sf.write(str(wav_path), y, sample_rate)
    else:
        wav_path = candidates[0]
        y, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)

    final = work / "aether_input.wav"
    sf.write(str(final), y, sample_rate)
    try:
        (work / "title.txt").write_text(title, encoding="utf-8")
    except OSError:
        pass
    return str(final)


def load_audio(
    path: str,
    settings: Optional[AnalysisSettings] = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Load audio at configured sample rate (mono for analysis).

    - Peak-normalize to [-1, 1]
    - Optionally trim leading/trailing silence

    Returns (y, sr, meta).
    """
    settings = settings or AnalysisSettings()
    sr = settings.sample_rate

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, loaded_sr = librosa.load(path, sr=sr, mono=settings.mono)
    except Exception as exc:
        raise RuntimeError(f"Could not load audio '{path}': {exc}") from exc

    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    # Peak normalize
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    trim_offset_sec = 0.0
    if settings.trim_silence:
        yt, idx = librosa.effects.trim(y, top_db=settings.trim_top_db)
        if yt.size > 0:
            trim_offset_sec = float(idx[0]) / sr
            y = yt
            peak = float(np.max(np.abs(y)))
            if peak > 0:
                y = y / peak

    meta: dict[str, Any] = {
        "path": path,
        "filename": os.path.basename(path),
        "sample_rate": sr,
        "duration": float(len(y) / sr),
        "n_samples": int(len(y)),
        "trim_offset_sec": trim_offset_sec,
        "track_id": str(uuid.uuid4()),
    }
    return y.astype(np.float32), sr, meta


def save_temp_upload(file_bytes: bytes, filename: str) -> str:
    """Persist uploaded bytes to a temp file; return its path."""
    suffix = Path(filename).suffix or ".wav"
    fd, path = tempfile.mkstemp(prefix="aether_up_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
