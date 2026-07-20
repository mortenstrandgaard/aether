"""
Runtime environment diagnostics for AETHER.

Detects optional backends (Essentia, Aubio), ffmpeg, Python version, OS,
and reports what's available vs recommended.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from typing import Any, Optional


def _pkg_version(module_name: str, version_attrs: tuple[str, ...] = ("__version__", "version")) -> Optional[str]:
    """Return package version string or None if not importable."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    for attr in version_attrs:
        v = getattr(mod, attr, None)
        if v is None:
            continue
        if callable(v):
            try:
                return str(v())
            except Exception:
                continue
        return str(v)
    return "installed"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_ffmpeg() -> dict[str, Any]:
    """Locate ffmpeg on PATH and try to read its version line."""
    path = shutil.which("ffmpeg")
    if not path:
        return {
            "ok": False,
            "path": None,
            "version": None,
            "detail": "Not found on PATH — required for YouTube/URL downloads (yt-dlp).",
        }
    version = None
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        line = (proc.stdout or proc.stderr or "").splitlines()
        version = line[0].strip() if line else "ffmpeg (unknown version)"
    except Exception as exc:
        version = f"found but could not query version ({exc})"
    return {
        "ok": True,
        "path": path,
        "version": version,
        "detail": "Ready for yt-dlp URL downloads.",
    }


def check_essentia() -> dict[str, Any]:
    """
    Essentia is optional (better key detection).

    Official wheels are strongest on Linux; macOS often via conda;
    Windows + modern Python (esp. 3.13) usually has no pip wheel.
    """
    # Prefer essentia.standard path used by AETHER
    try:
        import essentia  # type: ignore
        import essentia.standard as es  # type: ignore

        ver = getattr(essentia, "__version__", None) or "installed"
        # Smoke-test KeyExtractor exists
        _ = es.KeyExtractor
        return {
            "ok": True,
            "version": str(ver),
            "detail": "KeyExtractor available — will be used for key detection.",
            "recommended_platforms": _essentia_platform_notes(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "version": None,
            "detail": (
                f"Not available ({type(exc).__name__}: {exc}). "
                "AETHER falls back to librosa chroma + Krumhansl–Schmuckler."
            ),
            "recommended_platforms": _essentia_platform_notes(),
        }


def _essentia_platform_notes() -> list[str]:
    return [
        "Best: Linux (Ubuntu/Debian) with Python 3.10–3.11 — pip or conda often works.",
        "Good: macOS (Intel/Apple Silicon) via conda-forge / MTG channels — pip may work on 3.10–3.11.",
        "Hard: Windows native pip — frequently no wheel; use WSL2 (Ubuntu) if you need Essentia on a PC.",
        "Avoid: Python 3.13 for Essentia — use 3.11 (or 3.10) for optional backends.",
        "AETHER does NOT require Essentia — key detection works without it.",
    ]


def check_aubio() -> dict[str, Any]:
    try:
        import aubio  # type: ignore

        ver = getattr(aubio, "version", None) or getattr(aubio, "__version__", "installed")
        if callable(ver):
            try:
                ver = ver()
            except Exception:
                ver = "installed"
        return {
            "ok": True,
            "version": str(ver),
            "detail": "Onset backend available (Settings → Onset backend → auto/aubio).",
        }
    except Exception as exc:
        return {
            "ok": False,
            "version": None,
            "detail": (
                f"Not installed ({type(exc).__name__}). "
                "Librosa onset detection is used instead. Optional: pip install aubio"
            ),
        }


def check_core_packages() -> dict[str, dict[str, Any]]:
    """Required runtime packages."""
    packages = {
        "streamlit": "streamlit",
        "librosa": "librosa",
        "numpy": "numpy",
        "scipy": "scipy",
        "soundfile": "soundfile",
        "plotly": "plotly",
        "mido": "mido",
        "yt_dlp": "yt_dlp",
    }
    out: dict[str, dict[str, Any]] = {}
    for label, mod in packages.items():
        ver = _pkg_version(mod)
        out[label] = {
            "ok": ver is not None,
            "version": ver,
        }
    return out


def python_recommendation(version_info: Optional[tuple] = None) -> dict[str, Any]:
    vi = version_info or sys.version_info
    major, minor = vi[0], vi[1]
    recommended = major == 3 and minor in (10, 11, 12)
    ideal = major == 3 and minor == 11
    return {
        "version": f"{major}.{minor}.{vi[2] if len(vi) > 2 else 0}",
        "recommended": recommended,
        "ideal": ideal,
        "detail": (
            "Ideal: Python 3.11. Supported: 3.10–3.12. "
            "3.13 works for core AETHER but optional packages (Essentia) often missing."
            if not ideal
            else "Python 3.11 — ideal for AETHER + optional backends."
        ),
    }


def gather_environment_report() -> dict[str, Any]:
    """Full diagnostic snapshot for UI / logging."""
    py = python_recommendation()
    core = check_core_packages()
    core_ok = all(v["ok"] for v in core.values())
    essentia = check_essentia()
    aubio = check_aubio()
    ffmpeg = check_ffmpeg()

    # Overall readiness tiers
    can_run_local = core_ok
    can_url = core_ok and ffmpeg["ok"]
    has_best_key = essentia["ok"]
    has_aubio_onsets = aubio["ok"]

    if can_run_local and can_url and (has_best_key or has_aubio_onsets):
        status = "excellent"
        status_label = "Excellent — core + extras ready"
    elif can_run_local and can_url:
        status = "good"
        status_label = "Good — full local + URL analysis"
    elif can_run_local:
        status = "ok"
        status_label = "OK — local files work; install ffmpeg for YouTube"
    else:
        status = "broken"
        status_label = "Missing required packages — pip install -r requirements.txt"

    return {
        "status": status,
        "status_label": status_label,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": py,
        "core": core,
        "essentia": essentia,
        "aubio": aubio,
        "ffmpeg": ffmpeg,
        "capabilities": {
            "local_file_analysis": can_run_local,
            "youtube_url": can_url,
            "essentia_key": has_best_key,
            "aubio_onsets": has_aubio_onsets,
        },
        "install_hints": _install_hints(
            platform.system(),
            py["version"],
            ffmpeg["ok"],
            aubio["ok"],
            essentia["ok"],
        ),
    }


def _install_hints(
    system: str,
    py_version: str,
    has_ffmpeg: bool,
    has_aubio: bool,
    has_essentia: bool,
) -> list[str]:
    hints: list[str] = []
    if not has_ffmpeg:
        if system == "Windows":
            hints.append("Install ffmpeg:  winget install ffmpeg   (then restart terminal)")
        elif system == "Darwin":
            hints.append("Install ffmpeg:  brew install ffmpeg")
        else:
            hints.append("Install ffmpeg:  sudo apt install ffmpeg   # or your distro equivalent")
    if not has_aubio:
        hints.append("Optional better onsets:  pip install aubio")
    if not has_essentia:
        if system == "Windows":
            hints.append(
                "Essentia: skip on native Windows, or use WSL2 Ubuntu + Python 3.11 "
                "(see SETUP.md). AETHER key detection works without it."
            )
        elif system == "Darwin":
            hints.append(
                "Optional Essentia (macOS): prefer conda with Python 3.11 — "
                "conda install -c conda-forge essentia   (or MTG docs). Not required."
            )
        else:
            hints.append(
                "Optional Essentia (Linux): pip install essentia  # Python 3.10–3.11 often works"
            )
    if py_version.startswith("3.13"):
        hints.append(
            "You are on Python 3.13 — core AETHER is fine; for Essentia/max compatibility "
            "create a 3.11 venv (see SETUP.md)."
        )
    if not hints:
        hints.append("Environment looks complete for AETHER workflows.")
    return hints
