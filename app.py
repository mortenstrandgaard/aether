"""
AETHER Analyzer v1.0 — Streamlit UI
See every sound. Understand every layer.

Classical DSP only — zero AI / zero ML.

v1 features + extensions:
  - Sound DNA fingerprint
  - Rule-based sound-type classification
  - Serum / Vital preset dump
  - Cross-event Resonance Map
  - Environment status panel
  - Session save / load (.aether.zip)
  - Forensics: compare two sounds/voices (% + written report)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st

from aether import __version__
from aether.config import (
    RESONANCE_HIGHLIGHT_THRESHOLD,
    AnalysisSettings,
    COLORS,
    SIMILARITY_TOP_K,
    SOUND_CLASS_COLORS,
)
from aether.environment import gather_environment_report
from aether.export_utils import (
    export_all_events_zip,
    export_event_wav_bytes,
    export_json_bytes,
    export_midi_bytes,
    extract_event_audio,
)
from aether.forensics import (
    compare_audio_arrays,
    compare_files,
    compare_reference_vs_many,
    write_analysis_report,
)
from aether.loader import is_url, load_audio, save_temp_upload
from aether.pipeline import run_analysis
from aether.resonance import resonance_for_event
from aether.session_io import load_session_zip, save_session_zip
from aether.similarity import find_similar, index_library
from aether.sound_dna import format_dna_display
from aether.explain import (
    explain_score_snapshot,
    render_dimension_help_markdown,
    render_voice_guide_markdown,
)
from aether.mixer import (
    LAYER_LABELS,
    build_layers,
    mix_layers,
    mix_to_wav_bytes,
    slice_layer,
)
from aether.pitch_forensics import build_pitch_forensics
from aether.pitch_scrubber import prepare_scrubber_payload, render_pitch_scrubber
from aether.voice_character import (
    analyze_voice_character_file,
    compare_voice_characters,
)
from aether.viz import (
    hpss_figure,
    mfcc_bar_figure,
    resonance_heatmap_figure,
    spectrogram_figure,
    timeline_figure,
)
from aether.viz_forensics import (
    forensics_dimension_bars,
    forensics_pitch_overlay,
    forensics_ranking_bars,
    forensics_reliability_bar,
    forensics_score_gauge,
    forensics_spectrogram_pair,
    voice_character_pitch_figure,
    voice_character_summary_bars,
)
from aether.viz_pitch import pitch_lane_figure

# ---------------------------------------------------------------------------
# Page config & dark theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AETHER Analyzer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, .stApp {{
        font-family: 'IBM Plex Sans', system-ui, sans-serif;
    }}

    .stApp {{
        background: linear-gradient(160deg, {COLORS['bg']} 0%, #12121a 55%, #08080c 100%);
        color: {COLORS['text']};
    }}

    .aether-header {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 1.25rem;
    }}
    .aether-logo {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 0 24px rgba(124, 92, 252, 0.35);
    }}
    .aether-title {{
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin: 0;
        background: linear-gradient(90deg, {COLORS['text']}, {COLORS['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .aether-tagline {{
        color: {COLORS['muted']};
        font-size: 0.9rem;
        margin: 0.15rem 0 0 0;
    }}
    .aether-badge {{
        margin-left: auto;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        color: {COLORS['muted']};
        border: 1px solid {COLORS['border']};
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: {COLORS['surface']};
    }}

    .metric-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 0.75rem 0 1.25rem 0;
    }}
    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }}
    .metric-label {{
        color: {COLORS['muted']};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }}
    .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        color: {COLORS['text']};
    }}
    .metric-value.accent {{ color: {COLORS['accent2']}; }}

    .param-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }}
    .param-chip {{
        background: {COLORS['surface2']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
    }}
    .param-k {{
        color: {COLORS['muted']};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .param-v {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        color: {COLORS['text']};
        margin-top: 0.1rem;
    }}

    .type-harmonic {{ color: {COLORS['harmonic']}; }}
    .type-percussive {{ color: {COLORS['percussive']}; }}
    .type-mixed {{ color: {COLORS['mixed']}; }}

    .dna-box {{
        background: linear-gradient(135deg, {COLORS['surface2']} 0%, #1a1530 100%);
        border: 1px solid {COLORS['accent']};
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        line-height: 1.45;
        color: {COLORS['text']};
        white-space: pre-wrap;
        margin: 0.4rem 0 0.75rem 0;
    }}
    .dna-code {{
        color: {COLORS['accent2']};
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    .class-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid {COLORS['border']};
        margin: 0.25rem 0 0.6rem 0;
    }}

    .preset-box {{
        background: {COLORS['surface2']};
        border: 1px solid {COLORS['border']};
        border-left: 3px solid {COLORS['accent2']};
        border-radius: 8px;
        padding: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        line-height: 1.4;
        color: {COLORS['text']};
        white-space: pre-wrap;
        max-height: 360px;
        overflow-y: auto;
    }}

    .res-pair {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0.55rem;
        border-radius: 8px;
        background: {COLORS['surface2']};
        border: 1px solid {COLORS['border']};
        margin-bottom: 0.35rem;
        font-size: 0.85rem;
    }}
    .res-score {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: {COLORS['accent2']};
    }}
    .res-score.hot {{ color: #e8fff7; text-shadow: 0 0 8px {COLORS['accent2']}; }}

    .env-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid {COLORS['border']};
        font-size: 0.82rem;
    }}
    .env-ok {{ color: {COLORS['success']}; }}
    .env-bad {{ color: {COLORS['warning']}; }}
    .env-muted {{ color: {COLORS['muted']}; font-size: 0.75rem; }}

    section[data-testid="stSidebar"] {{
        background: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}

    /* Do NOT hide header — Streamlit 1.39+ puts the app shell there;
       visibility:hidden on header causes a blank white screen. */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* Ensure main block is visible above dark background */
    .block-container {{
        padding-top: 1.5rem;
    }}
    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "analysis": None,
        "selected_event_id": None,
        "library": None,
        "library_path": None,
        "matches": None,
        "last_error": None,
        "timeline_color_mode": "sound_class",
        "session_zip_bytes": None,
        "session_zip_stem": None,
        "app_mode": "track",
        "forensics_result": None,
        "forensics_ranking": None,
        "voice_char_result": None,
        "voice_char_compare": None,
        "pitch_forensics": None,
        "scrubber_payload": None,
        "mixer_enabled": {
            "full": False,
            "harmonic": True,
            "percussive": True,
            "residual": False,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _settings_from_ui(
    onset_threshold: float,
    min_event_ms: int,
    max_event_ms: int,
    merge_gap_ms: float,
    trim_silence: bool,
    hpss_margin: float,
    onset_backend: str,
) -> AnalysisSettings:
    return AnalysisSettings(
        onset_threshold=onset_threshold,
        min_event_ms=min_event_ms,
        max_event_ms=max_event_ms,
        merge_gap_ms=merge_gap_ms,
        trim_silence=trim_silence,
        hpss_margin=hpss_margin,
        onset_backend=onset_backend,
    )


def _get_event(analysis: dict, event_id: str) -> Optional[dict]:
    for e in analysis.get("events") or []:
        if e["id"] == event_id:
            return e
    return None


@st.cache_data(show_spinner=False, ttl=60)
def cached_environment_report() -> dict[str, Any]:
    """Cache env diagnostics briefly so the sidebar stays snappy."""
    return gather_environment_report()


def render_environment_panel() -> None:
    """Sidebar: detect Python, core packages, aubio, Essentia, ffmpeg."""
    with st.expander("Environment status", expanded=False):
        if st.button("Refresh diagnostics", key="env_refresh", use_container_width=True):
            cached_environment_report.clear()

        report = cached_environment_report()
        status = report.get("status", "ok")
        label = report.get("status_label", "")
        if status == "excellent":
            st.success(label)
        elif status == "good":
            st.success(label)
        elif status == "ok":
            st.warning(label)
        else:
            st.error(label)

        py = report.get("python") or {}
        plat = report.get("platform") or {}
        st.caption(
            f"{plat.get('system', '?')} {plat.get('release', '')} · "
            f"{plat.get('machine', '')} · Python **{py.get('version', '?')}**"
        )
        if not py.get("recommended"):
            st.caption("Tip: Python **3.11** is ideal — see SETUP.md")

        def row(name: str, ok: bool, ver: Any = None) -> None:
            mark = "✓" if ok else "○"
            cls = "env-ok" if ok else "env-bad"
            ver_s = f" · {ver}" if ver else ""
            st.markdown(
                f'<div class="env-row"><span>{name}</span>'
                f'<span class="{cls}">{mark}{ver_s}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Core**")
        for name, info in (report.get("core") or {}).items():
            row(name, bool(info.get("ok")), info.get("version"))

        st.markdown("**Optional**")
        aub = report.get("aubio") or {}
        ess = report.get("essentia") or {}
        ff = report.get("ffmpeg") or {}
        row("aubio (onsets)", bool(aub.get("ok")), aub.get("version"))
        row("essentia (key)", bool(ess.get("ok")), ess.get("version"))
        row("ffmpeg (YouTube)", bool(ff.get("ok")), None)
        if ff.get("version"):
            st.caption(str(ff["version"])[:80])

        caps = report.get("capabilities") or {}
        st.markdown("**Capabilities**")
        row("Local file analysis", bool(caps.get("local_file_analysis")))
        row("YouTube / URL", bool(caps.get("youtube_url")))
        row("Essentia key", bool(caps.get("essentia_key")))
        row("Aubio onsets", bool(caps.get("aubio_onsets")))

        hints = report.get("install_hints") or []
        if hints:
            st.markdown("**Install hints**")
            for h in hints:
                st.caption(f"• {h}")
        st.caption("Full guide: **SETUP.md** in the project folder.")


def render_session_panel(analysis: Optional[dict]) -> None:
    """Sidebar: save / load .aether.zip sessions."""
    with st.expander("Session save / load", expanded=False):
        st.caption(
            "Save analysis (DNA, class, presets, resonance) and reopen later "
            "without re-running DSP. Optional audio pack for playback."
        )

        # --- Load ---
        session_file = st.file_uploader(
            "Load session (.aether.zip)",
            type=["zip"],
            key="session_upload",
            help="Session pack exported from AETHER",
        )
        if session_file is not None and st.button(
            "Open session", key="session_load_btn", use_container_width=True
        ):
            try:
                result = load_session_zip(session_file.getvalue())
                st.session_state.analysis = result["analysis"]
                st.session_state.selected_event_id = result.get("selected_event_id")
                st.session_state.matches = None
                if result.get("library_path"):
                    st.session_state.library_path = result["library_path"]
                for w in result.get("warnings") or []:
                    st.warning(w)
                st.success(
                    f"Loaded session · "
                    f"{result['analysis'].get('n_events') or len(result['analysis'].get('events') or [])} events · "
                    f"{result['analysis'].get('filename', '—')}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load session: {exc}")

        # --- Save (prepare on demand — avoid re-encoding audio every rerun) ---
        if analysis is None:
            st.caption("Run or load an analysis first to save a session.")
            return

        include_audio = st.checkbox(
            "Include audio in session",
            value=True,
            help="Larger file, but enables spectrogram/playback after reload",
            key="session_include_audio",
        )
        if st.button("Prepare session pack", key="session_prepare", use_container_width=True):
            try:
                with st.spinner("Packing session…"):
                    st.session_state.session_zip_bytes = save_session_zip(
                        analysis,
                        selected_event_id=st.session_state.get("selected_event_id"),
                        library_path=st.session_state.get("library_path"),
                        include_audio=include_audio,
                    )
                    st.session_state.session_zip_stem = Path(
                        str(analysis.get("filename") or "track")
                    ).stem
                st.success("Session pack ready — download below.")
            except Exception as exc:
                st.session_state.session_zip_bytes = None
                st.error(f"Session save failed: {exc}")

        zip_bytes = st.session_state.get("session_zip_bytes")
        if zip_bytes:
            stem = st.session_state.get("session_zip_stem") or "track"
            st.download_button(
                "Download session (.aether.zip)",
                data=zip_bytes,
                file_name=f"aether_session_{stem}.aether.zip",
                mime="application/zip",
                use_container_width=True,
                key="session_download",
            )
            st.caption(f"Pack size ≈ {len(zip_bytes) / 1024:.0f} KB")


# ---------------------------------------------------------------------------
# Cached heavy operations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=8)
def cached_analyze_file(
    file_bytes: bytes,
    filename: str,
    settings_key: tuple,
) -> dict[str, Any]:
    """Cache full analysis keyed by file content + settings."""
    settings = AnalysisSettings(
        onset_threshold=settings_key[0],
        min_event_ms=settings_key[1],
        max_event_ms=settings_key[2],
        merge_gap_ms=settings_key[3],
        trim_silence=settings_key[4],
        hpss_margin=settings_key[5],
        onset_backend=settings_key[6] if len(settings_key) > 6 else "auto",
    )
    path = save_temp_upload(file_bytes, filename)
    try:
        return run_analysis(path, settings=settings, keep_audio=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@st.cache_data(show_spinner=False, max_entries=4)
def cached_analyze_url(url: str, settings_key: tuple) -> dict[str, Any]:
    settings = AnalysisSettings(
        onset_threshold=settings_key[0],
        min_event_ms=settings_key[1],
        max_event_ms=settings_key[2],
        merge_gap_ms=settings_key[3],
        trim_silence=settings_key[4],
        hpss_margin=settings_key[5],
        onset_backend=settings_key[6] if len(settings_key) > 6 else "auto",
    )
    return run_analysis(url, settings=settings, keep_audio=True)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_index_library(folder: str, settings_key: tuple) -> list[dict]:
    settings = AnalysisSettings(
        onset_threshold=settings_key[0],
        min_event_ms=settings_key[1],
        max_event_ms=settings_key[2],
        merge_gap_ms=settings_key[3],
        trim_silence=settings_key[4],
        hpss_margin=settings_key[5],
        onset_backend=settings_key[6] if len(settings_key) > 6 else "auto",
    )
    return index_library(folder, settings=settings)


# ---------------------------------------------------------------------------
# UI blocks
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="aether-header">
            <div class="aether-logo">◈</div>
            <div>
                <h1 class="aether-title">AETHER Analyzer</h1>
                <p class="aether-tagline">See every sound. Understand every layer. · Classical DSP only</p>
            </div>
            <div class="aether-badge">v{__version__} · no AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_bar(analysis: dict) -> None:
    dur = float(analysis.get("duration") or 0)
    mins = int(dur // 60)
    secs = dur % 60
    dur_s = f"{mins}:{secs:05.2f}" if mins else f"{secs:.2f}s"
    n_events = analysis.get("n_events", len(analysis.get("events") or []))
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">BPM</div>
                <div class="metric-value accent">{analysis.get('bpm', '—')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Key</div>
                <div class="metric-value">{analysis.get('key', '—')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{dur_s}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Events</div>
                <div class="metric-value accent">{n_events}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tid = str(analysis.get("track_id") or "")
    tid_short = f"{tid[:8]}…" if len(tid) > 8 else tid
    strongest = (analysis.get("resonance") or {}).get("strongest")
    extra = ""
    if strongest:
        extra = (
            f" · strongest resonance "
            f"**{strongest['a']}↔{strongest['b']}** "
            f"(`{strongest['score']}%`)"
        )
    st.caption(f"Track: **{analysis.get('filename', '—')}** · id `{tid_short}`{extra}")


def render_sound_class_pill(event: dict) -> None:
    sc = event.get("sound_class") or "Unknown"
    icon = event.get("sound_class_icon") or "❓"
    conf = event.get("sound_class_confidence")
    conf_s = f"{100 * conf:.0f}%" if conf is not None else "—"
    color = SOUND_CLASS_COLORS.get(sc, COLORS["muted"])
    st.markdown(
        f'<div class="class-pill" style="background:{color}22;color:{color};'
        f'border-color:{color}66;">{icon} {sc} · {conf_s}</div>',
        unsafe_allow_html=True,
    )
    reasons = event.get("sound_class_reasons") or []
    if reasons:
        st.caption("Rules: " + " · ".join(reasons))


def render_dna_section(event: dict) -> None:
    dna = event.get("sound_dna")
    if not isinstance(dna, dict):
        return
    st.markdown("#### ◈ Sound DNA")
    display = format_dna_display(dna)
    # First line is the code
    lines = display.split("\n", 1)
    code = lines[0]
    rest = lines[1] if len(lines) > 1 else ""
    st.markdown(
        f'<div class="dna-box"><div class="dna-code">{code}</div>{rest}</div>',
        unsafe_allow_html=True,
    )


def render_preset_section(event: dict) -> None:
    dump = event.get("preset_dump")
    if not isinstance(dump, dict):
        return
    text = dump.get("text") or ""
    st.markdown("#### Copy to Serum / Vital")
    st.caption("Rule-mapped wavetable preset — dial by ear, not a loadable .fxp")
    st.markdown(f'<div class="preset-box">{text}</div>', unsafe_allow_html=True)
    st.code(text, language="text")


def render_event_resonance(event: dict, analysis: dict) -> None:
    resonance = analysis.get("resonance") or {}
    partners = resonance_for_event(event["id"], resonance, top_k=5)
    if not partners:
        return
    st.markdown("#### Resonance partners")
    for p in partners:
        hot = p["score"] >= RESONANCE_HIGHLIGHT_THRESHOLD
        cls = "res-score hot" if hot else "res-score"
        icon = p.get("partner_icon") or ""
        st.markdown(
            f'<div class="res-pair"><span>{icon} **{p["partner"]}**'
            f' · {p.get("partner_class") or "—"}</span>'
            f'<span class="{cls}">{p["score"]:.1f}%</span></div>',
            unsafe_allow_html=True,
        )


def render_event_detail(event: dict, analysis: dict) -> None:
    st.markdown(f"### Event `{event['id']}`")
    render_sound_class_pill(event)

    etype = event.get("type") or "mixed"
    st.markdown(
        f'<span class="type-{etype}" style="font-weight:600;text-transform:uppercase;'
        f'font-size:0.8rem;">HPSS · {etype}</span>',
        unsafe_allow_html=True,
    )

    render_dna_section(event)

    chips = [
        ("Note", event.get("note") or "—"),
        ("Pitch", f"{event['pitch_hz']:.1f} Hz" if event.get("pitch_hz") else "—"),
        ("Start", f"{event['start_time']:.3f}s"),
        ("End", f"{event['end_time']:.3f}s"),
        ("Duration", f"{event['duration']:.3f}s"),
        ("Waveform", event.get("waveform_estimate") or "—"),
        ("Centroid", f"{event.get('spectral_centroid', 0):.0f} Hz"),
        ("Rolloff", f"{event.get('spectral_rolloff', 0):.0f} Hz"),
        ("Flatness", f"{event.get('spectral_flatness', 0):.4f}"),
        ("Flux", f"{event.get('spectral_flux', 0):.4f}"),
        ("Bandwidth", f"{event.get('spectral_bandwidth', 0):.0f} Hz"),
        ("Harmonic %", f"{100 * float(event.get('harmonic_ratio') or 0):.0f}%"),
        ("Attack", f"{event.get('attack_time', 0):.3f}s"),
        ("Decay", f"{event.get('decay_time', 0):.3f}s"),
        ("Reverb tail", f"{event.get('reverb_tail', 0):.3f}"),
        ("Distortion", f"{event.get('distortion', 0):.3f}"),
    ]

    html = '<div class="param-grid">'
    for k, v in chips:
        html += (
            f'<div class="param-chip"><div class="param-k">{k}</div>'
            f'<div class="param-v">{v}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    if event.get("mfcc_mean"):
        st.plotly_chart(
            mfcc_bar_figure(event["mfcc_mean"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    y = analysis.get("y")
    sr = int(analysis.get("sr") or 44100)
    if y is not None:
        seg = extract_event_audio(y, sr, event)
        st.audio(seg, sample_rate=sr)
        wav_bytes = export_event_wav_bytes(y, sr, event)
        st.download_button(
            "Download event WAV",
            data=wav_bytes,
            file_name=f"{event['id']}.wav",
            mime="audio/wav",
            use_container_width=True,
            key=f"dl_wav_{event['id']}",
        )

    render_event_resonance(event, analysis)
    render_preset_section(event)


def _forensics_light_json(result: dict) -> dict:
    skip = {"mel_mean", "mfcc_matrix", "y_plot", "pitch_contour_plot", "pitch_times"}
    return {
        "score_pct": result.get("score_pct"),
        "verdict": result.get("verdict"),
        "mode": result.get("mode"),
        "reliability": result.get("reliability"),
        "dimensions": result.get("dimensions"),
        "profile_a": {
            k: v for k, v in (result.get("profile_a") or {}).items() if k not in skip
        },
        "profile_b": {
            k: v for k, v in (result.get("profile_b") or {}).items() if k not in skip
        },
        "report": result.get("report"),
    }


def render_forensics_page(settings: AnalysisSettings) -> None:
    """
    Forensics v2: A/B or 1-vs-N, speech-ready, trim/normalize/reliability,
    spectrogram + pitch, written report.
    """
    st.markdown("## ◈ Forensics — sounds, voices & speech characteristics")
    st.caption(
        "Classical DSP only. **Characteristics ≠ personality.** "
        "Not court-grade speaker ID. Match scores, voice quality, and prosody are acoustic descriptions."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.radio(
            "Compare mode (A/B & lineup)",
            options=["voice", "sound"],
            format_func=lambda x: (
                "Voice / speech (tale)" if x == "voice" else "General sound / sample"
            ),
            horizontal=True,
            key="forensics_mode",
        )
    with c2:
        language = st.radio(
            "Report language",
            options=["da", "en"],
            format_func=lambda x: "Dansk" if x == "da" else "English",
            horizontal=True,
            key="forensics_lang",
        )
    with c3:
        workflow = st.radio(
            "Workflow",
            options=["character", "pair", "lineup"],
            format_func=lambda x: {
                "character": "Voice characteristics",
                "pair": "A vs B match",
                "lineup": "1 ref vs many",
            }[x],
            horizontal=True,
            key="forensics_workflow",
        )

    with st.expander("Tips + scope (tale / characteristics)", expanded=False):
        st.markdown(
            """
**Match (A vs B / lineup)**  
- Voice mode virker på almindelig tale — samme ord er **ikke** nødvendige  
- ≥ 3–5 sek tør tale; reliability falder på støj/korte klip  

**Voice characteristics**  
- **Voice quality** = hvordan instrumentet lyder (f0, klang, voicing, formant-proxies)  
- **Prosody / stemmeføring** = hvordan det bruges over tid (melodi, rate, pauser, dynamik)  
- **Ingen personlighed, ingen diagnose, ingen juridisk ID** — kun akustiske beskrivelser  
- Formanter og “talehastighed” er klassiske **proxies**, ikke lab-faciliteter  
            """
        )
    with st.expander("Explain: age / training / intervention / scores", expanded=False):
        st.markdown(render_voice_guide_markdown(language))
        st.markdown(render_dimension_help_markdown())

    # ----- Voice characteristics (single or dual card) -----
    if workflow == "character":
        st.markdown("#### Voice characteristics (quality + prosody)")
        st.caption(
            "Acoustic character card only — not personality. "
            "Optional second file for descriptive side-by-side comparison."
        )
        u1, u2 = st.columns(2)
        with u1:
            f_a = st.file_uploader(
                "Primary speech clip (A)",
                type=["mp3", "wav", "flac", "ogg", "m4a"],
                key="vchar_a",
            )
        with u2:
            f_b = st.file_uploader(
                "Optional clip B (compare characteristics)",
                type=["mp3", "wav", "flac", "ogg", "m4a"],
                key="vchar_b",
            )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            start_c = st.number_input("Start (s)", 0.0, 600.0, 0.0, 0.1, key="vchar_start")
        with col_s2:
            end_c = st.number_input("End (s, 0=full)", 0.0, 600.0, 0.0, 0.1, key="vchar_end")
        end_c_v = None if end_c <= 0 else float(end_c)

        if st.button("▶ Analyse characteristics", type="primary", use_container_width=True):
            if f_a is None:
                st.error("Upload at least primary clip A.")
            else:
                paths = []
                try:
                    with st.spinner("Extracting voice quality + prosody…"):
                        p_a = save_temp_upload(f_a.getvalue(), f_a.name)
                        paths.append(p_a)
                        char_a = analyze_voice_character_file(
                            p_a,
                            label=f_a.name,
                            settings=settings,
                            language=language,
                            start_sec=float(start_c),
                            end_sec=end_c_v,
                        )
                        st.session_state.voice_char_result = char_a
                        st.session_state.voice_char_compare = None
                        if f_b is not None:
                            p_b = save_temp_upload(f_b.getvalue(), f_b.name)
                            paths.append(p_b)
                            char_b = analyze_voice_character_file(
                                p_b,
                                label=f_b.name,
                                settings=settings,
                                language=language,
                                start_sec=float(start_c),
                                end_sec=end_c_v,
                            )
                            st.session_state.voice_char_compare = compare_voice_characters(
                                char_a, char_b, language=language
                            )
                            # keep both cards in compare payload
                            st.session_state.voice_char_result = char_a
                            st.session_state["_voice_char_b"] = char_b
                except Exception as exc:
                    st.error(f"Characteristics failed: {exc}")
                finally:
                    for p in paths:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

        char = st.session_state.get("voice_char_result")
        if not char:
            st.info(
                "Upload a speech clip (≥4–8s recommended). "
                "You’ll get a **voice quality** card + **prosody** card + written report."
            )
            return

        rel = char.get("reliability") or {}
        st.markdown(
            f"### `{char.get('label')}`  ·  Reliability `{rel.get('reliability_pct', '—')}%`"
        )
        st.caption(rel.get("label") or "")
        st.warning(char.get("disclaimer") or "Acoustic characteristics only — not personality.")

        vq = char.get("voice_quality") or {}
        pr = char.get("prosody") or {}
        cqa, cqb = st.columns(2)
        with cqa:
            st.markdown("##### 1. Voice quality")
            st.markdown(f"- **Pitch register:** {vq.get('pitch_register', '—')}")
            st.markdown(f"- **Voicing:** {vq.get('voicing', '—')}")
            st.markdown(f"- **Spectral colour:** {vq.get('spectral_colour', '—')}")
            m = char.get("measurements") or {}
            f = m.get("formants") or {}
            st.markdown(
                f"- **f0 mean:** {m.get('f0_mean_hz') or '—'} Hz  ·  "
                f"**range:** {m.get('f0_range_semitones', 0):.1f} st"
            )
            st.markdown(
                f"- **Formants (proxy):** F1={f.get('f1_hz') or '—'}  "
                f"F2={f.get('f2_hz') or '—'}  F3={f.get('f3_hz') or '—'} Hz"
            )
            st.markdown(
                f"- **HNR proxy:** {m.get('hnr_proxy_db') if m.get('hnr_proxy_db') is not None else '—'} dB"
            )
        with cqb:
            st.markdown("##### 2. Prosody / stemmeføring")
            st.markdown(f"- **Pitch variability:** {pr.get('pitch_variability', '—')}")
            st.markdown(f"- **Delivery rate:** {pr.get('delivery_rate', '—')}")
            st.markdown(f"- **Pausing:** {pr.get('pausing', '—')}")
            st.markdown(f"- **Dynamics:** {pr.get('dynamics', '—')}")
            st.markdown(
                f"- **Energy peaks/sec:** {m.get('energy_peaks_per_sec', 0)}  ·  "
                f"**pause frac:** {m.get('pause_fraction', 0):.2f}"
            )

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                voice_character_summary_bars(char),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with g2:
            st.plotly_chart(
                voice_character_pitch_figure(char),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        if char.get("y_plot") is not None:
            try:
                st.plotly_chart(
                    forensics_spectrogram_pair(
                        {
                            "label": char.get("label"),
                            "y_plot": char["y_plot"],
                            "sample_rate": char.get("sample_rate"),
                        },
                        {
                            "label": "—",
                            "y_plot": char["y_plot"][:1],
                            "sample_rate": char.get("sample_rate"),
                        },
                        sr=int(char.get("sample_rate") or 44100),
                    ),
                    use_container_width=True,
                )
            except Exception:
                pass

        st.markdown("#### Written report")
        st.code(char.get("report") or "", language="text")
        st.download_button(
            "Download characteristics report (.txt)",
            data=(char.get("report") or "").encode("utf-8"),
            file_name="aether_voice_characteristics.txt",
            mime="text/plain",
            key="dl_vchar_txt",
            use_container_width=True,
        )

        st.markdown("#### Pitch lane + audio-sync scrubber")
        if st.button("Run pitch-forensics on this clip", use_container_width=True, key="vchar_pitch"):
            try:
                y_pf = char.get("y_plot")
                sr_pf = int(char.get("sample_rate") or 44100)
                if y_pf is None:
                    st.error("No audio buffer on character result.")
                else:
                    with st.spinner("Building pitch lane + scrubber…"):
                        lab = str(char.get("label") or "clip")
                        st.session_state.pitch_forensics = build_pitch_forensics(
                            y_pf, sr_pf, label=lab
                        )
                        st.session_state.scrubber_payload = prepare_scrubber_payload(
                            y_pf, sr_pf, label=lab
                        )
            except Exception as exc:
                st.error(str(exc))
        if st.session_state.get("pitch_forensics"):
            render_pitch_forensics_block(st.session_state.pitch_forensics)

        with st.expander("What can change with age / training / tech?"):
            st.markdown(render_voice_guide_markdown(language))

        cmp = st.session_state.get("voice_char_compare")
        char_b = st.session_state.get("_voice_char_b")
        if cmp and char_b:
            st.markdown("---")
            st.markdown("#### Side-by-side characteristics (A vs B)")
            st.code(cmp.get("report") or "", language="text")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(f"**A — {char.get('label')}**")
                st.plotly_chart(
                    voice_character_pitch_figure(char),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with b2:
                st.markdown(f"**B — {char_b.get('label')}**")
                st.plotly_chart(
                    voice_character_pitch_figure(char_b),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            for bullet in cmp.get("bullets") or []:
                st.markdown(f"- {bullet}")
            st.download_button(
                "Download comparison report (.txt)",
                data=(cmp.get("report") or "").encode("utf-8"),
                file_name="aether_voice_char_compare.txt",
                mime="text/plain",
                key="dl_vchar_cmp",
                use_container_width=True,
            )
        return

    # ----- 1-vs-N lineup -----
    if workflow == "lineup":
        st.markdown("#### Reference vs candidates")
        ref_file = st.file_uploader(
            "Reference (known voice / sound)",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
            key="for_ref",
        )
        cand_files = st.file_uploader(
            "Candidates (multiple)",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            key="for_cands",
        )
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            start_ref = st.number_input("Ref start (s)", 0.0, 600.0, 0.0, 0.1, key="ref_start")
        with col_r2:
            end_ref = st.number_input(
                "Ref end (s, 0 = full)", 0.0, 600.0, 0.0, 0.1, key="ref_end"
            )
        end_ref_v = None if end_ref <= 0 else float(end_ref)

        if st.button("▶ Rank candidates", type="primary", use_container_width=True):
            if ref_file is None or not cand_files:
                st.error("Upload a reference and at least one candidate.")
            else:
                paths = []
                try:
                    with st.spinner("Comparing reference to all candidates…"):
                        p_ref = save_temp_upload(ref_file.getvalue(), ref_file.name)
                        paths.append(p_ref)
                        y_ref, sr, _ = load_audio(p_ref, settings=settings)
                        cands = []
                        for cf in cand_files:
                            p = save_temp_upload(cf.getvalue(), cf.name)
                            paths.append(p)
                            y_c, sr_c, _ = load_audio(p, settings=settings)
                            if sr_c != sr:
                                import librosa as _lr

                                y_c = _lr.resample(y_c, orig_sr=sr_c, target_sr=sr)
                            cands.append((cf.name, y_c))
                        ranking = compare_reference_vs_many(
                            y_ref,
                            cands,
                            sr,
                            mode=mode,
                            settings=settings,
                            language=language,
                            label_ref=ref_file.name,
                            start_ref=float(start_ref),
                            end_ref=end_ref_v,
                        )
                        st.session_state.forensics_ranking = ranking
                        if ranking.get("best") and ranking["best"].get("result"):
                            st.session_state.forensics_result = ranking["best"]["result"]
                except Exception as exc:
                    st.error(f"Line-up failed: {exc}")
                finally:
                    for p in paths:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

        ranking = st.session_state.get("forensics_ranking")
        if ranking and ranking.get("rankings"):
            st.success(
                f"Best match: **{ranking['best']['label']}** "
                f"at **{ranking['best']['score_pct']}%**"
                if ranking.get("best")
                else "Done"
            )
            st.plotly_chart(
                forensics_ranking_bars(ranking["rankings"]),
                use_container_width=True,
            )
            for i, r in enumerate(ranking["rankings"], 1):
                st.markdown(
                    f"**#{i} {r['label']}** — `{r['score_pct']}%` match · "
                    f"reliability `{r['reliability_pct']}%`  \n"
                    f"{r['verdict']}"
                )
        return

    # ----- A vs B -----
    source_mode = st.radio(
        "Sources",
        options=["upload", "events"],
        format_func=lambda x: "Upload two files" if x == "upload" else "Two events from track",
        horizontal=True,
        key="forensics_source",
    )

    y_a = y_b = None
    sr = 44100
    label_a, label_b = "A", "B"
    path_a = path_b = None
    dur_a = dur_b = 30.0

    if source_mode == "upload":
        u1, u2 = st.columns(2)
        with u1:
            f_a = st.file_uploader(
                "Sample A (e.g. speech clip)",
                type=["mp3", "wav", "flac", "ogg", "m4a"],
                key="for_a",
            )
        with u2:
            f_b = st.file_uploader(
                "Sample B",
                type=["mp3", "wav", "flac", "ogg", "m4a"],
                key="for_b",
            )
        if f_a is not None:
            path_a = save_temp_upload(f_a.getvalue(), f_a.name)
            label_a = f_a.name
            st.audio(f_a.getvalue())
            try:
                ya_tmp, sr_tmp, _ = load_audio(path_a, settings=settings)
                dur_a = len(ya_tmp) / sr_tmp
            except Exception:
                pass
        if f_b is not None:
            path_b = save_temp_upload(f_b.getvalue(), f_b.name)
            label_b = f_b.name
            st.audio(f_b.getvalue())
            try:
                yb_tmp, sr_tmp, _ = load_audio(path_b, settings=settings)
                dur_b = len(yb_tmp) / sr_tmp
            except Exception:
                pass
    else:
        analysis = st.session_state.get("analysis")
        if analysis is None or not analysis.get("events") or analysis.get("y") is None:
            st.warning(
                "Run a **track analysis** first (or load a session with audio), then pick two events."
            )
            return
        events = analysis["events"]
        labels = [
            f"{e.get('sound_class_icon', '')} {e['id']} · {e.get('sound_class', '?')} · {e.get('note') or '—'}"
            for e in events
        ]
        id_by_label = {lab: e["id"] for lab, e in zip(labels, events)}
        e1, e2 = st.columns(2)
        with e1:
            pick_a = st.selectbox("Event A", options=labels, key="for_ev_a")
        with e2:
            pick_b = st.selectbox(
                "Event B", options=labels, index=min(1, len(labels) - 1), key="for_ev_b"
            )
        sr = int(analysis.get("sr") or 44100)
        y = analysis["y"]
        ev_a = next(e for e in events if e["id"] == id_by_label[pick_a])
        ev_b = next(e for e in events if e["id"] == id_by_label[pick_b])
        y_a = extract_event_audio(y, sr, ev_a)
        y_b = extract_event_audio(y, sr, ev_b)
        label_a, label_b = ev_a["id"], ev_b["id"]
        dur_a = float(ev_a.get("duration") or len(y_a) / sr)
        dur_b = float(ev_b.get("duration") or len(y_b) / sr)
        st.audio(y_a, sample_rate=sr)
        st.audio(y_b, sample_rate=sr)

    st.markdown("#### Region + normalize")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        start_a = st.number_input("A start (s)", 0.0, float(max(dur_a, 0.1)), 0.0, 0.1, key="sa")
    with r2:
        end_a_in = st.number_input(
            "A end (s, 0=full)", 0.0, 600.0, 0.0, 0.1, key="ea"
        )
    with r3:
        start_b = st.number_input("B start (s)", 0.0, float(max(dur_b, 0.1)), 0.0, 0.1, key="sb")
    with r4:
        end_b_in = st.number_input(
            "B end (s, 0=full)", 0.0, 600.0, 0.0, 0.1, key="eb"
        )
    rms_norm = st.checkbox("RMS loudness normalize (recommended)", value=True, key="rms_n")
    hp_speech = st.checkbox(
        "Speech high-pass (~80 Hz) in voice mode", value=True, key="hp_sp"
    )

    end_a = None if end_a_in <= 0 else float(end_a_in)
    end_b = None if end_b_in <= 0 else float(end_b_in)

    run = st.button("▶ Run forensics comparison", type="primary", use_container_width=True)

    if run:
        try:
            with st.spinner("Preprocess + forensic profiles + scoring…"):
                if source_mode == "upload":
                    if not path_a or not path_b:
                        st.error("Upload **both** Sample A and Sample B.")
                        return
                    result = compare_files(
                        path_a,
                        path_b,
                        mode=mode,
                        settings=settings,
                        language=language,
                        start_a=float(start_a),
                        end_a=end_a,
                        start_b=float(start_b),
                        end_b=end_b,
                        rms_normalize=rms_norm,
                        highpass_speech=hp_speech,
                    )
                    result["profile_a"]["label"] = label_a
                    result["profile_b"]["label"] = label_b
                    result["report"] = write_analysis_report(result, language=language)
                else:
                    result = compare_audio_arrays(
                        y_a,
                        y_b,
                        sr,
                        label_a=label_a,
                        label_b=label_b,
                        mode=mode,
                        settings=settings,
                        language=language,
                        start_a=float(start_a),
                        end_a=end_a,
                        start_b=float(start_b),
                        end_b=end_b,
                        rms_normalize=rms_norm,
                        highpass_speech=hp_speech,
                    )
            st.session_state.forensics_result = result
        except Exception as exc:
            st.error(f"Forensics failed: {exc}")
            return
        finally:
            for p in (path_a, path_b):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    result = st.session_state.get("forensics_result")
    if not result:
        st.info(
            "Upload two speech/sound clips (or pick two events). "
            "You get **match %**, **reliability %**, spectrograms, pitch curves, and a written report."
        )
        return

    score = result["score_pct"]
    rel = result.get("reliability") or {}
    st.markdown(f"### Match: `{score:.1f}%`  ·  Reliability: `{rel.get('reliability_pct', '—')}%`")
    st.markdown(f"**{result['verdict']}**")
    st.caption(rel.get("label") or "")

    with st.expander("What does this score mean?", expanded=True):
        st.markdown(
            explain_score_snapshot(
                result.get("dimensions") or {},
                float(score),
                float(rel.get("reliability_pct") or 0),
            )
        )
        st.markdown(render_dimension_help_markdown())

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            forensics_score_gauge(score, title=f"{result.get('mode', 'sound').title()} match"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            forensics_reliability_bar(
                float(rel.get("reliability_pct") or 0),
                label="Reliability (clip length / quality)",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with g2:
        st.plotly_chart(
            forensics_dimension_bars(result.get("dimensions") or {}),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    if rel.get("reasons"):
        with st.expander("Reliability reasons", expanded=True):
            for r in rel["reasons"]:
                st.markdown(f"- {r}")

    st.markdown("#### Visual evidence")
    try:
        st.plotly_chart(
            forensics_spectrogram_pair(
                result.get("profile_a") or {},
                result.get("profile_b") or {},
                sr=int((result.get("profile_a") or {}).get("sample_rate") or 44100),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            forensics_pitch_overlay(
                result.get("profile_a") or {}, result.get("profile_b") or {}
            ),
            use_container_width=True,
        )
    except Exception as exc:
        st.caption(f"Plots unavailable: {exc}")

    st.markdown("#### Written analysis")
    st.code(result.get("report") or "", language="text")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download report (.txt)",
            data=(result.get("report") or "").encode("utf-8"),
            file_name="aether_forensics_report.txt",
            mime="text/plain",
            key="dl_forensics_txt",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download forensics JSON",
            data=json.dumps(_forensics_light_json(result), indent=2).encode("utf-8"),
            file_name="aether_forensics.json",
            mime="application/json",
            key="dl_forensics_json",
            use_container_width=True,
        )

    with st.expander("Dimension comments"):
        for key, dim in (result.get("dimensions") or {}).items():
            st.markdown(
                f"**{key}** — `{dim.get('pct', 0):.1f}%`  \n{dim.get('comment', '')}"
            )


def render_mixer_panel(analysis: dict) -> None:
    """Play / solo-style mute / export HPSS layers with optional event region."""
    st.markdown("#### Layer mixer (HPSS)")
    st.caption(
        "Not neural stems — **Harmonic / Percussive / Residual** from classical HPSS. "
        "**Event region** = only the time window of a detected event (A-00N)."
    )
    try:
        layers = build_layers(analysis)
    except Exception as exc:
        st.warning(f"Mixer unavailable: {exc}")
        return

    sr = int(analysis.get("sr") or 44100)
    dur = float(len(layers["full"]) / sr)
    events = analysis.get("events") or []

    # ----- Event region -----
    st.markdown("##### Event region")
    st.caption(
        "Full track = hele analysen. Ellers vælg et event — play/export/forensics "
        "bruger kun **start→end** for det event."
    )
    region_options = ["Full track"]
    region_map: dict[str, tuple[float, Optional[float]]] = {
        "Full track": (0.0, None),
    }
    for ev in events:
        lab = (
            f"{ev.get('sound_class_icon', '')} {ev['id']} · "
            f"{ev.get('sound_class', '?')} · "
            f"{ev['start_time']:.2f}–{ev['end_time']:.2f}s"
        )
        region_options.append(lab)
        region_map[lab] = (float(ev["start_time"]), float(ev["end_time"]))

    # Default to selected event if any
    default_idx = 0
    sel_id = st.session_state.get("selected_event_id")
    if sel_id:
        for i, lab in enumerate(region_options):
            if lab.startswith(" ") and sel_id in lab:
                default_idx = i
                break
            if sel_id in lab:
                default_idx = i
                break

    region_lab = st.selectbox(
        "Region source",
        options=region_options,
        index=min(default_idx, len(region_options) - 1),
        key="mix_region_sel",
    )
    reg_start, reg_end = region_map[region_lab]
    is_event_region = region_lab != "Full track"

    m1, m2, m3 = st.columns(3)
    with m1:
        start = st.number_input(
            "Region start (s)",
            0.0,
            float(max(dur, 0.1)),
            float(reg_start),
            0.01,
            key="mix_exp_s",
            disabled=is_event_region,
        )
    with m2:
        end_default = 0.0 if reg_end is None else float(reg_end)
        end = st.number_input(
            "Region end (s, 0=full track)",
            0.0,
            float(max(dur, 0.1)),
            end_default,
            0.01,
            key="mix_exp_e",
            disabled=is_event_region,
        )
    with m3:
        if is_event_region and reg_end is not None:
            st.metric("Event length", f"{reg_end - reg_start:.3f}s")
            start, end_v = float(reg_start), float(reg_end)
        else:
            start = float(start)
            end_v = None if end <= 0 else float(end)
            st.metric(
                "Region length",
                f"{(end_v or dur) - start:.3f}s" if end_v else f"{dur - start:.3f}s+",
            )

    if is_event_region and reg_end is not None:
        start, end_v = float(reg_start), float(reg_end)

    st.markdown("**Enable layers** (when mixing parts, Full is ignored to avoid double-count)")
    cols = st.columns(4)
    enabled = dict(st.session_state.get("mixer_enabled") or {})
    for i, key in enumerate(("full", "harmonic", "percussive", "residual")):
        with cols[i]:
            enabled[key] = st.checkbox(
                LAYER_LABELS[key].split("(")[0].strip(),
                value=enabled.get(key, key != "full" and key != "residual"),
                key=f"mix_en_{key}",
                help=LAYER_LABELS[key],
            )
    st.session_state.mixer_enabled = enabled

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        if st.button("Solo full", use_container_width=True):
            st.session_state.mixer_enabled = {
                "full": True, "harmonic": False, "percussive": False, "residual": False
            }
            st.rerun()
    with s2:
        if st.button("Solo harmonic", use_container_width=True):
            st.session_state.mixer_enabled = {
                "full": False, "harmonic": True, "percussive": False, "residual": False
            }
            st.rerun()
    with s3:
        if st.button("Solo percussive", use_container_width=True):
            st.session_state.mixer_enabled = {
                "full": False, "harmonic": False, "percussive": True, "residual": False
            }
            st.rerun()
    with s4:
        if st.button("Harm+Perc", use_container_width=True):
            st.session_state.mixer_enabled = {
                "full": False, "harmonic": True, "percussive": True, "residual": False
            }
            st.rerun()

    enabled = st.session_state.mixer_enabled
    use_full_only = bool(enabled.get("full")) and not any(
        enabled.get(k) for k in ("harmonic", "percussive", "residual")
    )
    mix_full = mix_layers(layers, enabled=enabled, use_full_only=use_full_only)
    clip = slice_layer(mix_full, sr, start, end_v)
    harm_clip = slice_layer(layers["harmonic"], sr, start, end_v)

    st.markdown(
        f"**Playback region:** `{start:.3f}s` → "
        f"`{(end_v if end_v is not None else dur):.3f}s`"
        + (f" · event `{region_lab.split('·')[0].strip()}`" if is_event_region else "")
    )
    st.audio(clip, sample_rate=sr)

    wav_b = mix_to_wav_bytes(clip, sr)
    region_tag = "event" if is_event_region else "region"
    st.download_button(
        "Download region mix WAV",
        data=wav_b,
        file_name=f"aether_{region_tag}_mix.wav",
        mime="audio/wav",
        use_container_width=True,
        key="dl_mix_wav",
    )
    st.download_button(
        "Download region **harmonic** WAV",
        data=mix_to_wav_bytes(harm_clip, sr),
        file_name=f"aether_{region_tag}_harmonic.wav",
        mime="audio/wav",
        use_container_width=True,
        key="dl_harm_wav",
    )

    st.markdown("##### Pitch forensics + audio-sync scrubber")
    st.caption(
        "Runs on the **selected region** only. Harmonic = closest classical "
        "'song/melody' isolation for pitch / autotune heuristics."
    )
    pf1, pf2, pf3 = st.columns(3)
    with pf1:
        if st.button("Forensics: **harmonic** region", use_container_width=True, type="primary"):
            _run_region_pitch_forensics(harm_clip, sr, label=f"harmonic@{start:.2f}s")
    with pf2:
        if st.button("Forensics: **mix** region", use_container_width=True):
            _run_region_pitch_forensics(clip, sr, label=f"mix@{start:.2f}s")
    with pf3:
        if st.button("Open scrubber only (mix)", use_container_width=True):
            try:
                with st.spinner("Preparing scrubber…"):
                    st.session_state.scrubber_payload = prepare_scrubber_payload(
                        clip, sr, label=f"mix@{start:.2f}s"
                    )
                    st.session_state.pitch_forensics = build_pitch_forensics(
                        clip, sr, label=f"mix@{start:.2f}s"
                    )
            except Exception as exc:
                st.error(str(exc))

    pf = st.session_state.get("pitch_forensics")
    if pf:
        render_pitch_forensics_block(pf)


def _run_region_pitch_forensics(y: np.ndarray, sr: int, label: str) -> None:
    try:
        with st.spinner("Pitch lane + scrubber + correction heuristics…"):
            st.session_state.pitch_forensics = build_pitch_forensics(y, sr, label=label)
            st.session_state.scrubber_payload = prepare_scrubber_payload(
                y, sr, label=label
            )
        st.success("Pitch forensics + scrubber ready below.")
    except Exception as exc:
        st.error(str(exc))


def render_pitch_forensics_block(pf: dict) -> None:
    corr = pf.get("correction") or {}
    st.markdown("#### Pitch lane + correction evidence")
    st.markdown(
        f"**{corr.get('correction_likelihood_pct', 0)}%** quantisation/correction likelihood — "
        f"*{corr.get('label', '')}*"
    )
    st.caption(corr.get("disclaimer") or "")

    # Audio-sync scrubber (primary interactive view)
    payload = st.session_state.get("scrubber_payload")
    if payload and payload.get("audio_b64"):
        st.markdown("##### Audio-sync scrubber")
        st.caption(
            "Playhead follows audio · click to seek · mouse wheel zooms time · "
            "horizontal lines = equal-tempered notes."
        )
        render_pitch_scrubber(payload, height=500, key="mix_scrubber")
    else:
        st.plotly_chart(
            pitch_lane_figure(pf),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

    with st.expander("Static Plotly pitch (extra zoom tools)", expanded=False):
        st.plotly_chart(
            pitch_lane_figure(pf),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

    with st.expander("Evidence list + metrics", expanded=True):
        for e in corr.get("evidence") or []:
            st.markdown(f"- {e}")
        st.json(corr.get("metrics") or {})
    st.download_button(
        "Download pitch-forensics JSON",
        data=json.dumps(
            {
                "label": pf.get("label"),
                "correction": corr,
                "note_grid_count": len(pf.get("note_grid") or []),
            },
            indent=2,
        ).encode("utf-8"),
        file_name="aether_pitch_forensics.json",
        mime="application/json",
        key="dl_pitch_for",
    )


def render_resonance_panel(analysis: dict, selected_id: Optional[str]) -> None:
    resonance = analysis.get("resonance")
    if not resonance:
        st.info("No resonance data.")
        return

    strongest = resonance.get("strongest")
    if strongest:
        st.success(
            f"Strongest pair: **{strongest['a']}** ↔ **{strongest['b']}** "
            f"at **{strongest['score']}%** "
            f"({strongest.get('a_class', '?')} · {strongest.get('b_class', '?')})"
        )

    st.plotly_chart(
        resonance_heatmap_figure(resonance, selected_id=selected_id),
        use_container_width=True,
    )

    st.markdown("##### Top resonant pairs")
    pairs = resonance.get("pairs") or []
    if not pairs:
        st.caption("No pairs computed.")
        return

    for p in pairs:
        hot = p["score"] >= RESONANCE_HIGHLIGHT_THRESHOLD
        badge = "🔥 " if hot else ""
        st.markdown(
            f"{badge}**{p['a']}** ({p.get('a_icon', '')} {p.get('a_class', '—')}) "
            f"↔ **{p['b']}** ({p.get('b_icon', '')} {p.get('b_class', '—')}) "
            f"— `{p['score']}%`  \n"
            f"<span style='color:{COLORS['muted']};font-size:0.8rem'>"
            f"pitch {p.get('pitch', 0):.0f} · spectral {p.get('spectral', 0):.0f} · "
            f"envelope {p.get('envelope', 0):.0f} · timing {p.get('timing', 0):.0f}"
            f"</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    render_header()

    # ----- Sidebar -----
    with st.sidebar:
        app_mode = st.radio(
            "Workspace",
            options=["track", "forensics"],
            format_func=lambda x: "🎚 Track analyzer" if x == "track" else "🔍 Forensics compare",
            horizontal=False,
            key="app_mode_radio",
        )
        st.session_state.app_mode = app_mode
        st.markdown("---")

        st.markdown("### Analyze New Track")
        uploaded = st.file_uploader(
            "Drop MP3 / WAV / FLAC",
            type=["mp3", "wav", "flac", "ogg", "aiff", "aif", "m4a"],
            help="Upload a track for classical DSP analysis",
        )
        url = st.text_input(
            "Or paste URL (YouTube supported)",
            placeholder="https://www.youtube.com/watch?v=…",
        )

        with st.expander("Advanced settings", expanded=False):
            onset_threshold = st.slider("Onset threshold", 0.01, 0.3, 0.07, 0.01)
            min_event_ms = st.slider("Min event length (ms)", 40, 500, 80, 10)
            max_event_ms = st.slider("Max event length (ms)", 500, 8000, 4000, 100)
            merge_gap_ms = st.slider("Merge gap (ms)", 10, 200, 50, 5)
            trim_silence = st.checkbox("Trim silence", value=True)
            hpss_margin = st.slider("HPSS margin", 0.5, 3.0, 1.0, 0.1)
            onset_backend = st.selectbox(
                "Onset backend",
                options=["auto", "librosa", "aubio"],
                index=0,
                help="auto = aubio if installed, else librosa",
            )
            use_dtw = st.checkbox("Similarity: use DTW blend", value=False)
            color_mode = st.radio(
                "Timeline colours",
                options=["sound_class", "hpss"],
                format_func=lambda x: "Sound class" if x == "sound_class" else "HPSS type",
                horizontal=True,
            )
            st.session_state.timeline_color_mode = color_mode

        settings = _settings_from_ui(
            onset_threshold,
            min_event_ms,
            max_event_ms,
            merge_gap_ms,
            trim_silence,
            hpss_margin,
            onset_backend,
        )
        settings_key = (
            settings.onset_threshold,
            settings.min_event_ms,
            settings.max_event_ms,
            settings.merge_gap_ms,
            settings.trim_silence,
            settings.hpss_margin,
            settings.onset_backend,
        )

        run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### Sample library")
        lib_path = st.text_input(
            "Folder path to samples",
            value=st.session_state.library_path or "",
            placeholder=r"C:\Samples\Kicks",
            help="Local folder of WAV/MP3 samples for similarity search",
        )
        index_btn = st.button("Analyze my library", use_container_width=True)

        if st.session_state.library is not None:
            st.success(f"Library indexed: **{len(st.session_state.library)}** samples")

        st.markdown("---")
        # Session uses current analysis if any (may be None before first run)
        render_session_panel(st.session_state.get("analysis"))

        st.markdown("---")
        render_environment_panel()

        st.markdown("---")
        st.caption(
            "Limitations: HPSS ≠ neural stems · "
            "Event detection imperfect on dense mixes · "
            "Classification & presets are rule-based heuristics · "
            "Similarity / resonance / forensics are feature-based, not AI. · "
            "Install help: SETUP.md"
        )

    # ----- Forensics workspace (independent of full track analysis) -----
    if st.session_state.get("app_mode") == "forensics":
        # settings may only exist if sidebar ran advanced block — rebuild defaults
        try:
            _settings = settings
        except NameError:
            _settings = AnalysisSettings()
        render_forensics_page(_settings)
        return

    # ----- Run analysis -----
    if run_btn:
        st.session_state.matches = None
        st.session_state.selected_event_id = None
        try:
            with st.spinner("Running classical DSP pipeline…"):
                progress = st.progress(0.0, text="Starting…")
                analysis = None

                if uploaded is not None:
                    progress.progress(0.1, text="Analyzing uploaded file…")
                    analysis = cached_analyze_file(
                        uploaded.getvalue(), uploaded.name, settings_key
                    )
                    progress.progress(1.0, text="Done")
                elif url and url.strip():
                    if not is_url(url.strip()):
                        st.error("Please enter a valid http(s) URL.")
                    else:
                        progress.progress(0.05, text="Downloading…")
                        analysis = cached_analyze_url(url.strip(), settings_key)
                        progress.progress(1.0, text="Done")
                else:
                    st.warning("Upload a file or paste a URL first.")

                if analysis is not None:
                    st.session_state.analysis = analysis
                    if analysis.get("events"):
                        st.session_state.selected_event_id = analysis["events"][0]["id"]
                    st.session_state.last_error = None
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Analysis failed: {exc}")

    if index_btn:
        if not lib_path or not Path(lib_path).is_dir():
            st.sidebar.error("Enter a valid local folder path.")
        else:
            try:
                with st.spinner("Indexing sample library (MFCC + spectral)…"):
                    lib = cached_index_library(lib_path, settings_key)
                st.session_state.library = lib
                st.session_state.library_path = lib_path
                st.sidebar.success(f"Indexed {len(lib)} samples.")
            except Exception as exc:
                st.sidebar.error(f"Library index failed: {exc}")

    analysis = st.session_state.analysis
    if analysis is None:
        st.info(
            "Upload a track (or paste a YouTube URL) and click **Run analysis** to "
            "dissect it into individual sound events with full classical DSP parameters."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**① Detect**\n\nHPSS + onsets → A-001…")
        with c2:
            st.markdown("**② DNA + class**\n\nFingerprint & Kick/Lead/Pad…")
        with c3:
            st.markdown("**③ Preset dump**\n\nSerum/Vital-style params")
        with c4:
            st.markdown("**④ Resonance**\n\nWhich events lock together")
        return

    # ----- Main layout -----
    render_global_bar(analysis)

    main_col, side_col = st.columns([2.2, 1], gap="large")

    with main_col:
        selected_id = st.session_state.selected_event_id
        events = analysis.get("events") or []

        ids = [e["id"] for e in events]
        labels = [
            f"{e.get('sound_class_icon', '')} {e['id']}  ·  "
            f"{e.get('sound_class', '?'):8s}  ·  "
            f"{e.get('note') or '—':5s}  ·  {e['start_time']:.2f}s"
            for e in events
        ]
        id_to_label = dict(zip(ids, labels))
        label_to_id = dict(zip(labels, ids))

        current_label = id_to_label.get(selected_id, labels[0] if labels else None)
        chosen = st.selectbox(
            "Select event",
            options=labels,
            index=labels.index(current_label) if current_label in labels else 0,
            label_visibility="collapsed",
        )
        if chosen:
            st.session_state.selected_event_id = label_to_id[chosen]
            selected_id = st.session_state.selected_event_id

        tab_tl, tab_spec, tab_hpss, tab_mix, tab_res = st.tabs(
            ["Timeline", "Spectrogram", "HPSS", "Mixer", "Resonance Map"]
        )

        color_mode = st.session_state.get("timeline_color_mode", "sound_class")

        with tab_tl:
            # Full-track playback
            y_full = analysis.get("y")
            sr_full = int(analysis.get("sr") or 44100)
            if y_full is not None:
                st.caption("Full track (analysis mono)")
                st.audio(y_full, sample_rate=sr_full)
            fig = timeline_figure(
                analysis, selected_id=selected_id, color_mode=color_mode
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        with tab_spec:
            y = analysis.get("y")
            sr = int(analysis.get("sr") or 44100)
            if y is not None:
                fig_s = spectrogram_figure(
                    y,
                    sr,
                    events=events,
                    selected_id=selected_id,
                    hop_length=settings.hop_length,
                    n_fft=settings.n_fft,
                )
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.warning("Audio buffer not available for spectrogram.")

        with tab_hpss:
            yh = analysis.get("y_harmonic")
            yp = analysis.get("y_percussive")
            sr = int(analysis.get("sr") or 44100)
            if yh is not None and yp is not None:
                st.plotly_chart(hpss_figure(yh, yp, sr), use_container_width=True)
            else:
                st.warning("HPSS buffers not available.")

        with tab_mix:
            render_mixer_panel(analysis)

        with tab_res:
            render_resonance_panel(analysis, selected_id)

        # Export row
        st.markdown("#### Export")
        ex1, ex2, ex3 = st.columns(3)
        y = analysis.get("y")
        sr = int(analysis.get("sr") or 44100)
        stem = Path(str(analysis.get("filename") or "track")).stem

        with ex1:
            st.download_button(
                "JSON analysis",
                data=export_json_bytes(analysis),
                file_name=f"aether_{stem}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )
        with ex2:
            try:
                midi_data = export_midi_bytes(analysis)
                st.download_button(
                    "MIDI (notes)",
                    data=midi_data,
                    file_name=f"aether_{stem}.mid",
                    mime="audio/midi",
                    use_container_width=True,
                    key="dl_midi",
                )
            except Exception as exc:
                st.caption(f"MIDI unavailable: {exc}")
        with ex3:
            if y is not None:
                zip_bytes = export_all_events_zip(y, sr, analysis)
                st.download_button(
                    "All events (ZIP)",
                    data=zip_bytes,
                    file_name=f"aether_events_{stem}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip",
                )

    with side_col:
        event = _get_event(analysis, selected_id) if selected_id else None
        if event:
            render_event_detail(event, analysis)

            st.markdown("---")
            st.markdown("#### Find similar sounds")
            if st.session_state.library is None:
                st.caption("Index a sample folder in the sidebar first.")
            else:
                if st.button("Search library", type="primary", use_container_width=True):
                    y = analysis.get("y")
                    sr = int(analysis.get("sr") or 44100)
                    event_audio = (
                        extract_event_audio(y, sr, event) if y is not None else None
                    )
                    with st.spinner("Computing cosine / DTW scores…"):
                        matches = find_similar(
                            event,
                            st.session_state.library,
                            top_k=SIMILARITY_TOP_K,
                            use_dtw=use_dtw,
                            event_audio=event_audio,
                            sr=sr,
                            settings=settings,
                        )
                    st.session_state.matches = matches
                    if matches:
                        event["similarity_score_to_library"] = matches[0]["score"]

                matches = st.session_state.matches
                if matches:
                    for m in matches:
                        score_pct = f"{100 * m['score']:.1f}%"
                        st.markdown(
                            f"**{m['filename']}** · `{score_pct}`  \n"
                            f"<span style='color:{COLORS['muted']};font-size:0.8rem'>"
                            f"centroid {m.get('spectral_centroid', 0):.0f} Hz · "
                            f"{m.get('duration', 0):.2f}s</span>",
                            unsafe_allow_html=True,
                        )
                        try:
                            if m.get("path") and Path(m["path"]).is_file():
                                st.audio(m["path"])
                        except Exception:
                            pass
        else:
            st.info("Select an event from the list to inspect parameters.")


if __name__ == "__main__":
    main()
