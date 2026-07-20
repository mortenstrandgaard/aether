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
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

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
from aether.loader import is_url, save_temp_upload
from aether.pipeline import run_analysis
from aether.resonance import resonance_for_event
from aether.session_io import load_session_zip, save_session_zip
from aether.similarity import find_similar, index_library
from aether.sound_dna import format_dna_display
from aether.viz import (
    hpss_figure,
    mfcc_bar_figure,
    resonance_heatmap_figure,
    spectrogram_figure,
    timeline_figure,
)

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

    html, body, [class*="css"] {{
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

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace;
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
            "Similarity / resonance are feature-based, not AI. · "
            "Install help: SETUP.md"
        )

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

        tab_tl, tab_spec, tab_hpss, tab_res = st.tabs(
            ["Timeline", "Spectrogram", "HPSS", "Resonance Map"]
        )

        color_mode = st.session_state.get("timeline_color_mode", "sound_class")

        with tab_tl:
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
