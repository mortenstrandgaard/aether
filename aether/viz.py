"""
Plotly visualizations: interactive timeline, spectrogram, HPSS energy,
MFCC bars, and resonance heatmap.

All figures use the shared dark COLORS palette for a studio-like look.
"""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np
import plotly.graph_objects as go

from aether.config import COLORS, SOUND_CLASS_COLORS


TYPE_COLORS = {
    "harmonic": COLORS["harmonic"],
    "percussive": COLORS["percussive"],
    "mixed": COLORS["mixed"],
}


def _event_color(ev: dict[str, Any], color_mode: str = "sound_class") -> str:
    """Pick bar color: sound_class (default) or HPSS type."""
    if color_mode == "hpss":
        return TYPE_COLORS.get(ev.get("type") or "mixed", COLORS["mixed"])
    sc = ev.get("sound_class") or "Unknown"
    return SOUND_CLASS_COLORS.get(sc, COLORS["muted"])


def timeline_figure(
    analysis: dict[str, Any],
    selected_id: Optional[str] = None,
    max_duration_display: Optional[float] = None,
    color_mode: str = "sound_class",
) -> go.Figure:
    """
    Interactive event timeline with color-coded bars and hover info.

    color_mode: "sound_class" (Kick/Lead/…) or "hpss" (harmonic/percussive/mixed)
    """
    events = analysis.get("events") or []
    duration = float(analysis.get("duration") or 0.0)
    if max_duration_display is not None:
        duration = min(duration, max_duration_display)

    fig = go.Figure()

    # Background energy curve for context
    rms_t = analysis.get("rms_times") or []
    rms = analysis.get("rms") or []
    if rms_t and rms:
        rms_arr = np.asarray(rms, dtype=float)
        rms_n = rms_arr / rms_arr.max() if rms_arr.max() > 0 else rms_arr
        fig.add_trace(
            go.Scatter(
                x=rms_t,
                y=rms_n * 0.35 + 0.05,
                mode="lines",
                line=dict(color="rgba(138,138,154,0.35)", width=1),
                name="Energy",
                hoverinfo="skip",
            )
        )

    # Lane by sound class (hash) or HPSS type
    def y_lane(ev: dict) -> float:
        if color_mode == "hpss":
            etype = ev.get("type") or "mixed"
            return 0.55 if etype == "harmonic" else (0.35 if etype == "percussive" else 0.45)
        # Spread sound classes across vertical lanes for readability
        classes = list(SOUND_CLASS_COLORS.keys())
        sc = ev.get("sound_class") or "Unknown"
        idx = classes.index(sc) if sc in classes else len(classes) - 1
        return 0.28 + (idx % 7) * 0.08

    for ev in events:
        start = float(ev["start_time"])
        end = float(ev["end_time"])
        if start > duration:
            continue
        color = _event_color(ev, color_mode=color_mode)
        selected = selected_id is not None and ev["id"] == selected_id
        y_base = y_lane(ev)
        height = 0.14 if selected else 0.10
        opacity = 1.0 if selected else 0.82

        note = ev.get("note") or "—"
        pitch = ev.get("pitch_hz")
        pitch_s = f"{pitch:.1f} Hz" if pitch else "—"
        icon = ev.get("sound_class_icon") or ""
        sclass = ev.get("sound_class") or "—"
        conf = ev.get("sound_class_confidence")
        conf_s = f"{100 * conf:.0f}%" if conf is not None else "—"
        dna_code = ""
        if isinstance(ev.get("sound_dna"), dict):
            dna_code = ev["sound_dna"].get("code") or ""

        hover = (
            f"<b>{icon} {ev['id']}</b> · {sclass} ({conf_s})<br>"
            f"HPSS: {ev.get('type', '—')}<br>"
            f"Note: {note} ({pitch_s})<br>"
            f"Time: {start:.2f}s – {end:.2f}s<br>"
            f"Duration: {ev.get('duration', end - start):.3f}s<br>"
            f"Waveform: {ev.get('waveform_estimate', '—')}<br>"
            f"Centroid: {ev.get('spectral_centroid', '—')}<br>"
            f"<span style='font-size:10px'>{dna_code}</span>"
        )

        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[y_base, y_base, y_base + height, y_base + height, y_base],
                fill="toself",
                fillcolor=color,
                line=dict(color="#ffffff" if selected else color, width=2 if selected else 0.5),
                opacity=opacity,
                mode="lines",
                name=ev["id"],
                text=hover,
                hoverinfo="text",
                customdata=[ev["id"]],
                showlegend=False,
            )
        )

        label = f"{icon}{ev['id']}" if icon else ev["id"]
        if selected or (end - start) > 0.35:
            fig.add_trace(
                go.Scatter(
                    x=[(start + end) / 2],
                    y=[y_base + height / 2],
                    mode="text",
                    text=[label],
                    textfont=dict(color=COLORS["text"], size=9 if selected else 8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Legend: unique sound classes present (or HPSS)
    if color_mode == "hpss":
        legend_items = [
            ("Harmonic", COLORS["harmonic"]),
            ("Percussive", COLORS["percussive"]),
            ("Mixed", COLORS["mixed"]),
        ]
    else:
        present = []
        seen = set()
        for ev in events:
            sc = ev.get("sound_class") or "Unknown"
            if sc not in seen:
                seen.add(sc)
                present.append((sc, SOUND_CLASS_COLORS.get(sc, COLORS["muted"])))
        legend_items = present[:10]

    for label, col in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=col),
                name=label,
            )
        )

    fig.update_layout(
        title=dict(
            text="Event Timeline"
            + (" · by sound class" if color_mode == "sound_class" else " · by HPSS"),
            font=dict(color=COLORS["text"], size=14),
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"], family="IBM Plex Sans, system-ui, sans-serif"),
        xaxis=dict(
            title="Time (s)",
            range=[0, max(duration, 0.1)],
            gridcolor=COLORS["border"],
            zeroline=False,
            color=COLORS["muted"],
        ),
        yaxis=dict(visible=False, range=[0.2, 1.0], fixedrange=True),
        margin=dict(l=20, r=20, t=40, b=40),
        height=300,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        hovermode="closest",
    )
    return fig


def spectrogram_figure(
    y: np.ndarray,
    sr: int,
    events: Optional[list[dict]] = None,
    selected_id: Optional[str] = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_sec: float = 60.0,
) -> go.Figure:
    """Log-power spectrogram with optional selected-event markers."""
    if len(y) > int(max_sec * sr):
        y = y[: int(max_sec * sr)]

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    f_max_idx = int(np.searchsorted(freqs, 8000))
    S_plot = S_db[:f_max_idx, :]
    freqs_plot = freqs[:f_max_idx]

    fig = go.Figure(
        data=go.Heatmap(
            z=S_plot,
            x=times,
            y=freqs_plot,
            colorscale="Magma",
            zsmooth="best",
            colorbar=dict(title="dB", len=0.8),
            hovertemplate="t=%{x:.2f}s<br>f=%{y:.0f}Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )

    if events and selected_id:
        for ev in events:
            if ev["id"] != selected_id:
                continue
            fig.add_vline(
                x=float(ev["start_time"]),
                line=dict(color=COLORS["accent2"], width=1, dash="dot"),
            )
            fig.add_vline(
                x=float(ev["end_time"]),
                line=dict(color=COLORS["accent"], width=1, dash="dot"),
            )

    fig.update_layout(
        title=dict(text="Spectrogram", font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title="Time (s)", gridcolor=COLORS["border"], color=COLORS["muted"]),
        yaxis=dict(title="Hz", gridcolor=COLORS["border"], color=COLORS["muted"]),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
    )
    return fig


def hpss_figure(
    y_harm: np.ndarray,
    y_perc: np.ndarray,
    sr: int,
    hop_length: int = 512,
    max_sec: float = 60.0,
) -> go.Figure:
    """RMS envelopes of harmonic vs percussive components."""
    n = int(max_sec * sr)
    yh = y_harm[:n] if len(y_harm) > n else y_harm
    yp = y_perc[:n] if len(y_perc) > n else y_perc

    rms_h = librosa.feature.rms(y=yh, hop_length=hop_length)[0]
    rms_p = librosa.feature.rms(y=yp, hop_length=hop_length)[0]
    t = librosa.frames_to_time(np.arange(len(rms_h)), sr=sr, hop_length=hop_length)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=rms_h,
            mode="lines",
            name="Harmonic",
            line=dict(color=COLORS["harmonic"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=rms_p,
            mode="lines",
            name="Percussive",
            line=dict(color=COLORS["percussive"], width=1.5),
        )
    )
    fig.update_layout(
        title=dict(text="HPSS Energy", font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title="Time (s)", gridcolor=COLORS["border"], color=COLORS["muted"]),
        yaxis=dict(title="RMS", gridcolor=COLORS["border"], color=COLORS["muted"]),
        margin=dict(l=50, r=20, t=40, b=40),
        height=240,
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def mfcc_bar_figure(mfcc_mean: list[float]) -> go.Figure:
    """Compact bar chart of mean MFCC coefficients for the detail panel."""
    fig = go.Figure(
        data=go.Bar(
            x=[f"M{i}" for i in range(len(mfcc_mean))],
            y=mfcc_mean,
            marker_color=COLORS["accent"],
        )
    )
    fig.update_layout(
        title=dict(text="MFCC Mean", font=dict(size=12, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"], size=10),
        margin=dict(l=30, r=10, t=30, b=30),
        height=180,
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    return fig


def resonance_heatmap_figure(
    resonance: dict[str, Any],
    selected_id: Optional[str] = None,
    max_events: int = 40,
) -> go.Figure:
    """
    NxN resonance heatmap (0–100%). Caps display size for readability on long tracks.
    """
    ids = list(resonance.get("ids") or [])
    matrix = np.asarray(resonance.get("matrix") or [], dtype=float)

    if len(ids) == 0 or matrix.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Resonance Map (no events)",
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface2"],
            height=200,
            font=dict(color=COLORS["text"]),
        )
        return fig

    if len(ids) > max_events:
        ids = ids[:max_events]
        matrix = matrix[:max_events, :max_events]

    # Highlight selected row/col with a slight alpha boost via customdata only in hover
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=ids,
            y=ids,
            colorscale=[
                [0.0, "#141418"],
                [0.35, "#2a2050"],
                [0.55, "#7c5cfc"],
                [0.75, "#00d4aa"],
                [1.0, "#e8fff7"],
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="%", len=0.8),
            hovertemplate="%{y} ↔ %{x}<br>Resonance: %{z:.1f}%<extra></extra>",
        )
    )

    # Mark selected event with a hollow box (shapes)
    if selected_id and selected_id in ids:
        i = ids.index(selected_id)
        # light grid lines around the row/col via annotations
        fig.add_hline(y=i, line=dict(color=COLORS["accent2"], width=1, dash="dot"))
        fig.add_vline(x=i, line=dict(color=COLORS["accent2"], width=1, dash="dot"))

    fig.update_layout(
        title=dict(text="Resonance Map", font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(side="top", tickangle=-45, color=COLORS["muted"], dtick=1),
        yaxis=dict(autorange="reversed", color=COLORS["muted"], dtick=1),
        margin=dict(l=50, r=20, t=60, b=20),
        height=min(520, 160 + 18 * len(ids)),
    )
    return fig
