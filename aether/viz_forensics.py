"""Plotly helpers for forensics comparison UI."""

from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aether.config import COLORS


def forensics_score_gauge(score_pct: float, title: str = "Match") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={"suffix": "%", "font": {"size": 42, "color": COLORS["text"]}},
            title={"text": title, "font": {"size": 14, "color": COLORS["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar": {"color": COLORS["accent2"]},
                "bgcolor": COLORS["surface2"],
                "borderwidth": 1,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 40], "color": "#2a1520"},
                    {"range": [40, 55], "color": "#2a2418"},
                    {"range": [55, 70], "color": "#1a2430"},
                    {"range": [70, 85], "color": "#152a28"},
                    {"range": [85, 100], "color": "#1a2a22"},
                ],
                "threshold": {
                    "line": {"color": COLORS["accent"], "width": 3},
                    "thickness": 0.75,
                    "value": score_pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        font={"color": COLORS["text"]},
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def forensics_reliability_bar(reliability_pct: float, label: str = "") -> go.Figure:
    color = (
        COLORS["accent2"]
        if reliability_pct >= 70
        else (COLORS["warning"] if reliability_pct >= 45 else COLORS["percussive"])
    )
    fig = go.Figure(
        go.Bar(
            x=[reliability_pct],
            y=["Reliability"],
            orientation="h",
            marker_color=color,
            text=[f"{reliability_pct:.0f}%"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(
            text=label or "Test reliability (clip quality)",
            font=dict(size=12, color=COLORS["text"]),
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"], size=11),
        xaxis=dict(range=[0, 100], gridcolor=COLORS["border"]),
        height=120,
        margin=dict(l=10, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def forensics_dimension_bars(dimensions: dict[str, Any]) -> go.Figure:
    labels = {
        "timbre_mfcc": "Timbre",
        "pitch": "Pitch",
        "spectral": "Spectral",
        "envelope": "Envelope",
        "energy_dynamics": "Energy",
        "temporal_align": "DTW align",
    }
    keys = [k for k in labels if k in dimensions]
    xs = [labels[k] for k in keys]
    ys = [dimensions[k].get("pct", 0) for k in keys]
    colors = [
        COLORS["accent2"] if y >= 70 else (COLORS["accent"] if y >= 55 else COLORS["percussive"])
        for y in ys
    ]
    fig = go.Figure(
        go.Bar(
            x=ys,
            y=xs,
            orientation="h",
            marker_color=colors,
            text=[f"{y:.0f}%" for y in ys],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="Dimension scores", font=dict(size=13, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"], size=11),
        xaxis=dict(range=[0, 100], gridcolor=COLORS["border"], title="%"),
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=10, r=20, t=40, b=30),
    )
    return fig


def forensics_pitch_overlay(profile_a: dict, profile_b: dict) -> go.Figure:
    """Pitch contours A vs B over normalized time."""
    fig = go.Figure()
    for prof, name, color in [
        (profile_a, profile_a.get("label") or "A", COLORS["accent"]),
        (profile_b, profile_b.get("label") or "B", COLORS["accent2"]),
    ]:
        times = prof.get("pitch_times") or []
        vals = prof.get("pitch_contour_plot") or []
        if not times or not vals:
            continue
        # filter Nones
        xs, ys = [], []
        for t, v in zip(times, vals):
            if v is not None and v > 0:
                xs.append(t)
                ys.append(v)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", name=str(name)[:24],
                    line=dict(color=color, width=1.5),
                )
            )
    fig.update_layout(
        title=dict(text="Pitch contour (speech intonation)", font=dict(size=13, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title="Time (s)", gridcolor=COLORS["border"], color=COLORS["muted"]),
        yaxis=dict(title="Hz", gridcolor=COLORS["border"], color=COLORS["muted"]),
        height=260,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def forensics_spectrogram_pair(
    profile_a: dict,
    profile_b: dict,
    sr: int = 44100,
    max_sec: float = 8.0,
) -> go.Figure:
    """Side-by-side log spectrograms from stored y_plot buffers."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            str(profile_a.get("label") or "A")[:40],
            str(profile_b.get("label") or "B")[:40],
        ),
    )

    def add_spec(prof: dict, col: int) -> None:
        y = prof.get("y_plot")
        if y is None or len(y) < 64:
            return
        y = np.asarray(y, dtype=float)
        nmax = int(max_sec * sr)
        if len(y) > nmax:
            y = y[:nmax]
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=256)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        f_idx = int(np.searchsorted(freqs, 5000))
        fig.add_trace(
            go.Heatmap(
                z=S_db[:f_idx, :],
                x=times,
                y=freqs[:f_idx],
                colorscale="Magma",
                showscale=(col == 2),
                zsmooth="best",
            ),
            row=1, col=col,
        )

    add_spec(profile_a, 1)
    add_spec(profile_b, 2)
    fig.update_layout(
        title=dict(text="Spectrogram A | B", font=dict(size=13, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], size=10),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="s", gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="Hz", gridcolor=COLORS["border"])
    return fig


def forensics_ranking_bars(rankings: list[dict]) -> go.Figure:
    labels = [r["label"][:28] for r in rankings]
    scores = [r["score_pct"] for r in rankings]
    colors = [
        COLORS["accent2"] if s >= 70 else (COLORS["accent"] if s >= 55 else COLORS["muted"])
        for s in scores
    ]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f}%" for s in scores],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="Reference vs candidates (ranked)", font=dict(size=13, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(range=[0, 100], title="Match %", gridcolor=COLORS["border"]),
        yaxis=dict(autorange="reversed"),
        height=max(220, 40 * len(labels) + 80),
        margin=dict(l=10, r=20, t=40, b=30),
    )
    return fig
