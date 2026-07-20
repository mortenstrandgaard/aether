"""Plotly helpers for forensics comparison UI."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

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
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
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
