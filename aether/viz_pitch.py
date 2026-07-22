"""Plotly pitch-lane with horizontal note guides (zoomable)."""

from __future__ import annotations

from typing import Any, Optional

import plotly.graph_objects as go

from aether.config import COLORS


def pitch_lane_figure(
    pitch_forensics: dict[str, Any],
    *,
    title: Optional[str] = None,
    show_grid: bool = True,
) -> go.Figure:
    lane = pitch_forensics.get("lane") or {}
    grid = pitch_forensics.get("note_grid") or []
    times = lane.get("times") or []
    f0 = lane.get("f0") or []

    fig = go.Figure()

    # Note guide lines (horizontal)
    if show_grid and grid:
        for g in grid:
            fig.add_hline(
                y=g["hz"],
                line=dict(
                    color="rgba(124,92,252,0.25)" if g["midi"] % 12 != 0 else "rgba(0,212,170,0.35)",
                    width=1 if g["midi"] % 12 != 0 else 1.5,
                    dash="dot",
                ),
                annotation_text=g["note"] if g["midi"] % 12 == 0 else "",
                annotation_position="right",
                annotation_font=dict(size=9, color=COLORS["muted"]),
            )

    # Pitch curve — only voiced
    xs, ys = [], []
    for t, f in zip(times, f0):
        if f and f > 0:
            xs.append(t)
            ys.append(f)
        else:
            # break line on unvoiced
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=COLORS["accent2"], width=1.8),
                        showlegend=False,
                        hovertemplate="t=%{x:.3f}s<br>f0=%{y:.1f} Hz<extra></extra>",
                    )
                )
                xs, ys = [], []
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="f0",
                line=dict(color=COLORS["accent2"], width=1.8),
                hovertemplate="t=%{x:.3f}s<br>f0=%{y:.1f} Hz<extra></extra>",
            )
        )

    corr = pitch_forensics.get("correction") or {}
    ttitle = title or pitch_forensics.get("label") or "Pitch lane"
    sub = corr.get("label") or ""
    fig.update_layout(
        title=dict(
            text=f"{ttitle}" + (f"<br><sup>{sub}</sup>" if sub else ""),
            font=dict(size=14, color=COLORS["text"]),
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface2"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            title="Time (s) — drag to pan, scroll/box to zoom",
            gridcolor=COLORS["border"],
            color=COLORS["muted"],
            rangeslider=dict(visible=True, bgcolor=COLORS["surface"]),
        ),
        yaxis=dict(
            title="Hz",
            gridcolor=COLORS["border"],
            color=COLORS["muted"],
            fixedrange=False,
        ),
        height=420,
        margin=dict(l=50, r=40, t=60, b=40),
        dragmode="zoom",
    )
    return fig
