"""
Audio-sync pitch scrubber — self-contained HTML/JS component for Streamlit.

Playhead follows audio timeupdate; click seeks; wheel zooms time window.
No React/npm build required (streamlit.components.v1.html).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Optional

import numpy as np
import soundfile as sf
import streamlit.components.v1 as components

from aether.pitch_forensics import build_pitch_forensics


def _wav_b64(y: np.ndarray, sr: int) -> str:
    y = np.asarray(y, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    buf = io.BytesIO()
    sf.write(buf, y, int(sr), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_scrubber_payload(
    y: np.ndarray,
    sr: int,
    *,
    label: str = "clip",
    max_seconds: float = 45.0,
) -> dict[str, Any]:
    """
    Build pitch forensics + short audio payload for the scrubber.
    Caps length for browser memory.
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    nmax = int(max_seconds * sr)
    if len(y) > nmax:
        y = y[:nmax]

    pf = build_pitch_forensics(y, sr, label=label)
    lane = pf.get("lane") or {}
    times = lane.get("times") or []
    f0 = lane.get("f0") or []
    # Downsample for JS if huge
    if len(times) > 4000:
        step = max(1, len(times) // 4000)
        times = times[::step]
        f0 = f0[::step]

    grid = pf.get("note_grid") or []
    # Limit grid labels
    grid_js = [{"hz": g["hz"], "note": g["note"], "midi": g["midi"]} for g in grid]

    return {
        "label": label,
        "sr": int(sr),
        "duration": float(len(y) / sr),
        "audio_b64": _wav_b64(y, sr),
        "times": [float(t) for t in times],
        "f0": [float(f) if f else 0.0 for f in f0],
        "grid": grid_js,
        "correction": pf.get("correction") or {},
    }


def render_pitch_scrubber(
    payload: dict[str, Any],
    *,
    height: int = 480,
    key: str = "pitch_scrubber",
) -> None:
    """Render interactive audio + pitch scrubber in Streamlit."""
    # Embed JSON safely
    data_json = json.dumps(payload)
    # Escape for script tag
    data_json = data_json.replace("</", "<\\/")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --bg: #141418;
    --bg2: #1c1c22;
    --text: #ececf1;
    --muted: #8a8a9a;
    --accent: #7c5cfc;
    --accent2: #00d4aa;
    --border: #2e2e38;
  }}
  body {{
    margin: 0; padding: 10px 12px 12px 12px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, Segoe UI, sans-serif;
    font-size: 13px;
  }}
  .row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
  button {{
    background: var(--bg2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  button.primary {{ background: linear-gradient(135deg, #7c5cfc55, #00d4aa33); border-color: var(--accent); }}
  .time {{ font-family: ui-monospace, monospace; color: var(--accent2); min-width: 9rem; }}
  .hint {{ color: var(--muted); font-size: 11px; }}
  canvas {{
    width: 100%; height: 320px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    cursor: crosshair;
    display: block;
    touch-action: none;
  }}
  audio {{ width: 100%; margin-top: 8px; }}
  .badge {{
    border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px;
    color: var(--muted); font-size: 11px;
  }}
</style>
</head>
<body>
  <div class="row">
    <strong id="label"></strong>
    <span class="badge" id="corr"></span>
    <span class="time" id="clock">0.000 / 0.000 s</span>
  </div>
  <div class="row">
    <button class="primary" id="btnPlay">Play / Pause</button>
    <button id="btnRestart">Restart</button>
    <button id="btnZoomOut">Zoom out</button>
    <span class="hint">Click curve to seek · wheel to zoom time · drag playhead area</span>
  </div>
  <canvas id="cv"></canvas>
  <audio id="au" controls preload="auto"></audio>
<script>
const DATA = {data_json};
const au = document.getElementById('au');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const clock = document.getElementById('clock');
const labelEl = document.getElementById('label');
const corrEl = document.getElementById('corr');

labelEl.textContent = DATA.label || 'Pitch scrubber';
const c = DATA.correction || {{}};
corrEl.textContent = (c.correction_likelihood_pct != null)
  ? (c.correction_likelihood_pct + '% quantisation · ' + (c.label || ''))
  : 'pitch lane';

au.src = 'data:audio/wav;base64,' + DATA.audio_b64;

// View window in seconds
let viewStart = 0;
let viewEnd = Math.max(DATA.duration || 1, 0.1);
const fullDur = Math.max(DATA.duration || 1, 0.1);

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  cv.width = Math.floor(w * dpr);
  cv.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function xToTime(x) {{
  const w = cv.clientWidth;
  const pad = 48;
  const t = viewStart + (x - pad) / Math.max(w - pad - 12, 1) * (viewEnd - viewStart);
  return Math.min(fullDur, Math.max(0, t));
}}
function timeToX(t) {{
  const w = cv.clientWidth;
  const pad = 48;
  return pad + (t - viewStart) / Math.max(viewEnd - viewStart, 1e-9) * (w - pad - 12);
}}
function hzToY(hz, h, fmin, fmax) {{
  const padT = 12, padB = 24;
  const ny = h - padT - padB;
  const log = (v) => Math.log(Math.max(v, 1));
  const u = (log(hz) - log(fmin)) / Math.max(log(fmax) - log(fmin), 1e-9);
  return padT + (1 - u) * ny;
}}

function draw() {{
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1c1c22';
  ctx.fillRect(0, 0, w, h);

  const times = DATA.times || [];
  const f0 = DATA.f0 || [];
  let fmin = 70, fmax = 500;
  const voiced = [];
  for (let i = 0; i < f0.length; i++) if (f0[i] > 0) voiced.push(f0[i]);
  if (voiced.length) {{
    fmin = Math.max(50, Math.min(...voiced) * 0.85);
    fmax = Math.min(1200, Math.max(...voiced) * 1.15);
  }}

  // Grid lines
  const grid = DATA.grid || [];
  ctx.lineWidth = 1;
  for (const g of grid) {{
    if (g.hz < fmin || g.hz > fmax) continue;
    const y = hzToY(g.hz, h, fmin, fmax);
    const isC = (g.midi % 12 === 0);
    ctx.strokeStyle = isC ? 'rgba(0,212,170,0.35)' : 'rgba(124,92,252,0.18)';
    ctx.beginPath(); ctx.moveTo(48, y); ctx.lineTo(w - 12, y); ctx.stroke();
    if (isC) {{
      ctx.fillStyle = '#8a8a9a';
      ctx.font = '10px system-ui';
      ctx.fillText(g.note, w - 40, y - 2);
    }}
  }}

  // Pitch curve in view
  ctx.strokeStyle = '#00d4aa';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < times.length; i++) {{
    const t = times[i], f = f0[i];
    if (t < viewStart || t > viewEnd) {{ pen = false; continue; }}
    if (!(f > 0)) {{ pen = false; continue; }}
    const x = timeToX(t), y = hzToY(f, h, fmin, fmax);
    if (!pen) {{ ctx.moveTo(x, y); pen = true; }}
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Playhead
  const ct = au.currentTime || 0;
  const px = timeToX(ct);
  ctx.strokeStyle = '#7c5cfc';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 8); ctx.lineTo(px, h - 20); ctx.stroke();

  // Axes labels
  ctx.fillStyle = '#8a8a9a';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(viewStart.toFixed(2) + 's', 48, h - 6);
  ctx.fillText(viewEnd.toFixed(2) + 's', w - 70, h - 6);
  ctx.fillText(Math.round(fmax) + ' Hz', 4, 16);
  ctx.fillText(Math.round(fmin) + ' Hz', 4, h - 28);

  clock.textContent = ct.toFixed(3) + ' / ' + fullDur.toFixed(3) + ' s';
}}

function tick() {{
  // Autoscroll playhead into view when playing
  if (!au.paused) {{
    const ct = au.currentTime;
    const span = viewEnd - viewStart;
    if (ct > viewEnd - span * 0.15) {{
      const mid = ct + span * 0.2;
      viewStart = Math.max(0, mid - span);
      viewEnd = Math.min(fullDur, viewStart + span);
      if (viewEnd - viewStart < span) viewStart = Math.max(0, viewEnd - span);
    }}
  }}
  draw();
  requestAnimationFrame(tick);
}}

cv.addEventListener('click', (e) => {{
  const rect = cv.getBoundingClientRect();
  const t = xToTime(e.clientX - rect.left);
  au.currentTime = t;
  draw();
}});

cv.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = cv.getBoundingClientRect();
  const tFocus = xToTime(e.clientX - rect.left);
  const span = viewEnd - viewStart;
  const factor = e.deltaY > 0 ? 1.2 : 0.8;
  let newSpan = Math.min(fullDur, Math.max(0.15, span * factor));
  const r = (tFocus - viewStart) / span;
  viewStart = Math.max(0, tFocus - r * newSpan);
  viewEnd = Math.min(fullDur, viewStart + newSpan);
  viewStart = Math.max(0, viewEnd - newSpan);
  draw();
}}, {{ passive: false }});

document.getElementById('btnPlay').onclick = () => {{
  if (au.paused) au.play(); else au.pause();
}};
document.getElementById('btnRestart').onclick = () => {{
  au.currentTime = 0; au.play();
}};
document.getElementById('btnZoomOut').onclick = () => {{
  viewStart = 0; viewEnd = fullDur; draw();
}};

window.addEventListener('resize', resize);
au.addEventListener('loadedmetadata', () => {{
  viewEnd = au.duration || fullDur;
  resize();
}});
resize();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    components.html(html, height=height, scrolling=False)


def render_scrubber_from_audio(
    y: np.ndarray,
    sr: int,
    *,
    label: str = "clip",
    key: str = "scrubber",
) -> dict[str, Any]:
    """Prepare payload + render scrubber; returns correction metrics for Python side."""
    payload = prepare_scrubber_payload(y, sr, label=label)
    render_pitch_scrubber(payload, key=key)
    return payload.get("correction") or {}
