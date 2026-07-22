# AETHER Analyzer v1.0

**See every sound. Understand every layer.**

Classical DSP audio dissection for electronic music producers.  
**Zero AI / Zero Machine Learning** — only signal processing, feature extraction, and rule-based logic.

---

## What it does

1. **Load** MP3/WAV (or YouTube URL via yt-dlp)
2. **Analyze** BPM, musical key, RMS energy, spectral centroid
3. **Separate** harmonic / percussive layers (HPSS)
4. **Detect** sound events (`A-001`, `A-002`, …) and extract rich parameters
5. **Map** pitch (Hz) → musical note names
6. **Sound DNA** — compact fingerprint per event
7. **Classify** each event (Kick, Lead, Pad, …) with rule-based logic
8. **Preset dump** — Serum/Vital-style text parameters from features
9. **Resonance Map** — which events lock together (0–100% scores)
10. **Forensics v2** — compare speech/sounds → **match %** + **reliability %** + spectrograms/pitch + 1-vs-N ranking  
11. **Voice characteristics** — voice quality + prosody/stemmeføring cards (acoustic only — **not** personality)
11. **Search** your sample library with MFCC + spectral cosine similarity (optional DTW)
12. **Export** JSON, MIDI, individual event WAVs, or a full ZIP

### Per-event parameters

| Category   | Features                                              |
|------------|-------------------------------------------------------|
| Timing     | Start, end, duration                                  |
| Pitch      | Avg Hz, contour, note name (e.g. `F#4`)               |
| Spectral   | Centroid, rolloff, flatness, contrast, bandwidth, flux|
| Timbre     | 13 MFCCs + delta means                                |
| Waveform   | Saw / square / sine / triangle / noise / complex      |
| Envelope   | Attack, decay (approx)                                |
| HPSS       | Harmonic / percussive ratio                           |
| Effects    | Reverb tail, distortion, harmonic richness (rule-based)|
| **DNA**    | Compact code + pitch class, shape, envelope profile   |
| **Class**  | Kick, Snare, HiHat, Perc, Bass, Lead, Pluck, Pad, …   |
| **Preset** | Osc / filter / ADSR / LFO / FX text dump              |

---

## New in this build (extensions)

### 1. Sound DNA Fingerprint
Every event gets a compact DNA dict + code string, e.g.:

```text
DNA:A3|sawtooth|bright|pluck|H72|R18|D35
```

Shown in the event side panel under **Sound DNA**.

### 2. Automatic sound-type classification
Rule-based labels (no ML):

`Kick · Snare · HiHat · Perc · Bass · Lead · Pluck · Pad · Riser · Hit · Texture · FX · Atmos`

Uses thresholds on centroid, attack, duration, harmonic ratio, flatness, spectral flux, etc.  
Coloured timeline + icon pills in the UI.

### 3. Serum / Vital preset dump
Text block mapped from features:

- Oscillator / wavetable guess  
- Filter type + cutoff + resonance  
- Amp + filter ADSR  
- LFO rate suggestion  
- FX chain (dist / reverb / chorus)

Copy from the side panel (**Copy to Serum / Vital**). Not a loadable `.fxp` — dial by ear.

### 4. Resonance Score map
Pairwise 0–100% scores from:

| Component | Weight (default) |
|-----------|------------------|
| Pitch class proximity | 25% |
| Spectral / MFCC shape | 35% |
| Envelope similarity | 25% |
| Timing (proximity + beat grid) | 15% |

UI: **Resonance Map** tab (heatmap + top pairs) and per-event partner list.

---

## Quick start

> **Full install guide (Windows / macOS / Essentia / Python 3.11):** see **[SETUP.md](SETUP.md)**  
> **Share online with a non-technical friend:** see **[DEPLOY.md](DEPLOY.md)** (tunnel in 5 min, or Streamlit Cloud)

```bash
# Prefer Python 3.11
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt
pip install aubio          # optional onsets

# ffmpeg (YouTube) — once per machine
# Windows:  winget install ffmpeg
# macOS:    brew install ffmpeg

# Launch
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### What to install

| Component | Required? | Notes |
|-----------|-----------|--------|
| Python **3.11** (or 3.10–3.12) | Yes | 3.13 works for core; extras harder |
| `requirements.txt` | Yes | Streamlit, Librosa, Plotly, … |
| **ffmpeg** | For YouTube/URL | winget / brew / apt |
| **aubio** | Optional | Better onset backend |
| **Essentia** | Optional | Better key — see platforms below |

### Essentia — where it should run

AETHER **does not need** Essentia (librosa key fallback is built-in).

| Platform | Essentia practical? |
|----------|---------------------|
| **Linux** Ubuntu + Python 3.10–3.11 | Best |
| **macOS** + conda / 3.11 | Good chance |
| **Windows native** pip | Often fails (esp. 3.13) |
| **Windows + WSL2 Ubuntu** | Recommended if you want Essentia on a PC |

### In-app tools

- **Environment status** (sidebar) — shows aubio / Essentia / ffmpeg / Python  
- **Session save / load** (sidebar) — `.aether.zip` without re-analysing  
- Full steps and troubleshooting → **[SETUP.md](SETUP.md)**

---

## Project layout

```
Aether/
├── app.py                 # Streamlit UI
├── requirements.txt
├── README.md
├── SETUP.md               # Install guide (Win/Mac, 3.11, Essentia, ffmpeg)
└── aether/
    ├── __init__.py
    ├── config.py          # Settings, constants, dark palette, class colours
    ├── loader.py          # Load / normalize / yt-dlp
    ├── global_analysis.py # BPM, key, RMS, centroid
    ├── event_detection.py # HPSS + onsets + segments
    ├── features.py        # Per-event DSP features (+ spectral flux)
    ├── notes.py           # Hz ↔ note mapping
    ├── sound_dna.py       # DNA + classification + preset dump
    ├── resonance.py       # Cross-event resonance scores
    ├── similarity.py      # MFCC cosine + optional DTW (library search)
    ├── environment.py     # Runtime diagnostics (aubio/essentia/ffmpeg)
    ├── session_io.py      # Save/load .aether.zip sessions
    ├── export_utils.py    # JSON, MIDI, WAV, ZIP
    ├── pipeline.py        # Full analysis orchestration
    └── viz.py             # Plotly timeline / spectrogram / HPSS / resonance
```

---

## Usage

1. **Upload** a track or paste a YouTube URL → **Run analysis**
2. Inspect **BPM / Key / Duration / Events** (+ strongest resonance pair)
3. Browse **Timeline** (by sound class or HPSS), **Spectrogram**, **HPSS**, **Resonance Map**
4. Select an event → **DNA**, class pill, params, **preset dump**, resonance partners
5. Optionally index a **sample folder** → **Search library**
6. **Export** JSON (includes DNA, class, preset, resonance), MIDI, or ZIP of WAVs

**Advanced settings:** onset threshold, min/max event length, merge gap, HPSS margin, onset backend, DTW blend, timeline colour mode.

Heavy steps use `@st.cache_data`. Results live in `st.session_state`.

---

## Goals & non-goals (v1)

**Goals:** detailed per-sound analysis, interactive timeline, BPM/key/note mapping, DNA + classification, preset guidance, resonance map, feature-based similarity, clean exports.

**Non-goals:** real-time analysis, cloud processing, neural stem separation, generative AI suggestions, mobile app, loadable synth presets.

---

## Honest limitations

- **Stem separation** is HPSS only — not neural-net quality.
- **Event detection** is imperfect on dense, layered mixes.
- **Sound class / presets** are transparent heuristics, not ground truth.
- **Similarity & resonance** are strong for feature-space matches, not semantic “AI” understanding.
- Longer tracks need **CPU time** (features + O(n²) resonance pairs).

---

## Tech stack

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| Language         | Python 3.11+                                    |
| UI               | Streamlit (dark production theme)               |
| DSP              | Librosa, SciPy, NumPy, SoundFile (+ Essentia/Aubio optional) |
| Plots            | Plotly                                          |
| Similarity       | Cosine (MFCC + spectral) · optional DTW         |
| Resonance        | Weighted pitch / spectral / envelope / timing   |
| Export           | JSON, mido (MIDI), SoundFile (WAV)              |
| URLs             | yt-dlp                                          |

---

## License

Use freely for your productions and tooling. Built as a classical-DSP instrument for reverse-engineering electronic music.
