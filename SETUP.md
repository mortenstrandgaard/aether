# AETHER Analyzer — Setup & install guide

**See every sound. Understand every layer.**  
Classical DSP only — no AI required.

This guide covers: what to install, Windows + macOS, Python 3.11 recommendation, and where **Essentia** actually works.

---

## What you need (checklist)

| Component | Required? | Purpose | Windows | macOS |
|-----------|-----------|---------|---------|-------|
| **Python 3.11** (ideal) or 3.10–3.12 | **Yes** | Run AETHER | [python.org](https://www.python.org/downloads/) | python.org or `brew install python@3.11` |
| **pip + venv** | **Yes** | Dependencies | included with Python | included |
| **requirements.txt packages** | **Yes** | Core app | `pip install -r requirements.txt` | same |
| **ffmpeg** | For YouTube/URL | yt-dlp audio extract | `winget install ffmpeg` | `brew install ffmpeg` |
| **aubio** | Optional | Extra onset backend | `pip install aubio` | `pip install aubio` |
| **Essentia** | Optional | Better key detection | Prefer **WSL2 Ubuntu**, not native Win | conda / pip on 3.11 |
| **Git** | Optional | Clone/update repo | winget / git-scm | Xcode CLT / brew |

### What works without optional tools

- Local **MP3/WAV** analysis → full pipeline, DNA, class, presets, resonance, exports  
- **Key detection** via librosa (no Essentia)  
- **Onsets** via librosa (no aubio)  

### What needs extras

| Feature | Needs |
|---------|--------|
| Paste YouTube / URL | **ffmpeg** + yt-dlp (yt-dlp is in requirements) |
| Onset backend “aubio” | **aubio** |
| Essentia KeyExtractor | **Essentia** (platform-dependent) |

---

## Recommended: Python 3.11 venv (Windows)

You may currently be on Python 3.13. Core AETHER works, but optional packages are happier on **3.11**.

### 1. Install Python 3.11

1. Download **Python 3.11.x** from https://www.python.org/downloads/  
2. Install with **“Add python.exe to PATH”**  
3. Verify in a **new** terminal:

```powershell
py -3.11 --version
# or
python3.11 --version
```

### 2. Create a clean venv for AETHER

```powershell
cd "C:\Users\Morten\Documents\App concepts\Aether"

# Remove old 3.13 venv only if you want a clean switch (optional)
# Remove-Item -Recurse -Force .venv

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
# expect: Python 3.11.x

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install aubio
```

### 3. Install ffmpeg (YouTube)

```powershell
winget install ffmpeg
```

Close and reopen the terminal, then:

```powershell
ffmpeg -version
```

### 4. Run

```powershell
cd "C:\Users\Morten\Documents\App concepts\Aether"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

In the app: open the **Environment** expander in the sidebar to verify green checks.

---

## Recommended: macOS

### 1. Homebrew + Python 3.11

```bash
# Homebrew: https://brew.sh
brew install python@3.11 ffmpeg

# Optional
brew install git
```

### 2. Venv + deps

```bash
cd "/path/to/Aether"
/opt/homebrew/bin/python3.11 -m venv .venv   # Apple Silicon
# or: /usr/local/bin/python3.11 -m venv .venv  # Intel

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install aubio
```

### 3. Run

```bash
source .venv/bin/activate
streamlit run app.py
```

---

## Essentia — where should it run?

**AETHER never requires Essentia.**  
If missing, key detection uses **librosa chroma + Krumhansl–Schmuckler templates**.

### Reality by platform

| Platform | Essentia via pip? | Practical approach |
|----------|-------------------|--------------------|
| **Linux** (Ubuntu 22.04+, Python 3.10–3.11) | Often **yes** | Best first choice for Essentia |
| **macOS** (3.10–3.11) | Sometimes | Prefer **conda-forge** if pip fails |
| **Windows native** | Rarely / broken on 3.12–3.13 | **WSL2 Ubuntu** or skip Essentia |
| **Python 3.13** | Almost never | Use **3.11** if you want Essentia |

### If you want Essentia on a Windows PC

Use **WSL2** (Linux inside Windows) — not native Win Python:

```powershell
wsl --install
# reboot if asked, then open Ubuntu
```

```bash
# inside Ubuntu (WSL)
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg build-essential
cd /mnt/c/Users/Morten/Documents/App\ concepts/Aether
python3.11 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install -r requirements.txt
pip install essentia aubio
streamlit run app.py
```

Open the URL Streamlit prints (often need to use the WSL IP or `localhost` forwarding).

### If you want Essentia on macOS (conda)

```bash
# Install Miniconda/Miniforge first, then:
conda create -n aether python=3.11
conda activate aether
conda install -c conda-forge essentia
pip install -r requirements.txt
streamlit run app.py
```

(Exact conda package name can vary; check [Essentia install docs](https://essentia.upf.edu/installing.html) if the channel changes.)

### Bottom line

- **Daily use (Win or Mac):** install **Python 3.11 + requirements + ffmpeg (+ aubio)**. Skip Essentia.  
- **Maximum key-detection quality:** Linux or macOS conda / WSL2 Ubuntu with Essentia.  
- **Do not block shipping** on Essentia — AETHER is designed to run without it.

---

## Session save / load

After analysis:

1. Sidebar → **Session** → **Download session (.aether.zip)**  
2. Later: upload that file under **Load session**  
3. Restores events, DNA, classes, presets, resonance (and audio if packed)

No need to re-run the full DSP pipeline.

---

## Environment panel in the app

Sidebar → **Environment status** shows:

- Python version  
- Core packages (streamlit, librosa, …)  
- aubio / Essentia / ffmpeg  
- Capabilities: local analysis, YouTube, etc.  
- Install hints tailored to your OS  

---

## Quick troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit` not found | Activate venv first |
| YouTube fails | Install ffmpeg; restart terminal |
| Essentia install fails | Expected on Win/3.13 — skip or use WSL/3.11 |
| aubio build fails | Optional; use onset backend **librosa** |
| Spectrogram missing after session load | Session saved without audio — re-save with “include audio” or re-upload track |
| Very slow on long tracks | Shorter clip, higher onset threshold, or fewer events via advanced settings |

---

## Minimal “just run it” path

```text
1. Python 3.11
2. python -m venv .venv && activate
3. pip install -r requirements.txt
4. winget install ffmpeg   OR   brew install ffmpeg
5. streamlit run app.py
```

That’s enough for a complete local AETHER workflow on **Windows and macOS**.
