"""
Sound DNA, rule-based sound-type classification, and Serum/Vital preset dumps.

All logic is classical DSP + thresholds — zero AI / ML.

Per event this module can produce:
  1. sound_dna   — compact fingerprint dict + readable code string
  2. sound_class — Lead, Kick, Pad, … with confidence
  3. preset_dump — human-readable Serum/Vital-style parameter text
"""

from __future__ import annotations

import re
from typing import Any, Optional

from aether.config import NOTE_NAMES


# ---------------------------------------------------------------------------
# Sound class catalogue (icons used in UI)
# ---------------------------------------------------------------------------

SOUND_CLASSES = (
    "Kick",
    "Snare",
    "HiHat",
    "Perc",
    "Bass",
    "Lead",
    "Pluck",
    "Pad",
    "Riser",
    "Hit",
    "Texture",
    "FX",
    "Atmos",
    "Unknown",
)

SOUND_CLASS_ICONS = {
    "Kick": "🥁",
    "Snare": "💥",
    "HiHat": "🎩",
    "Perc": "🔔",
    "Bass": "🔊",
    "Lead": "🎸",
    "Pluck": "✨",
    "Pad": "🌊",
    "Riser": "📈",
    "Hit": "⚡",
    "Texture": "🧵",
    "FX": "🌀",
    "Atmos": "☁️",
    "Unknown": "❓",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def pitch_class_from_event(event: dict[str, Any]) -> str:
    """Extract pitch class (C, C#, …) from note name, else '—'."""
    note = event.get("note")
    if not note or not isinstance(note, str):
        return "—"
    # Match leading letter + optional accidental
    m = re.match(r"^([A-G][#b]?)", note.strip())
    if not m:
        return "—"
    name = m.group(1)
    flat_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    return flat_map.get(name, name)


def envelope_profile(event: dict[str, Any]) -> str:
    """
    Coarse envelope shape label from attack/decay/duration.
      punch | pluck | sustain | swell | soft
    """
    attack = _safe_float(event.get("attack_time"))
    decay = _safe_float(event.get("decay_time"))
    dur = _safe_float(event.get("duration"), 0.1)

    if attack < 0.02 and decay < 0.15 and dur < 0.35:
        return "punch"
    if attack < 0.04 and decay < 0.35 and dur < 0.8:
        return "pluck"
    if attack > 0.15 and dur > 1.0:
        return "swell"
    if attack < 0.08 and dur > 1.2:
        return "sustain"
    if attack > 0.08:
        return "soft"
    return "pluck"


def spectral_shape_label(event: dict[str, Any]) -> str:
    """Coarse brightness / noise class from centroid + flatness."""
    cent = _safe_float(event.get("spectral_centroid"))
    flat = _safe_float(event.get("spectral_flatness"))
    if flat > 0.25:
        return "noisy"
    if cent < 400:
        return "dark"
    if cent < 1200:
        return "warm"
    if cent < 3000:
        return "bright"
    return "air"


def harmonic_balance_label(ratio: float) -> str:
    if ratio >= 0.65:
        return "harmonic"
    if ratio <= 0.35:
        return "percussive"
    return "hybrid"


# ---------------------------------------------------------------------------
# 1. Sound DNA
# ---------------------------------------------------------------------------

def build_sound_dna(event: dict[str, Any]) -> dict[str, Any]:
    """
    Compact Sound DNA fingerprint for one event.

    Returns a dict with structured fields + a short code string like:
      DNA:A3|saw|bright|pluck|H72|R18|D35
    """
    pc = pitch_class_from_event(event)
    note = event.get("note") or "—"
    wave = (event.get("waveform_estimate") or "complex")[:8]
    shape = spectral_shape_label(event)
    env = envelope_profile(event)
    h_ratio = _safe_float(event.get("harmonic_ratio"), 0.5)
    h_pct = int(round(100 * h_ratio))
    reverb = int(round(100 * _safe_float(event.get("reverb_tail"))))
    dist = int(round(100 * _safe_float(event.get("distortion"))))
    cent = int(round(_safe_float(event.get("spectral_centroid"))))
    attack_ms = int(round(1000 * _safe_float(event.get("attack_time"))))
    decay_ms = int(round(1000 * _safe_float(event.get("decay_time"))))
    dur_ms = int(round(1000 * _safe_float(event.get("duration"))))
    balance = harmonic_balance_label(h_ratio)
    flat = _safe_float(event.get("spectral_flatness"))

    # Compact code: easy to scan / copy
    code = (
        f"DNA:{note}|{wave}|{shape}|{env}|H{h_pct}|R{reverb}|D{dist}"
    )

    return {
        "code": code,
        "pitch_class": pc,
        "note": note,
        "waveform": wave,
        "spectral_shape": shape,
        "envelope_profile": env,
        "harmonic_balance": balance,
        "harmonic_ratio": round(h_ratio, 3),
        "centroid_hz": cent,
        "flatness": round(flat, 4),
        "attack_ms": attack_ms,
        "decay_ms": decay_ms,
        "duration_ms": dur_ms,
        "reverb_tail_pct": reverb,
        "distortion_pct": dist,
    }


def format_dna_display(dna: dict[str, Any]) -> str:
    """Multi-line human-readable DNA block for the UI."""
    return (
        f"{dna.get('code', '')}\n"
        f"  pitch     {dna.get('note')}  ({dna.get('pitch_class')})\n"
        f"  wave      {dna.get('waveform')}\n"
        f"  spectrum  {dna.get('spectral_shape')}  ·  {dna.get('centroid_hz')} Hz\n"
        f"  envelope  {dna.get('envelope_profile')}  "
        f"A{dna.get('attack_ms')}ms D{dna.get('decay_ms')}ms\n"
        f"  balance   {dna.get('harmonic_balance')}  H{int(100 * _safe_float(dna.get('harmonic_ratio')))}%\n"
        f"  fx        reverb {dna.get('reverb_tail_pct')}%  ·  dist {dna.get('distortion_pct')}%"
    )


# ---------------------------------------------------------------------------
# 2. Rule-based sound type classification
# ---------------------------------------------------------------------------

def classify_sound_type(event: dict[str, Any]) -> dict[str, Any]:
    """
    Classify event into a production-oriented sound class using feature thresholds.

    Returns:
      {
        "sound_class": "Kick",
        "confidence": 0.0–1.0,
        "icon": "🥁",
        "reasons": ["…"],
      }
    """
    dur = _safe_float(event.get("duration"), 0.1)
    attack = _safe_float(event.get("attack_time"))
    decay = _safe_float(event.get("decay_time"))
    cent = _safe_float(event.get("spectral_centroid"))
    flat = _safe_float(event.get("spectral_flatness"))
    h_ratio = _safe_float(event.get("harmonic_ratio"), 0.5)
    pitch = event.get("pitch_hz")
    pitch_hz = _safe_float(pitch, 0.0) if pitch is not None else 0.0
    reverb = _safe_float(event.get("reverb_tail"))
    dist = _safe_float(event.get("distortion"))
    bw = _safe_float(event.get("spectral_bandwidth"))
    richness = _safe_float(event.get("harmonic_richness"))
    flux = _safe_float(event.get("spectral_flux"))  # optional, 0 if missing
    wave = (event.get("waveform_estimate") or "complex").lower()
    etype = (event.get("type") or "mixed").lower()  # harmonic/percussive/mixed

    scores: dict[str, float] = {c: 0.0 for c in SOUND_CLASSES if c != "Unknown"}
    reasons: dict[str, list[str]] = {c: [] for c in scores}

    def bump(cls: str, pts: float, why: str) -> None:
        scores[cls] = scores.get(cls, 0.0) + pts
        reasons[cls].append(why)

    # --- Kick: short, dark, percussive, low pitch ---
    if dur < 0.45 and cent < 900 and h_ratio < 0.55 and attack < 0.03:
        bump("Kick", 3.0, "short dark percussive")
    if pitch_hz and pitch_hz < 120 and attack < 0.04:
        bump("Kick", 2.0, f"sub-ish pitch {pitch_hz:.0f} Hz")
    if etype == "percussive" and cent < 600:
        bump("Kick", 1.5, "percussive + low centroid")

    # --- Snare: mid-bright, short, some noise ---
    if 0.08 < dur < 0.5 and 1200 < cent < 5500 and attack < 0.03:
        bump("Snare", 2.5, "short mid-bright snap")
    if flat > 0.12 and h_ratio < 0.5 and dur < 0.45:
        bump("Snare", 2.0, "noisy body")
    if etype == "percussive" and 1500 < cent < 4500:
        bump("Snare", 1.5, "perc mid centroid")

    # --- HiHat: very short, very bright, noisy ---
    if dur < 0.25 and cent > 4000 and (flat > 0.15 or wave == "noise"):
        bump("HiHat", 3.5, "short bright noise")
    if dur < 0.15 and cent > 3000 and h_ratio < 0.4:
        bump("HiHat", 2.0, "tick / hat profile")

    # --- Perc: short percussive residual ---
    if dur < 0.4 and h_ratio < 0.45 and attack < 0.05 and cent > 800:
        bump("Perc", 1.8, "short perc hit")
    if etype == "percussive" and 0.1 < dur < 0.6:
        bump("Perc", 1.0, "HPSS percussive")

    # --- Bass: low pitch, harmonic, longer than kick ---
    if pitch_hz and pitch_hz < 200 and h_ratio > 0.5 and dur > 0.15:
        bump("Bass", 3.0, f"low pitch {pitch_hz:.0f} Hz")
    if cent < 500 and h_ratio > 0.55 and dur > 0.2:
        bump("Bass", 2.0, "dark harmonic body")
    if wave in ("sawtooth", "square") and pitch_hz and pitch_hz < 180:
        bump("Bass", 1.5, f"{wave} low")

    # --- Lead: mid pitch, harmonic, clear attack, mid duration ---
    if pitch_hz and 180 < pitch_hz < 1500 and h_ratio > 0.55 and 0.15 < dur < 2.5:
        bump("Lead", 2.5, "mid pitched harmonic")
    if wave in ("sawtooth", "square") and attack < 0.08 and cent > 800:
        bump("Lead", 2.0, f"{wave} lead-like")
    if richness > 0.3 and h_ratio > 0.6 and 0.2 < dur < 2.0:
        bump("Lead", 1.5, "rich harmonics")

    # --- Pluck: fast attack, medium-short, decaying ---
    if attack < 0.04 and 0.1 < dur < 1.0 and h_ratio > 0.45 and decay < 0.6:
        bump("Pluck", 3.0, "fast attack + short decay")
    if envelope_profile(event) == "pluck" and h_ratio > 0.4:
        bump("Pluck", 2.0, "pluck envelope")

    # --- Pad: long, slow attack or sustain, harmonic, smooth ---
    if dur > 1.5 and h_ratio > 0.55 and attack > 0.05:
        bump("Pad", 3.0, "long harmonic body")
    if dur > 2.0 and reverb > 0.25 and flat < 0.2:
        bump("Pad", 2.0, "long + reverb")
    if envelope_profile(event) in ("sustain", "swell", "soft") and dur > 1.2:
        bump("Pad", 1.5, "sustained envelope")

    # --- Riser: long, rising energy proxy via flux / slow attack / brightening ---
    if dur > 1.5 and attack > 0.2 and cent > 1500:
        bump("Riser", 2.5, "long swell / build")
    if flux > 0.15 and dur > 1.0 and attack > 0.1:
        bump("Riser", 2.0, "spectral motion")
    if envelope_profile(event) == "swell" and dur > 1.0:
        bump("Riser", 2.0, "swell envelope")

    # --- Hit: short, impactful, hybrid ---
    if dur < 0.35 and attack < 0.025 and (cent > 1000 or dist > 0.3):
        bump("Hit", 2.5, "impact hit")
    if 0.05 < dur < 0.4 and h_ratio > 0.4 and etype in ("mixed", "percussive"):
        bump("Hit", 1.2, "short hybrid impact")

    # --- Texture: noisy, mid-long, not strongly pitched ---
    if flat > 0.2 and dur > 0.5 and (not pitch_hz or pitch_hz < 80 or pitch_hz > 0):
        if flat > 0.2 and h_ratio < 0.55:
            bump("Texture", 2.0, "noisy mid-length")
    if wave == "noise" and dur > 0.4:
        bump("Texture", 2.5, "noise texture")
    if bw > 3000 and flat > 0.15 and dur > 0.6:
        bump("Texture", 1.5, "wide noisy band")

    # --- FX: extreme parameters, distortion, weirdness ---
    if dist > 0.45 or (flat > 0.3 and reverb > 0.35):
        bump("FX", 2.5, "extreme fx signature")
    if wave == "complex" and (dist > 0.3 or reverb > 0.4) and dur > 0.3:
        bump("FX", 1.5, "complex + processing")

    # --- Atmos: long, soft, dark/warm, high reverb ---
    if dur > 2.0 and attack > 0.1 and reverb > 0.3 and cent < 2500:
        bump("Atmos", 3.0, "long soft atmospheric")
    if dur > 2.5 and h_ratio > 0.4 and flat < 0.25 and attack > 0.08:
        bump("Atmos", 1.5, "ambient pad-like")

    # Pick winner
    best = max(scores.items(), key=lambda kv: kv[1])
    best_class, best_score = best
    if best_score < 1.5:
        best_class = "Unknown"
        conf = 0.25
        why = ["no strong rule match"]
    else:
        # Softmax-ish confidence vs second place
        sorted_scores = sorted(scores.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        conf = float(min(0.95, 0.35 + 0.12 * best_score + 0.08 * max(0.0, best_score - second)))
        why = reasons.get(best_class, [])[:3]

    return {
        "sound_class": best_class,
        "confidence": round(conf, 3),
        "icon": SOUND_CLASS_ICONS.get(best_class, "❓"),
        "reasons": why,
        "scores": {k: round(v, 2) for k, v in scores.items() if v > 0},
    }


# ---------------------------------------------------------------------------
# 3. Serum / Vital style preset dump (text, rule-mapped from features)
# ---------------------------------------------------------------------------

def _wave_to_osc(wave: str) -> str:
    mapping = {
        "sine": "Basic Shapes → Sine",
        "triangle": "Basic Shapes → Triangle",
        "sawtooth": "Basic Shapes → Saw",
        "square": "Basic Shapes → Square",
        "noise": "Noise → White (or Analog Noise)",
        "complex": "Wavetable → Complex / Mixed (manual pick)",
    }
    return mapping.get((wave or "complex").lower(), mapping["complex"])


def _cutoff_from_centroid(cent: float) -> int:
    """Map spectral centroid (Hz) to a synth-ish filter cutoff (Hz)."""
    # Rough: cutoff sits near / a bit above perceived brightness
    c = max(80.0, min(18000.0, cent * 1.15 + 200.0))
    return int(round(c))


def _adsr_from_event(event: dict[str, Any]) -> dict[str, float]:
    """Map measured attack/decay/duration to ADSR seconds (clamped)."""
    a = max(0.001, min(2.0, _safe_float(event.get("attack_time"), 0.01)))
    d = max(0.01, min(3.0, _safe_float(event.get("decay_time"), 0.2)))
    dur = max(0.05, _safe_float(event.get("duration"), 0.3))
    # Sustain level: longer harmonic sounds sustain more
    h = _safe_float(event.get("harmonic_ratio"), 0.5)
    if dur > 1.2 and h > 0.55:
        s = 0.65
        r = min(2.5, max(0.15, dur * 0.35))
    elif dur < 0.35:
        s = 0.05
        r = min(0.4, max(0.05, d * 0.5))
    else:
        s = 0.35
        r = min(1.5, max(0.08, d * 0.6))
    return {
        "attack": round(a, 3),
        "decay": round(d, 3),
        "sustain": round(s, 2),
        "release": round(r, 3),
    }


def _filter_type(event: dict[str, Any]) -> str:
    flat = _safe_float(event.get("spectral_flatness"))
    cent = _safe_float(event.get("spectral_centroid"))
    if flat > 0.25:
        return "High-Pass (or Band-Pass for noise)"
    if cent < 500:
        return "Low-Pass 24 dB"
    if cent > 4000:
        return "Low-Pass 12 dB (bright)"
    return "Low-Pass 24 dB"


def _lfo_rate(event: dict[str, Any], bpm: Optional[float] = None) -> str:
    """Heuristic LFO suggestion from duration / motion."""
    dur = _safe_float(event.get("duration"), 0.5)
    flux = _safe_float(event.get("spectral_flux"))
    if flux > 0.12 or envelope_profile(event) == "swell":
        if bpm and bpm > 0:
            return f"1/4 note @ {bpm:.0f} BPM (or free ~{max(0.1, 1.0 / max(dur, 0.5)):.2f} Hz)"
        return f"free ~{max(0.05, 0.5 / max(dur, 0.5)):.2f} Hz"
    if dur > 1.5:
        return "slow free ~0.1–0.3 Hz (pad drift)"
    return "off (or subtle ~2–4 Hz vibrato)"


def build_preset_dump(
    event: dict[str, Any],
    bpm: Optional[float] = None,
    track_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a text-based Serum/Vital-style preset dump from analysed features.

    Returns dict with 'text' (copy-paste block) and structured 'params'.
    """
    wave = event.get("waveform_estimate") or "complex"
    osc = _wave_to_osc(wave)
    cent = _safe_float(event.get("spectral_centroid"), 1200)
    cutoff = _cutoff_from_centroid(cent)
    adsr = _adsr_from_event(event)
    ftype = _filter_type(event)
    h_ratio = _safe_float(event.get("harmonic_ratio"), 0.5)
    reverb = _safe_float(event.get("reverb_tail"))
    dist = _safe_float(event.get("distortion"))
    richness = _safe_float(event.get("harmonic_richness"))
    note = event.get("note") or "—"
    pitch_hz = event.get("pitch_hz")
    sclass = (event.get("sound_class") or event.get("classification", {}) or {})
    if isinstance(sclass, dict):
        class_name = sclass.get("sound_class") or "Unknown"
    else:
        class_name = str(sclass)

    # Unison / detune from richness
    if richness > 0.45 or wave in ("sawtooth", "square"):
        unison = 3 if class_name in ("Lead", "Pad", "Atmos") else 2
        detune = 0.12 if class_name in ("Pad", "Atmos") else 0.06
    else:
        unison = 1
        detune = 0.0

    # Drive / warp
    warp = "off"
    if wave == "square":
        warp = "PWM ~35–50%"
    elif wave == "sawtooth" and dist > 0.25:
        warp = "Sync or Bend + mild drive"
    elif dist > 0.4:
        warp = "Drive / Distortion stage ~{:.0f}%".format(100 * dist)

    # Sub osc for bass/kick-ish
    sub = "off"
    if class_name in ("Bass", "Kick") or (pitch_hz and float(pitch_hz) < 120):
        sub = "Sine −12 st, level ~−6 to −3 dB"

    # FX chain
    fx_parts = []
    if dist > 0.2:
        fx_parts.append(f"Distortion/Diode ~{int(100 * dist)}%")
    if reverb > 0.15:
        size = "Hall" if reverb > 0.4 else "Room"
        fx_parts.append(f"Reverb {size} mix ~{int(100 * reverb * 0.6)}%")
    if class_name in ("Pad", "Atmos", "Texture"):
        fx_parts.append("Chorus mild 10–20%")
    if not fx_parts:
        fx_parts.append("clean (no heavy FX)")

    res = int(np_clip_res(richness, h_ratio, flatness=_safe_float(event.get("spectral_flatness"))))
    lfo = _lfo_rate(event, bpm=bpm)

    params = {
        "osc_a": osc,
        "osc_b": sub if sub != "off" else "off",
        "unison": unison,
        "detune": detune,
        "warp": warp,
        "filter_type": ftype,
        "cutoff_hz": cutoff,
        "resonance": res,
        "filter_env_amt": int(40 + 40 * min(1.0, cent / 4000)),
        "amp_adsr": adsr,
        "filter_adsr": {
            "attack": adsr["attack"],
            "decay": round(adsr["decay"] * 0.85, 3),
            "sustain": round(max(0.05, adsr["sustain"] * 0.7), 2),
            "release": adsr["release"],
        },
        "lfo": lfo,
        "fx": fx_parts,
        "root_note": note,
        "sound_class": class_name,
    }

    text = (
        f"// AETHER → Serum / Vital preset dump  [{event.get('id', '?')} · {class_name}]\n"
        f"// Source note: {note}"
        + (f"  ({float(pitch_hz):.1f} Hz)" if pitch_hz else "")
        + (f"  |  Track key: {track_key}" if track_key else "")
        + (f"  |  BPM: {bpm:.1f}" if bpm else "")
        + "\n"
        f"// DNA: {(event.get('sound_dna') or {}).get('code', '—') if isinstance(event.get('sound_dna'), dict) else '—'}\n"
        f"\n"
        f"[OSCILLATORS]\n"
        f"  OSC A: {osc}\n"
        f"  OSC B / Sub: {sub}\n"
        f"  Unison: {unison}  ·  Detune: {detune:.2f}\n"
        f"  WT Position / Warp: {warp}\n"
        f"  Level: 0 dB (trim to taste)\n"
        f"\n"
        f"[FILTER]\n"
        f"  Type: {ftype}\n"
        f"  Cutoff: ~{cutoff} Hz\n"
        f"  Resonance: {res}%\n"
        f"  Env Amount: {params['filter_env_amt']}%\n"
        f"\n"
        f"[AMP ENVELOPE]\n"
        f"  A {adsr['attack']:.3f}s   D {adsr['decay']:.3f}s   "
        f"S {adsr['sustain']:.2f}   R {adsr['release']:.3f}s\n"
        f"\n"
        f"[FILTER ENVELOPE]\n"
        f"  A {params['filter_adsr']['attack']:.3f}s   "
        f"D {params['filter_adsr']['decay']:.3f}s   "
        f"S {params['filter_adsr']['sustain']:.2f}   "
        f"R {params['filter_adsr']['release']:.3f}s\n"
        f"\n"
        f"[LFO]\n"
        f"  Rate: {lfo}\n"
        f"  Target suggestion: Filter cutoff or WT position (subtle)\n"
        f"\n"
        f"[FX]\n"
        + "".join(f"  - {p}\n" for p in fx_parts)
        + f"\n"
        f"[NOTES]\n"
        f"  Rule-mapped from classical DSP features (centroid, envelope, HPSS, flatness).\n"
        f"  Not a loadable .fxp — dial by ear in Serum / Vital / any wavetable synth.\n"
    )

    return {"text": text, "params": params}


def np_clip_res(richness: float, h_ratio: float, flatness: float = 0.0) -> int:
    """Resonance % heuristic 0–100."""
    base = 15 + 50 * richness + 20 * h_ratio - 40 * flatness
    return int(max(0, min(85, round(base))))


# ---------------------------------------------------------------------------
# Enrich a single event with DNA + class + preset
# ---------------------------------------------------------------------------

def enrich_event(
    event: dict[str, Any],
    bpm: Optional[float] = None,
    track_key: Optional[str] = None,
) -> dict[str, Any]:
    """Attach sound_dna, sound_class fields, and preset_dump to event (in-place copy)."""
    out = dict(event)
    dna = build_sound_dna(out)
    classification = classify_sound_type(out)
    out["sound_dna"] = dna
    out["sound_class"] = classification["sound_class"]
    out["sound_class_confidence"] = classification["confidence"]
    out["sound_class_icon"] = classification["icon"]
    out["sound_class_reasons"] = classification["reasons"]
    out["classification"] = classification
    # Preset uses class just assigned
    out["preset_dump"] = build_preset_dump(out, bpm=bpm, track_key=track_key)
    return out


def enrich_all_events(
    events: list[dict[str, Any]],
    bpm: Optional[float] = None,
    track_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    return [enrich_event(e, bpm=bpm, track_key=track_key) for e in events]
