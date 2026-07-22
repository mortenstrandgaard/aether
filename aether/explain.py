"""
Explanatory layer for AETHER forensics & voice characteristics.

Educational content only — acoustic / DSP literacy, not medical or legal advice.
No personality claims.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Match-score dimensions
# ---------------------------------------------------------------------------

DIMENSION_EXPLAIN = {
    "timbre_mfcc": {
        "title": "Timbre (MFCC / mel)",
        "what": (
            "How the *colour* of the sound compares — the shape of the spectrum over time, "
            "summarised with MFCCs (classic speech/music features) and mel averages."
        ),
        "high": (
            "**High (≈80–100%)**: the spectral colour fingerprints align closely. "
            "Same source, same mic, or very similar instrument/voice colour under similar conditions."
        ),
        "mid": (
            "**Mid (≈50–80%)**: related colour but clear differences — EQ, distance, room, "
            "codec, or a different take/person with some overlap."
        ),
        "low": (
            "**Low (<50%)**: different body of sound — another instrument, another voice colour, "
            "heavy processing, or totally different material."
        ),
        "note": (
            "100% does **not** mean 'same human identity'. It means the measured timbre vectors "
            "are nearly identical on this clip pair."
        ),
    },
    "pitch": {
        "title": "Pitch / intonation",
        "what": (
            "Pitch centre (typical f0), range, variation, and rough contour shape. "
            "In voice mode this is speech melody and register — not lyric content."
        ),
        "high": "**High**: similar register and intonation habits on these clips.",
        "mid": "**Mid**: related but shifted (different key, mood, or speech effort).",
        "low": "**Low**: very different pitch centres or contour behaviour.",
        "note": (
            "Different sentences often lower contour/DTW-related scores even for the same speaker. "
            "Prefer timbre + register over temporal pitch locks for speech."
        ),
    },
    "spectral": {
        "title": "Spectral shape",
        "what": (
            "Brightness (centroid), bandwidth, flatness (tonal vs noisy), rolloff, contrast. "
            "Captures 'dark/bright', 'noisy/clean', 'narrow/wide'."
        ),
        "high": "**High**: similar brightness and noise/tonal balance.",
        "mid": "**Mid**: same family of sound but different EQ/room/distance.",
        "low": "**Low**: clearly different spectral world.",
        "note": "Phone vs studio mics often drop this score a lot for the same person.",
    },
    "envelope": {
        "title": "Envelope / articulation",
        "what": "Attack, decay, duration — how energy rises and falls.",
        "high": "**High**: similar articulation (pluck vs pad, short vs long).",
        "mid": "**Mid**: loosely related dynamics.",
        "low": "**Low**: different event shapes.",
        "note": "For speech, envelope is less diagnostic than timbre/pitch.",
    },
    "energy_dynamics": {
        "title": "Energy dynamics",
        "what": "How loudness and spectral motion change over time; reverb-tail proxy.",
        "high": "**High**: similar dynamic behaviour and space-ish tail.",
        "mid": "**Mid**: some shared motion, different room or compression.",
        "low": "**Low**: one steady, one pumping — or different environments.",
        "note": "Compression and loudness normalisation change this.",
    },
    "temporal_align": {
        "title": "Temporal alignment (DTW on MFCC)",
        "what": (
            "Whether the *sequence* of timbre frames can be warped to match. "
            "Sensitive to wording, timing, and structure."
        ),
        "high": "**High**: similar material evolving similarly in time (same phrase / same sample).",
        "mid": "**Mid**: related but different pacing or wording.",
        "low": (
            "**Low (<50%)**: different words, arrangement, or timeline. "
            "**Normal for two different sentences from the same speaker.**"
        ),
        "note": (
            "Do not read low DTW alone as 'different person' on speech. "
            "Combine with timbre + pitch + reliability."
        ),
    },
}


SCORE_BANDS = """
### How to read match %

| Band | Rough reading |
|------|----------------|
| **80–100%** | Strong acoustic similarity on the weighted mix of dimensions |
| **55–80%** | Partial overlap — investigate which dimensions agree |
| **40–55%** | Weak / mixed — often different conditions or different sources |
| **<40%** | Low overall similarity on these features |

**Reliability %** is separate: it says whether the *test conditions* (length, voicing, noise)
are good enough to trust the match %. High match + low reliability = still a weak case.
"""


# ---------------------------------------------------------------------------
# Voice characteristics — age, training, intervention (acoustic literacy)
# ---------------------------------------------------------------------------

VOICE_TRAIT_GUIDE = [
    {
        "trait": "Pitch centre (f0 mean / register)",
        "means": "Typical fundamental frequency while speaking — 'how high/low the voice sits'.",
        "age": (
            "Adult voices often settle after puberty. Later life can bring gradual shifts "
            "(sometimes slightly lower or more unstable f0); effects vary a lot by person."
        ),
        "train": (
            "Partly trainable: speaking pitch habits, singing technique, breath support. "
            "You can *use* a different part of your range; the comfortable resting register "
            "is harder to permanently relocate without long practice or clinical work."
        ),
        "hard": (
            "Laryngeal anatomy and hormones set broad constraints. "
            "Extreme permanent change usually needs medical/surgical pathways — not a app claim."
        ),
        "tech": "Pitch correction, formant shift, and resampling can fake register on a *recording*.",
    },
    {
        "trait": "Pitch range & variability (prosody)",
        "means": "How much the melody of speech moves — flat vs expressive intonation.",
        "age": "Can narrow with fatigue, habit, or some age-related changes; highly individual.",
        "train": (
            "**Highly trainable**: public speaking, acting, singing, dialect coaching — "
            "this is stemmeføring, not fixed identity."
        ),
        "hard": "Not a hard biometric; changes with mood, language, and situation.",
        "tech": "Prosody can be edited with pitch lanes in a DAW; our grid view helps spot over-flattening.",
    },
    {
        "trait": "Timbre / spectral colour (MFCC, centroid)",
        "means": "Tone colour — darker/brighter, duller/sharper, more 'in the chest' vs 'in the head' colour on the recording.",
        "age": (
            "Tissue and breath patterns can change colour over decades; "
            "smoking, health, and technique matter as much as age."
        ),
        "train": (
            "Somewhat trainable via resonance strategies, language, singing pedagogy — "
            "but less plastic than pure prosody."
        ),
        "hard": "Vocal tract length/shape contribute; mic and room often dominate a short phone clip.",
        "tech": "EQ, saturation, codecs (phone audio) heavily alter measured timbre.",
    },
    {
        "trait": "Formant proxies (F1–F3)",
        "means": (
            "Rough resonances of the vocal tract from LPC — related to vowel colour / tract geometry estimates. "
            "Proxies only, not lab endoscopy."
        ),
        "age": "Tract geometry is relatively stable in adulthood; small shifts possible.",
        "train": "Vowel placement and language habits shift apparent formants; structure remains constrained.",
        "hard": "True anatomical change is medical territory; don't over-read phone-clip LPC.",
        "tech": "Formant shifters exist in plugins; treat outliers with suspicion + other evidence.",
    },
    {
        "trait": "HNR / periodicity proxy",
        "means": "How periodic vs noisy the voice is — clearer harmonic stack vs breathy/rough energy.",
        "age": "Breathiness/roughness can increase with age or fatigue for some speakers.",
        "train": "Breath management and technique can improve clarity; illness/fatigue reverse it fast.",
        "hard": "Pathology is clinical — AETHER never diagnoses.",
        "tech": "Noise gates, de-essers, and codecs change this measure.",
    },
    {
        "trait": "Delivery rate & pauses (prosody)",
        "means": "Energy-peak rate and low-energy fraction — proxies for pace and pausing.",
        "age": "Pace habits vary; no simple universal age law.",
        "train": "**Very trainable** — speaking coaching, nerves, coffee, context.",
        "hard": "Almost nothing 'identity-fixed' here.",
        "tech": "Editing silence and time-stretch changes it instantly on a file.",
    },
    {
        "trait": "Pitch quantisation / correction signs",
        "means": (
            "How often f0 sits on equal-tempered note centres, flat plateaus, step jumps, low micro-vibrato — "
            "acoustic *hints* of hard pitch correction on melodic material."
        ),
        "age": "Not an age trait — a production/performance trait of the *recording*.",
        "train": "Excellent singers can also sit near centres; always read evidence list, not one number.",
        "hard": "Cannot name Auto-Tune vs Melodyne vs manual editing from this alone.",
        "tech": "This is exactly the 'was this tuned?' forensic layer — for files, not for 'who you are'.",
    },
]


def render_dimension_help_markdown() -> str:
    parts = ["## Match dimensions — what they mean\n", SCORE_BANDS, "\n"]
    for key, d in DIMENSION_EXPLAIN.items():
        parts.append(f"### {d['title']}\n")
        parts.append(f"- **What we measure:** {d['what']}\n")
        parts.append(f"- {d['high']}\n")
        parts.append(f"- {d['mid']}\n")
        parts.append(f"- {d['low']}\n")
        parts.append(f"- *Note:* {d['note']}\n\n")
    return "".join(parts)


def render_voice_guide_markdown(language: str = "en") -> str:
    if language == "da":
        return _voice_guide_da()
    lines = [
        "## Voice characteristics — literacy guide\n\n",
        "This is **acoustic literacy**, not psychology and not medical advice.\n\n",
        "| Trait | What it means | Age | Trainable? | Hard / intervention | Tech on a file |\n",
        "|-------|---------------|-----|------------|---------------------|----------------|\n",
    ]
    for t in VOICE_TRAIT_GUIDE:
        lines.append(
            f"| **{t['trait']}** | {t['means']} | {t['age']} | {t['train']} | {t['hard']} | {t['tech']} |\n"
        )
    lines.append(
        "\n**Bottom line:** Prosody and delivery move easily. "
        "Timbre and register move less, but *recordings* can be rewritten with tools. "
        "AETHER reports what the **file** does — not who someone 'is'.\n"
    )
    return "".join(lines)


def _voice_guide_da() -> str:
    return """## Stemme-karakteristika — forklarende lag

Dette er **akustisk literacy**, ikke psykologi og ikke lægelig rådgivning.

### Pitch / register (f0)
- **Betyder:** Hvor højt/lavt stemmen typisk ligger.
- **Alder:** Pubertet ændrer meget; senere i livet kan der komme gradvise skift — meget individuelt.
- **Trænes:** Delvist (vaner, teknik, sang). Komfort-register er sværere at flytte permanent.
- **Indgreb / hårdt:** Anatomi og hormoner sætter rammer. Permanente ekstreme skift er klinisk/kirurgisk territorium — ikke app-domæne.
- **På en fil:** Pitch-shift og autotune kan flytte det du *hører* i optagelsen.

### Prosodi / stemmeføring (melodi, tempo, pauser)
- **Betyder:** Hvordan stemmen bruges over tid.
- **Alder:** Vaner ændrer sig; ingen simpel alderslov.
- **Trænes:** **Meget** — tale, skuespil, sprog, situation.
- **Hårdt:** Næsten intet “biometrisk fast” her.
- **På en fil:** Let at klippe, time-stretche, tegne pitch.

### Klang / timbre (MFCC, centroid)
- **Betyder:** Farve — mørk/lys, mat/skarp.
- **Alder:** Kan ændre sig med væv, helbred, teknik over år.
- **Trænes:** Delvist via resonans/sprog; mindre plastisk end ren prosodi.
- **Hårdt:** Vokaltrakt spiller ind; **mic/rum/telefon** fylder ofte mere end alder i et kort klip.
- **På en fil:** EQ og codecs ændrer scoren kraftigt.

### Formant-proxies
- **Betyder:** Grove resonanser (LPC) — vokal/trackt-agtige estimater.
- **Forsigtig:** Telefon-LPC er ikke et laboratorium.

### Autotune / pitch-korrektionstegn
- **Betyder:** Snap-to-grid, flade plateauer, trin-hop — tegn i **optagelsen**.
- **Ikke:** Bevis for et bestemt plugin-navn. Dygtige sangere kan også ramme rent.

### 100% timbre vs. under 50% på andet
- **Timbre ~100%:** Farvefingeraftryk ligner hinanden **på disse klip**.
- **DTW/temporal <50%:** Ofte **forskellig tekst/timing** — helt normalt for samme person.
- **Spectral lav + timbre høj:** Måske samme kilde, andet rum/mic/EQ.
- Brug altid **flere dimensioner + reliability**, aldrig ét tal alene.
"""


def explain_score_snapshot(
    dimensions: dict[str, Any],
    score_pct: float,
    reliability_pct: float | None = None,
) -> str:
    """Short plain-language reading of one comparison result."""
    lines = [
        f"Overall match **{score_pct:.1f}%** on the weighted acoustic mix.",
    ]
    if reliability_pct is not None:
        lines.append(f"Reliability **{reliability_pct:.1f}%** (clip quality / length / voicing).")
    if not dimensions:
        return "\n".join(lines)

    ranked = sorted(
        ((k, dimensions[k].get("pct", 0)) for k in dimensions),
        key=lambda x: x[1],
        reverse=True,
    )
    strong = [k for k, p in ranked if p >= 80]
    weak = [k for k, p in ranked if p < 50]
    name = {k: DIMENSION_EXPLAIN.get(k, {}).get("title", k) for k, _ in ranked}

    if strong:
        lines.append("**Strong agreement:** " + ", ".join(name[k] for k in strong) + ".")
        for k in strong[:2]:
            note = DIMENSION_EXPLAIN.get(k, {}).get("note")
            if note:
                lines.append(f"- {name[k]}: {note}")
    if weak:
        lines.append("**Weaker dimensions:** " + ", ".join(name[k] for k in weak) + ".")
        if "temporal_align" in weak:
            lines.append(
                "- Low temporal/DTW on speech often means different words or timing — "
                "not automatic proof of a different speaker."
            )
    if score_pct >= 80 and reliability_pct is not None and reliability_pct < 50:
        lines.append(
            "⚠ High match but **low reliability** — treat as a weak hint; re-record longer/cleaner clips."
        )
    return "\n\n".join(lines)
