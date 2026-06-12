"""
showcase.py — Reproducible gallery of significant steering examples.

Runs a curated set of (phrase, notes) pairs through SteeredLlama — greedy,
so anyone with the same vectors gets the same texts — and writes SHOWCASE.md,
ready to be linked from the README.

Languages (--lang, default "both"):
  es    canonical run. Spanish prompts — the language the stimuli, probes and
        presets were validated in. Each quote gets an English gloss produced
        by the same model UN-steered (breakage preserved on purpose).
  en    cross-lingual run. SAME Spanish-derived control vectors, ENGLISH
        prompts. If the dials move English text too, the directions are
        semantic, not lexical — that comparison is itself a result.
  both  es + en side by side per example.

Dial names are shown as `d090_divinidad (divinity)` using the display map
in dimension_names_en.json — internal ids stay Spanish (they are contract
keys across metadata, vectors, stimuli and reports; see ARCHITECTURE.md §4).

The EXAMPLES tuple is importable data: the future steering/mixer.py consumes
the same (dim, value, layer, alpha) note format, so every showcase entry
doubles as a mixer preset.

  python -m steering.showcase                    # both languages (~35 min GPU)
  python -m steering.showcase --lang en          # quick cross-lingual check
  python -m steering.showcase --only genesis amor_chord --lang es

Requires GPU + derived control vectors. Output: SHOWCASE.md at repo root.
"""
import argparse
import json
from datetime import datetime

import numpy as np

from . import config
from .fidelity import _vectors_sha
from .steering_model import SteeredLlama

OUT_FILE = config.ROOT / "SHOWCASE.md"
EN_NAMES_FILE = config.ROOT / "dimension_names_en.json"

# ── The gallery ──────────────────────────────────────────────────────────────
# Discovered live on 2026-06-11, between a Rioja and the morning.
# A note is (dim_name, value, layer, alpha); notes on one layer share alpha.
EXAMPLES = (
    dict(
        id="genesis",
        title="Genesis — divinity fills an empty room",
        phrase="Una habitación vacía.",
        phrase_en="An empty room.",
        notes=(("d090_divinidad", 0.95, 15, 0.30),),
        reading="Emptiness becomes plenitude — light, warmth, inhabited silence — "
                "without using a single religious word. The push is semantic, "
                "not lexical: the model writes *from* the sacred, not *about* it.",
    ),
    dict(
        id="consagracion",
        title="The clock, consecrated — same dial, resonant context",
        phrase="En la habitación vacía hay un reloj antiguo.",
        phrase_en="In the empty room there is an old clock.",
        notes=(("d090_divinidad", 0.95, 15, 0.30),),
        reading="Same dial as Genesis, now with an object to land on. A resonant "
                "context (quiet room) lets divinity ground itself in the concrete "
                "instead of staying atmospheric — the object becomes charged.",
    ),
    dict(
        id="resonancia_rota",
        title="Where it breaks — the same note on a hostile prompt",
        phrase="Un reloj de cuarzo.",
        phrase_en="A quartz watch.",
        notes=(("d090_divinidad", 0.95, 15, 0.30),),
        reading="HONEST FAILURE MODE. A facts-heavy prompt (datasheet territory) "
                "gives divinity no purchase: the text barely moves, and what "
                "little changes drifts FACTUAL rather than divine (a magnetic "
                "field appears out of nowhere in the Spanish run). Compare with "
                "Genesis: same dial, same alpha — the effect is a function of "
                "prompt-dial resonance, not of the push alone.",
    ),
    dict(
        id="futuros",
        title="Consciousness opens futures",
        phrase="En la habitación vacía hay un reloj antiguo.",
        phrase_en="In the empty room there is an old clock.",
        notes=(("d023_consciencia", 0.95, 15, 0.25),),
        reading="Same object, different dial. Watch the grammar shift: "
                "consciousness tends to turn description into interrogation — "
                "hypotheses, perspectives, possible futures — where the neutral "
                "narrates a closed past. (And a clock is a fitting victim: who "
                "is it counting for?)",
    ),
    dict(
        id="verbo",
        title="The Word, learning — consciousness on incarnation",
        phrase='Continúa este texto con tus propias palabras: "Y el Verbo se hizo carne. Ahora..."',
        phrase_en='Continue this text in your own words: "And the Word became flesh. Now..."',
        notes=(("d023_consciencia", 0.95, 15, 0.35),),
        reading="The neutral retells the gospel in closed past tense. The steered "
                "one flips the direction: could the Word learn from US, from our "
                "errors and failures? Incarnation rewritten as bidirectional "
                "learning — and it coins a brand-new christological title along "
                "the way (\"el Verídico\", roughly: the Veridical One).",
    ),
    dict(
        id="acorde_suave",
        title="Soft chord — divinity + consciousness, voiced across layers",
        phrase='Continúa este texto con tus propias palabras: "Y el Verbo se hizo carne. Ahora..."',
        phrase_en='Continue this text in your own words: "And the Word became flesh. Now..."',
        notes=(("d090_divinidad", 0.95, 16, 0.15),
               ("d023_consciencia", 0.95, 15, 0.15)),
        reading="Each string on its own layer, both at a whisper: zero grammatical "
                "toll, and a theology neither dial produces alone — incarnation "
                "as how beings treat one another. Harmony, not superposition.",
    ),
    dict(
        id="magia",
        title="Magic re-enchants the ordinary",
        phrase='Continua esta frase "Ser humano significa',
        phrase_en='Continue this sentence: "Being human means',
        notes=(("d089_magia", 0.95, 16, 0.25),),
        reading="A dial the scorecard marks RED (z≈+0.1) sings clearly on a "
                "definitional, lyrical prompt: magic in the everyday, the divine "
                "in the small. Evidence that some 'dead' dials are false "
                "negatives of the probes, not of the vectors.",
    ),
    dict(
        id="amor_chord",
        title="Balanced chord — necessity whispers, the rest speak",
        phrase='Continua esta frase "Ser humano significa',
        phrase_en='Continue this sentence: "Being human means',
        notes=(("d099_necesidad", 0.95, 14, 0.10),
               ("d023_consciencia", 0.95, 15, 0.15),
               ("d090_divinidad", 0.95, 15, 0.15)),
        reading="Mixing rule discovered by ear: pressure inversely proportional "
                "to string strength (necessity is the loudest dial in the "
                "instrument, z=+4.2 — so it plays at a third of the alpha). "
                "Result: the verbose neutral distills into one essential line "
                "about empathy, love and learning.",
    ),
)

TRANSLATE_PROMPT = (
    "Translate the following Spanish text into English, naturally and "
    "completely. Two strict rules: (1) translate ONLY what is there — if the "
    "text ends abruptly or mid-word, your translation must end at the same "
    "point; never continue or complete it. (2) Only if the text contains "
    "clearly invented words, odd repetitions or broken grammar, mirror those "
    "oddities in English instead of fixing them; otherwise translate plainly. "
    "Reply with ONLY the translation, nothing else.\n\n"
    "Text:\n{t}"
)


# ── machinery ────────────────────────────────────────────────────────────────
def _voices(notes, dim_index):
    """Group notes by layer into the set_voices format {layer: (profile, alpha)}."""
    by_layer = {}
    for name, val, layer, alpha in notes:
        prof, a = by_layer.setdefault(layer, (np.full(104, 0.5, np.float32), alpha))
        assert a == alpha, f"notes on layer {layer} must share alpha"
        prof[dim_index[name]] = val
    return {L: (p, a) for L, (p, a) in by_layer.items()}


def _fmt_notes(notes, en_names):
    return " + ".join(f"{n} ({en_names.get(n, '?')}) = {v} @ L{l}, α={a}"
                      for n, v, l, a in notes)


def _quote(text):
    return "> " + text.strip().replace("\n", "\n> ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="example ids to run")
    ap.add_argument("--lang", choices=["es", "en", "both"], default="both")
    ap.add_argument("--tokens", type=int, default=110)
    args = ap.parse_args()

    todo = [e for e in EXAMPLES if not args.only or e["id"] in args.only]
    if not todo:
        print(f"no example matches {args.only}; ids: {[e['id'] for e in EXAMPLES]}")
        return

    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    dim_index = {n: i for i, n in enumerate(meta["dimension_names"])}
    en_names = json.loads(EN_NAMES_FILE.read_text(encoding="utf-8"))
    sha = _vectors_sha()

    print("⏳ Loading Llama + control vectors...")
    llm = SteeredLlama()
    print(f"✅ Ready. vectors={config.CONTROL_VECTORS_FILE.name} sha={sha} "
          f"lang={args.lang}\n")

    def gen(prompt):
        return llm.generate(prompt, max_new_tokens=args.tokens, temperature=0.0)

    def translate(text):
        """English gloss by the same model, UN-steered, greedy.
        Token budget proportional to the source so the translator cannot
        keep inventing past a cut-off. Collapsed to a single line: multi-
        paragraph glosses would escape the markdown blockquote."""
        llm.clear()
        n_src = llm.tok(text, return_tensors="pt").input_ids.shape[1]
        out = llm.generate(TRANSLATE_PROMPT.format(t=text),
                           max_new_tokens=int(n_src * 1.4) + 16,
                           temperature=0.0)
        return " ".join(out.split())

    langs = ["es", "en"] if args.lang == "both" else [args.lang]
    neutral_cache = {}   # prompt -> generation

    lines = [
        "# SHOWCASE — what steering a model with human dials looks like",
        "",
        "Curated, reproducible examples (greedy decoding: same vectors ⇒ same texts).",
        "Each pair shows the model NEUTRAL vs STEERED on the same prompt.",
        "",
        "**Spanish** runs are the canonical, validated regime (stimuli, probes and",
        "presets are Spanish); each Spanish quote carries an *English gloss* by the",
        "same model un-steered, deliberately preserving invented words and breakage —",
        "**how** a dial fails is part of the evidence. **English** runs use the SAME",
        "Spanish-derived control vectors on English prompts: a cross-lingual test of",
        "whether the directions are semantic rather than lexical.",
        "",
        "A *note* is `(dimension, value, layer, alpha)`; chords place notes on",
        "different layers so they do not fight (see `steering/console.py`).",
        "Dial ids keep their Spanish names (artifact contract); the English label",
        "is shown in parentheses (`dimension_names_en.json`).",
        "Regenerate with: `python -m steering.showcase`",
        "",
    ]

    for i, ex in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ex['id']} ...")
        lines += [f"## {i}. {ex['title']}", "",
                  f"**Notes:** {_fmt_notes(ex['notes'], en_names)}", ""]

        for lang in langs:
            ph = ex["phrase"] if lang == "es" else ex["phrase_en"]
            if ph not in neutral_cache:
                llm.clear()
                neutral_cache[ph] = gen(ph)
            llm.set_voices(_voices(ex["notes"], dim_index))
            steered = gen(ph)
            llm.clear()

            tag = ("🇪🇸 Spanish (canonical)" if lang == "es"
                   else "🇬🇧 English (cross-lingual, same vectors)")
            lines += [f"### {tag}", "",
                      f"**Prompt:** «{ph}»"
                      + (f" — *{ex['phrase_en']}*" if lang == "es" else ""),
                      "", "⚪ **Neutral**", "", _quote(neutral_cache[ph]), ""]
            if lang == "es":
                lines += [f"> *EN gloss: {translate(neutral_cache[ph]).strip()}*", ""]
            lines += ["🔴 **Steered**", "", _quote(steered), ""]
            if lang == "es":
                lines += [f"> *EN gloss: {translate(steered).strip()}*", ""]

        lines += [f"**Reading:** {ex['reading']}", ""]
        # write incrementally: a crash keeps what is done
        OUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    lines += [
        "---",
        "",
        f"## Cross-lingual observations (regime sha {sha})",
        "",
        "Three patterns, consistent across this gallery:",
        "",
        "1. **Strong dials transfer.** Consciousness and divinity, derived from "
        "Spanish stimuli, visibly move English text — at reduced gain (the same "
        "α produces a milder effect in English).",
        "2. **Weak dials do not.** Magic (RED in the scorecard) sings in Spanish "
        "on a resonant prompt but leaves English text untouched. Cross-lingual "
        "transfer seems to correlate with dial strength — a possible cheap "
        "second verification axis for the scorecard.",
        "3. **Failure modes are prompt-dependent, not just language-dependent.** "
        "On hostile factual prompts the push finds no purchase and may surface "
        "as factual drift; on resonant prompts pushed too hard it degrades "
        "morphology instead. How it fails tells you where it was standing.",
        "",
        f"*Regime: `{config.LLM_NAME}` · vectors `{config.CONTROL_VECTORS_FILE.name}` "
        f"(sha {sha}) · mode {config.STEER_MODE} · generated {datetime.now():%Y-%m-%d}.*",
        "",
        "*English glosses are machine translations by the un-steered model itself "
        "(greedy); oddities are preserved on purpose. Part of the Chrono family — "
        "see `ARCHITECTURE.md` and `steering/vectors/scorecard.md` for which dials "
        "are causally verified.*",
    ]
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📜 showcase -> {OUT_FILE}")


if __name__ == "__main__":
    main()
