"""
mixer.py — THE MIXER. Closed-loop chord balancing with the judge at the desk.

The problem (discovered by ear, 2026-06-11): strings differ wildly in
strength — necessity (z=+4.2) drowns divinity (z=+0.4) at equal alpha, and a
3-note chord turns into a unison of the loudest string. Balancing by hand
takes a human and several tries.

The fix is the same idea as the clamp, one level up:
  inner loop  (steering_model): a P-controller pins each dim's PROJECTION
              to its setpoint, token by token.
  outer loop  (this file):      a P-controller pins each dim's MEASURED
              EFFECT on the generated text — judged by the Humanizer — to a
              target, generation by generation. The string that came out too
              loud gets less alpha; the drowned one gets more.

Initial gains come from the tuner's table (afinacion.md): alpha inversely
proportional to string strength. RED strings (z < 1) are accepted with a
loud warning — they tend to inject noise, not meaning (the Portuguese-leak
lesson).

  python -m steering.mixer --demo sinfonia
  python -m steering.mixer --note d090_divinidad 0.95 --note d023_consciencia 0.95 \\
                           --phrase "Una habitación vacía."
  python -m steering.mixer --from-showcase amor_chord --iters 6

Output: per-iteration mixing table, the final text, and a ready-to-paste
console preset. Log: steering/vectors/mixer_last.json (with regime sha).
"""
import argparse
import json
import re
from datetime import datetime

import numpy as np

from . import config
from .fidelity import _vectors_sha
from .translator import Humanizer
from .steering_model import SteeredLlama

TUNING_MD = config.VEC_DIR / "afinacion.md"
LOG_FILE = config.VEC_DIR / "mixer_last.json"

ALPHA_MIN, ALPHA_MAX = 0.03, 0.45

# A chord of mind-colors, all causally validated strings:
# curiosity (z=+4.2 @0.35), oneirism (z=+2.5), love (z=+2.4).
DEMOS = {
    "sinfonia": dict(
        notes=[("d053_curiosidad", 0.95), ("d095_onirismo", 0.95),
               ("d046_amor", 0.95)],
        phrase="Describe una mañana cualquiera.",
    ),
}


# ── string strengths from the tuner ──────────────────────────────────────────
def load_strengths():
    """dim_name -> best |z| from afinacion.md. Empty dict if not generated yet."""
    if not TUNING_MD.exists():
        return {}
    out = {}
    for line in TUNING_MD.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(d\d{3}_\S+).*z=([+-]?\d+(?:\.\d+)?)", line)
        if m:
            out[m.group(1)] = abs(float(m.group(2)))
    return out


def initial_alpha(z):
    """Pressure inversely proportional to strength (the A-minor lesson):
    z=4.2 -> ~0.07, z=1.5 -> 0.20, z<=1 (red/unknown) -> 0.30."""
    return float(np.clip(0.30 / max(z, 1.0), 0.06, 0.30))


# ── the outer control loop ───────────────────────────────────────────────────
def mix(llm, hz, notes, phrase, layer, target=0.15, iters=5, tokens=110,
        verbose=True):
    """notes: [(dim_index, value)]. Returns (alphas, text, history)."""
    names = hz.dim_names
    idx = [i for i, _ in notes]
    signs = np.array([1.0 if v >= 0.5 else -1.0 for _, v in notes])

    strengths = load_strengths()
    alphas = np.array([initial_alpha(strengths.get(names[i], 0.0)) for i in idx])
    for (i, _), a in zip(notes, alphas):
        z = strengths.get(names[i])
        tag = f"z={z:+.1f}" if z is not None else "z=? (no afinacion.md)"
        if verbose:
            print(f"  · {names[i]:<32} {tag:<22} alpha inicial {a:.2f}"
                  + ("  ⚠ CUERDA ROJA: puede inyectar ruido" if (z or 0) < 1 else ""))

    llm.clear()
    neutral = llm.generate(phrase, max_new_tokens=tokens, temperature=0.0)
    base = hz.profile(neutral)

    history, text = [], neutral
    for it in range(1, iters + 1):
        # one profile per note value, all on one layer; per-note alpha needs
        # one voicing entry per alpha — group notes sharing the same alpha.
        voices = {}
        # multi-clamp accepts ONE alpha per layer; emulate per-note alphas by
        # scaling each note's weight: w_i' = w_i * (alpha_i / alpha_ref).
        # (targets are alpha*|h|*w, so scaling w is exactly scaling alpha.)
        a_ref = float(alphas.max())
        prof = np.full(len(names), 0.5, np.float32)
        for (i, v), a in zip(notes, alphas):
            w = (v - 0.5) * (a / a_ref)          # rescaled deflection
            prof[i] = 0.5 + w
        llm.set_voices({layer: (prof, a_ref)})
        text = llm.generate(phrase, max_new_tokens=tokens, temperature=0.0)
        llm.clear()

        effect = hz.profile(text) - base
        eff = np.array([float(effect[i]) for i in idx]) * signs   # signed "up"
        history.append(dict(alphas=alphas.tolist(), effects=eff.tolist()))

        if verbose:
            print(f"\n— iteración {it} (alpha ref {a_ref:.2f}) —")
            for (i, _), a, e in zip(notes, alphas, eff):
                mark = ("🎯" if 0.6 * target <= e <= 1.6 * target else
                        "🔼" if e < 0.6 * target else "🔽")
                print(f"  {mark} {names[i]:<32} α={a:.2f}  efecto={e:+.3f}"
                      f"  (objetivo {target:+.2f})")

        if all(0.6 * target <= e <= 1.6 * target for e in eff):
            if verbose:
                print("\n🎼 acorde equilibrado")
            break

        # multiplicative P-update: louder than target -> turn down, and v.v.
        for k, e in enumerate(eff):
            if e <= 0.01:                         # silent or inverted: push harder
                alphas[k] = min(alphas[k] * 1.5, ALPHA_MAX)
            else:
                alphas[k] = float(np.clip(alphas[k] * np.clip(target / e, 0.5, 2.0),
                                          ALPHA_MIN, ALPHA_MAX))
    return alphas, neutral, text, history


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", nargs=2, action="append", metavar=("DIM", "VAL"),
                    help="dim (name or index) and value 0..1; repeatable")
    ap.add_argument("--phrase", default="Una habitación vacía.")
    ap.add_argument("--demo", choices=sorted(DEMOS), help="preset chord")
    ap.add_argument("--from-showcase", dest="showcase_id",
                    help="take notes+phrase from a showcase example id")
    ap.add_argument("--layer", type=int, default=config.INJECT_LAYERS[0])
    ap.add_argument("--target", type=float, default=0.15,
                    help="desired per-note judged effect")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=110)
    args = ap.parse_args()

    print("⏳ Cargando juez (CPU)...")
    hz = Humanizer(device="cpu")
    names = hz.dim_names
    pos = {n: i for i, n in enumerate(names)}

    if args.showcase_id:
        from .showcase import EXAMPLES
        ex = next((e for e in EXAMPLES if e["id"] == args.showcase_id), None)
        assert ex, f"showcase id desconocido: {args.showcase_id}"
        notes = [(pos[n], v) for n, v, _l, _a in ex["notes"]]
        phrase = ex["phrase"]
    elif args.demo:
        d = DEMOS[args.demo]
        notes = [(pos[n], v) for n, v in d["notes"]]
        phrase = d["phrase"]
    else:
        assert args.note, "usa --note, --demo o --from-showcase"
        notes = [(pos[n] if n in pos else int(n), float(v)) for n, v in args.note]
        phrase = args.phrase

    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    print(f"✅ Mezclando {len(notes)} notas sobre «{phrase}» (capa {args.layer})\n")

    alphas, neutral, text, history = mix(
        llm, hz, notes, phrase, args.layer,
        target=args.target, iters=args.iters, tokens=args.tokens)

    print("\n⚪ NEUTRAL\n" + neutral)
    print("\n🔴 ACORDE FINAL\n" + text)
    preset = [(int(i), float(v), args.layer, round(float(a), 2))
              for (i, v), a in zip(notes, alphas)]
    print("\n🎚 preset para la consola (dim, valor, capa, alpha):")
    print("   " + str(preset))

    LOG_FILE.write_text(json.dumps(dict(
        phrase=phrase, layer=args.layer, target=args.target,
        notes=[[names[i], v] for i, v in notes],
        final_alphas=[round(float(a), 3) for a in alphas],
        history=history, neutral=neutral, text=text,
        vectors_sha=_vectors_sha(), mode=config.STEER_MODE,
        generated=f"{datetime.now():%Y-%m-%d %H:%M}",
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📋 log -> {LOG_FILE}")


if __name__ == "__main__":
    main()
