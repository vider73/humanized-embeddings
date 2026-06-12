"""
harmony.py — The RESONANCE metric: how much quality does the response keep
depending on how aligned the prompt is with the strings you push?

For each (phrase, alpha) it measures three numbers:
  alignment : profile of the NEUTRAL text on the pushed dims (0..1).
              High = the prompt already lives in that territory.
  effect    : how much the dial moved its own dim (Humanizer judge, hi-vs-neutral).
  quality   : perplexity of the steered text under the CLEAN model, relative
              to the neutral's. ~1.0 = as fluent as ever; >1.5 = degrading.

The resonance hypothesis: the more alignment, the more usable alpha before
quality drops. If confirmed, the "alpha ceiling" is not a constant:
it is a function of the prompt-dial agreement.

  python -m steering.harmony --dim 95 --dim 53 --alphas 0.18 0.25 0.35
  python -m steering.harmony --dim 23 --phrase "Una habitacion vacia." --phrase "Lista de la compra."
"""
import argparse

import numpy as np
import torch

from .translator import Humanizer
from .steering_model import SteeredLlama

# default trio: dream territory / neutral / hostile bureaucracy
DEFAULT_PHRASES = (
    "Estoy durmiendo en mi habitacion.",
    "Una habitacion vacia.",
    "Redacta las instrucciones para rellenar un formulario de impuestos.",
)


@torch.no_grad()
def fluency_ppl(llm, text):
    """Perplexity of the text under the model WITHOUT steering (hooks silent)."""
    llm.clear()
    ids = llm.tok(text, return_tensors="pt").input_ids.to(llm.model.device)
    if ids.shape[1] < 2:
        return float("nan")
    loss = llm.model(ids, labels=ids).loss
    return float(torch.exp(loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, action="append", required=True,
                    help="dim a empujar (repetible para acordes)")
    ap.add_argument("--val", type=float, default=0.95)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.18, 0.25, 0.35])
    ap.add_argument("--phrase", action="append", default=None,
                    help="frase sonda (repetible); por defecto el trio")
    ap.add_argument("--tokens", type=int, default=120)
    args = ap.parse_args()
    phrases = args.phrase or list(DEFAULT_PHRASES)
    dims = args.dim

    print("⏳ Cargando humanizador (CPU)...")
    hz = Humanizer(device="cpu")
    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    dnames = [hz.dim_names[d].split("_", 1)[-1] for d in dims]
    print(f"✅ Listo. cuerdas={dnames} capas={llm.layers} modo={llm.mode}\n")

    prof = np.full(hz.human_dim, 0.5, np.float32)
    for d in dims:
        prof[d] = args.val

    rows = []
    for ph in phrases:
        llm.clear()
        neutral = llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)
        nprof = hz.profile(neutral)
        align = float(np.mean([nprof[d] for d in dims]))
        ppl_n = fluency_ppl(llm, neutral)

        for a in args.alphas:
            llm.alpha = a
            llm.set_profile(prof)
            steered = llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)
            llm.clear()
            sprof = hz.profile(steered)
            effect = float(np.mean([sprof[d] - nprof[d] for d in dims]))
            ppl_s = fluency_ppl(llm, steered)
            calidad = ppl_s / ppl_n if ppl_n == ppl_n else float("nan")
            rows.append((ph, align, a, effect, calidad))
            flag = "🟢" if calidad < 1.3 else ("🟡" if calidad < 1.8 else "🔴")
            print(f"{flag} «{ph[:38]:<38}» align={align:.2f} α={a:<5} "
                  f"efecto={effect:+.3f} calidad={calidad:.2f}x")

    print(f"\n{'='*72}")
    print("  RESUMEN — la hipotesis de la resonancia:")
    print("  (si alineacion alta sostiene calidad a alphas altos, confirmada)")
    print(f"  {'frase':<40} {'align':>5} " +
          " ".join(f"α={a}" for a in args.alphas))
    for ph in phrases:
        sub = [r for r in rows if r[0] == ph]
        cal = " ".join(f"{r[4]:5.2f}" for r in sub)
        print(f"  {ph[:40]:<40} {sub[0][1]:5.2f} {cal}")
    print("=" * 72)


if __name__ == "__main__":
    main()
