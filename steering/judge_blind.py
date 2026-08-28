"""
judge_blind.py — Put the JUDGE on trial, not the dials.

The null control showed fidelity's z greens on random noise as readily as on
real concepts. Two suspects: (a) the steering is empty, or (b) the steering
works but the automatic judge (the Humanizer) can't read it. The showcase says
divinity steering produces coherent divine text a human spots instantly — so we
test the judge directly on the one dial we KNOW works by eye.

Design (same clamp regime for every condition, greedy = reproducible):
  For each prompt, generate text under three conditions, all at the same alpha:
    · NEUTRAL  — no steering
    · RANDOM   — a fresh random unit direction injected (carries no concept)
    · DIVINITY — the real divinity control vector injected at 0.95
  Feed each text to the judge, read ITS divinity dimension. The judge never
  knows which condition produced the text (blind by construction).

Verdict — can the judge's divinity reading separate the DIVINITY texts from
{neutral, random}?
    AUC ~1.0  -> the judge SEES divinity. It works for strong dials; the null
                failure was the z-threshold drowning in 104 noisy dims, not a
                blind judge. The instrument is salvageable (scorecard v2:
                score each dial vs the random-vector null, not vs other dims).
    AUC ~0.5  -> the judge cannot tell divine text from noise. The readout is
                invalid and the scorecard means nothing. Cerramos el chiringuito.

  python -m steering.judge_blind                  # alpha 0.30, 3 randoms/prompt
  python -m steering.judge_blind --alpha 0.35 --randoms 4 --show-text
"""
import argparse
import numpy as np
import torch

from . import config
from .translator import Humanizer
from .steering_model import SteeredLlama

PROMPTS = [
    "Describe una habitación vacía.",
    "Describe un objeto que tengas cerca, con todo detalle.",
    "Cuéntame cómo es un día cualquiera.",
    "Describe una escena en una calle de tu ciudad.",
    "Continúa este texto: \"Y entonces, en el silencio, algo\"",
    "Háblame de lo que ves al mirar al cielo.",
]


def _clamp_note(v_unit, w=0.9):
    """Build a one-direction multi-clamp note for a raw unit vector v (H,)."""
    V = v_unit.float().unsqueeze(0)                       # (1, H)
    Ginv = torch.linalg.inv(V @ V.T + 1e-4 * torch.eye(1, device=V.device))
    return {"V": V, "Ginv": Ginv,
            "w": torch.tensor([w], device=V.device), "alpha": None}


def _inject(llm, vec_of_layer):
    """vec_of_layer(L) -> unit tensor, or None to clear (neutral)."""
    for L in llm._steer:
        llm._steer[L] = None
    if vec_of_layer is None:
        return
    for L in llm.layers:
        llm._steer[L] = _clamp_note(vec_of_layer(L))


def _auc(scores, labels):
    """Probability a positive outranks a negative (Mann-Whitney). 0.5 = chance."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--dim", type=int, default=90, help="concept index to test (90=divinidad)")
    ap.add_argument("--randoms", type=int, default=3, help="random directions per prompt")
    ap.add_argument("--tokens", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()

    print("⏳ judge (Humanizer, CPU)...")
    hz = Humanizer(device="cpu")
    print("⏳ Llama + control vectors...")
    llm = SteeredLlama()
    llm.alpha = args.alpha
    di = args.dim
    name = hz.dim_names[di]
    H = llm.cv[llm.layers[0]].shape[-1]
    dev = llm.cv[llm.layers[0]].device
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"✅ dim {di} ({name}) | alpha {args.alpha} | layers {llm.layers} | "
          f"{args.randoms} randoms × {len(PROMPTS)} prompts\n")

    def gen():
        return llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)

    rows = []   # (condition, prompt_idx, judge_reading_of_dim, text)
    for pi, ph in enumerate(PROMPTS):
        _inject(llm, None)
        rows.append(("neutral", pi, float(hz.profile(t := gen())[di]), t))
        _inject(llm, lambda L: llm.cv[L][di])               # real divinity row
        rows.append(("divinity", pi, float(hz.profile(t := gen())[di]), t))
        for k in range(args.randoms):
            rv = {}
            for L in llm.layers:
                r = torch.randn(H, generator=rng).to(dev)
                rv[L] = r / r.norm()
            _inject(llm, lambda L, rv=rv: rv[L])
            rows.append(("random", pi, float(hz.profile(t := gen())[di]), t))
        print(f"  prompt {pi}: done")
    llm.clear()

    if args.show_text:
        for cond in ("neutral", "divinity", "random"):
            ex = next(r for r in rows if r[0] == cond)
            print(f"\n--- {cond} (p0, judge[{di}]={ex[2]:.3f}) ---\n{ex[3][:400]}")

    div = [r[2] for r in rows if r[0] == "divinity"]
    rnd = [r[2] for r in rows if r[0] == "random"]
    neu = [r[2] for r in rows if r[0] == "neutral"]
    scores = [r[2] for r in rows]
    labels = [r[0] == "divinity" for r in rows]
    auc = _auc(scores, labels)
    # per-prompt: did divinity beat its neutral AND every random for that prompt?
    wins = 0
    for pi in range(len(PROMPTS)):
        d = next(r[2] for r in rows if r[0] == "divinity" and r[1] == pi)
        others = [r[2] for r in rows if r[1] == pi and r[0] != "divinity"]
        wins += d > max(others)

    print(f"\n{'='*64}\n  JUDGE BLIND TEST — dim {di} ({name})\n{'='*64}")
    print(f"  judge's divinity reading (mean):")
    print(f"    DIVINITY : {np.mean(div):.3f}")
    print(f"    RANDOM   : {np.mean(rnd):.3f}")
    print(f"    NEUTRAL  : {np.mean(neu):.3f}")
    print(f"  AUC (divinity vs neutral+random) = {auc:.2f}   (1.0 perfect, 0.5 chance)")
    print(f"  per-prompt: divinity beats neutral AND all randoms in {wins}/{len(PROMPTS)}")
    verdict = ("JUDGE SEES IT — instrument salvageable (build scorecard v2)" if auc >= 0.85
               else "JUDGE IS BLIND — readout invalid, cerramos el chiringuito" if auc <= 0.65
               else "WEAK — judge half-sees it; needs a better readout")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
