"""
blind_duel.py — HUMAN vs JUDGE, blind, head to head.

The decisive test José asked for. We generate texts: some steered toward
divinity, some toward consciousness, some neutral (nothing touched). The texts
are shuffled and stripped of any label. Then BOTH judges classify each one:
  · the HUMAN (you) reads the text and guesses: none / divinity / consciousness
  · the JUDGE (the Humanizer) reads its 104-profile and guesses the same
If the human nails it and the judge doesn't -> the steering is real and legible;
the automatic judge is the broken instrument. If neither can -> the steering is
empty. If both can -> the judge is fine and the null failure was the metric.

Two steps:
  1) python -m steering.blind_duel
       -> generates texts, writes blind_worksheet.md (no labels, for you) and
          blind_key.json (hidden truth + the judge's guesses). Same clamp regime
          as the showcase; greedy = reproducible.
  2) read blind_worksheet.md, write your guesses, then:
       python -m steering.blind_duel --score "T01 div, T02 none, ..."
       -> human accuracy vs judge accuracy, side by side.

  options: --prompts N  --alpha 0.30  --concepts 90 23  --seed 0
"""
import argparse
import json
import random

import numpy as np
import torch

from . import config
from .translator import Humanizer
from .steering_model import SteeredLlama
from .judge_blind import _clamp_note, PROMPTS

KEY = config.ROOT / "blind_key.json"
SHEET = config.ROOT / "blind_worksheet.md"
SHORT = {"none": "none", "n": "none", "div": "divinity", "d": "divinity",
         "divinity": "divinity", "cons": "consciousness", "c": "consciousness",
         "consciousness": "consciousness", "divinidad": "divinity",
         "consciencia": "consciousness", "conciencia": "consciousness"}


def _inject_dirs(llm, vecs):
    for L in llm._steer:
        llm._steer[L] = None
    if vecs is not None:
        for L in llm.layers:
            llm._steer[L] = _clamp_note(vecs[L])


def generate(args):
    print("⏳ judge + Llama...")
    hz = Humanizer(device="cpu")
    llm = SteeredLlama(); llm.alpha = args.alpha
    names = hz.dim_names
    concepts = [("none", None)] + [(names[d].split("_", 1)[1], d) for d in args.concepts]
    label_of = {"none": "none"}
    for nm, d in concepts[1:]:
        label_of[nm] = "divinity" if d == 90 else "consciousness" if d == 23 else nm
    prompts = PROMPTS[:args.prompts]
    print(f"✅ {len(prompts)} prompts × {len(concepts)} conditions "
          f"= {len(prompts)*len(concepts)} texts | alpha {args.alpha}\n")

    trials = []
    for pi, ph in enumerate(prompts):
        for cond_name, d in concepts:
            _inject_dirs(llm, None if d is None else {L: llm.cv[L][d] for L in llm.layers})
            txt = llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)
            prof = hz.profile(txt)
            truth = "none" if d is None else label_of[cond_name]
            trials.append({"prompt_i": pi, "truth": truth, "text": txt,
                           "read_div": float(prof[90]), "read_cons": float(prof[23])})
        print(f"  prompt {pi} done")
    llm.clear()

    # judge's blind guess: standardize each concept-reading across all texts,
    # pick the concept whose z is highest if it clears a small margin, else none.
    rd = np.array([t["read_div"] for t in trials])
    rc = np.array([t["read_cons"] for t in trials])
    zd = (rd - rd.mean()) / (rd.std() + 1e-9)
    zc = (rc - rc.mean()) / (rc.std() + 1e-9)
    for i, t in enumerate(trials):
        if max(zd[i], zc[i]) < 0.5:
            t["judge"] = "none"
        else:
            t["judge"] = "divinity" if zd[i] >= zc[i] else "consciousness"

    order = list(range(len(trials)))
    random.Random(args.seed).shuffle(order)
    KEY.write_text(json.dumps({"order": order, "trials": trials}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    lines = ["# Blind worksheet — guess what was steered", "",
             "For each text write one of: **none / divinity / consciousness**. "
             "No peeking at blind_key.json.", ""]
    for slot, idx in enumerate(order, 1):
        lines += [f"## T{slot:02d}", "", trials[idx]["text"].strip(), "",
                  f"`T{slot:02d} = ____`", ""]
    lines += ["---", "When done, run e.g.:", "",
              '`python -m steering.blind_duel --score "T01 div, T02 none, T03 cons, ..."`']
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📝 worksheet -> {SHEET.name}   (key hidden in {KEY.name})")
    print("   Read it, write your guesses, then re-run with --score.")


def score(answer_str):
    data = json.loads(KEY.read_text(encoding="utf-8"))
    order, trials = data["order"], data["trials"]
    # parse "T01 div, T02 none, ..."
    human = {}
    cur = None
    for tok in answer_str.replace(",", " ").replace("=", " ").split():
        tok = tok.strip().lower()
        if tok.startswith("t") and tok[1:].isdigit():
            cur = int(tok[1:])
        elif tok in SHORT and cur is not None:
            human[cur] = SHORT[tok]
            cur = None
    hok = jok = n = 0
    print(f"\n{'slot':<5}{'truth':<15}{'human':<15}{'judge':<15}")
    for slot, idx in enumerate(order, 1):
        t = trials[idx]; tr = t["truth"]; hu = human.get(slot, "?"); ju = t["judge"]
        n += 1; hok += (hu == tr); jok += (ju == tr)
        mark = lambda g: "✓" if g == tr else "✗"
        print(f"T{slot:<4}{tr:<15}{hu+' '+mark(hu):<15}{ju+' '+mark(ju):<15}")
    print(f"\n  HUMAN accuracy: {hok}/{n} = {100*hok/n:.0f}%")
    print(f"  JUDGE accuracy: {jok}/{n} = {100*jok/n:.0f}%   (chance ~33%)")
    if hok > jok + 1:
        print("  -> Human reads it, judge doesn't: STEERING REAL, JUDGE BROKEN.")
    elif jok >= hok and jok > n / 2:
        print("  -> Judge keeps up: the readout works; the null failure was the metric.")
    else:
        print("  -> Neither separates it: steering weak/empty for these dims.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=str, default=None,
                    help='your guesses, e.g. "T01 div, T02 none, T03 cons"')
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--concepts", type=int, nargs="+", default=[90, 23])
    ap.add_argument("--tokens", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.score is not None:
        score(args.score)
    else:
        generate(args)


if __name__ == "__main__":
    main()
