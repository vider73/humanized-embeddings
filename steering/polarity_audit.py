"""
polarity_audit.py — Is there a POLARITY BUG in the control vectors, or is a
negative `sign_ok` just sub-threshold noise read at a single low alpha?

Background. fidelity.py defines  sign_ok = (effect[di] > 0), i.e. "pushing the
dial UP makes the judge read MORE of that dim". At alpha=0.15 roughly half the
dims come out negative, which LOOKS like half the vectors are reversed. This
script decides whether that is real.

A GENUINE sign-flip (poles swapped in caa_stimuli.json, or a reversed vector)
would be:
  - alpha-stable : wrong at every alpha — scaling a reversed vector reverses
                   HARDER, it never self-corrects.
  - model-stable : wrong in every model — the stimuli are shared across the
                   ladder, so a swapped pair poisons 1B, 3B and 8B alike.
  - geometric    : a unit vector that is literally the negative of where it
                   should point (or a degenerate / collapsed set).
Noise / sub-threshold sign is the opposite: it FLIPS with alpha, is MIXED
across models, and concentrates on the weak-|effect| dims.

CPU-only. Reads the fidelity_*.json reports and control_vectors_*.npy already
on disk and computes; it writes and changes nothing.

  python -m steering.polarity_audit
"""
import json
import re

import numpy as np

from . import config

VEC = config.VEC_DIR
STRONG = 0.10   # |effect| considered a real movement
WEAK = 0.05     # |effect| below this is essentially noise


def _load(name):
    p = VEC / name
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("_meta", None)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _meta(name):
    p = VEC / name
    return json.loads(p.read_text(encoding="utf-8")).get("_meta", {}) if p.exists() else {}


def _eff(rep, k):
    return rep[k]["effect"] if (rep and k in rep) else None


def section(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def audit_across_alpha():
    """8B sweep: does the sign flip when we change the pressure?"""
    a = {x: _load(f"fidelity_a{str(x).replace('.', '')}.json") for x in (0.15, 0.25, 0.35)}
    a = {x: r for x, r in a.items() if r}
    if len(a) < 2:
        print("  (need >=2 of fidelity_a015/a025/a035.json — 8B sweep absent, skipping)")
        return
    lo, hi = min(a), max(a)
    keys = set(a[lo]) & set(a[hi])
    flips = sum((a[lo][k]["effect"] > 0) != (a[hi][k]["effect"] > 0) for k in keys)
    print(f"  8B, alpha {lo} vs {hi}: {flips}/{len(keys)} dims CHANGE SIGN with pressure.")
    print("  A reversed vector cannot flip with alpha -> flips = threshold/noise, not polarity.")


def audit_across_models():
    """@0.15: do the three models agree on sign? Shared stimuli => a real swap
    would make all three negative together."""
    m = {"8B": _load("fidelity_a015.json"),
         "3B": _load("fidelity_3b.json"),
         "1B": _load("fidelity_1b.json")}
    m = {k: r for k, r in m.items() if r}
    if len(m) < 2:
        print("  (need >=2 models @0.15, skipping)")
        return
    keys = set.intersection(*(set(r) for r in m.values()))
    allpos = allneg = mixed = 0
    stable_neg = []
    for k in keys:
        signs = [m[t][k]["effect"] > 0 for t in m]
        mag = max(abs(m[t][k]["effect"]) for t in m)
        if all(signs):
            allpos += 1
        elif not any(signs):
            allneg += 1
            stable_neg.append((mag, k, {t: m[t][k]["effect"] for t in m}))
        else:
            mixed += 1
    print(f"  models {list(m)} @0.15, {len(keys)} shared dims:")
    print(f"    all positive: {allpos}   all negative: {allneg}   MIXED: {mixed}")
    print("    (mixed = sign is model-specific = noise; a shared-stimuli swap "
          "would force all-negative)")
    real = [s for s in stable_neg if s[0] >= WEAK]
    print(f"\n  stable-NEGATIVE across all models AND |effect|>={WEAK} "
          f"(true swap candidates): {len(real)}")
    for mag, k, effs in sorted(real, reverse=True):
        print("    " + k + "  " + " ".join(f"{t}{e:+.3f}" for t, e in effs.items()))
    if not real:
        print("    none — every stable-negative dim is judge-blind (|effect|~0).")


def audit_strength():
    """If sign were a vector bug, STRONG dims would agree across models. Do they?"""
    m = {"8B": _load("fidelity_a015.json"),
         "3B": _load("fidelity_3b.json"),
         "1B": _load("fidelity_1b.json")}
    m = {k: r for k, r in m.items() if r}
    if len(m) < 2:
        return
    keys = set.intersection(*(set(r) for r in m.values()))
    def agree(k):
        s = [m[t][k]["effect"] > 0 for t in m]
        return all(s) or not any(s)
    strong = [k for k in keys if max(abs(m[t][k]["effect"]) for t in m) >= STRONG]
    weak = [k for k in keys if max(abs(m[t][k]["effect"]) for t in m) < WEAK]
    if strong:
        print(f"  strong dims (max|effect|>={STRONG}): {sum(agree(k) for k in strong)}"
              f"/{len(strong)} agree on sign across models")
    if weak:
        print(f"  weak dims  (max|effect|<{WEAK}):  {sum(agree(k) for k in weak)}"
              f"/{len(weak)} agree (≈coinflip if pure noise)")


def audit_vectors():
    """Geometric health of each derived vector set on disk."""
    for p in sorted(VEC.glob("control_vectors_caa_white*.npy")):
        if ".bak" in p.name:
            continue
        v = np.load(p)
        cv = v[0] if v.ndim == 3 else v          # (104, H) — any layer slice
        norms = np.linalg.norm(cv, axis=1)
        s = np.linalg.svd(cv, compute_uv=False)
        er = float((s.sum() ** 2) / (s ** 2).sum())
        C = cv / (norms[:, None] + 1e-12)
        G = C @ C.T
        np.fill_diagonal(G, 0.0)
        print(f"  {p.name:<38} shape{tuple(v.shape)}  "
              f"norm[{norms.min():.2f},{norms.max():.2f}]  "
              f"eff.rank {er:.0f}/{cv.shape[0]}  most-antialigned cos {G.min():+.2f}")
    print("  (norms~1, eff.rank>>1, no pair near cos -1 => no reversed/collapsed vectors)")


def audit_stimuli(report="fidelity_3b.json"):
    """Spot-check pole assignment for the stable-negative + a known-good pair."""
    sp = VEC / "caa_stimuli.json"
    if not sp.exists():
        print("  (caa_stimuli.json absent, skipping)")
        return
    stim = json.loads(sp.read_text(encoding="utf-8"))
    rep = _load(report) or {}
    cand = sorted(((v["effect"], k) for k, v in rep.items()
                   if v["effect"] < 0 and abs(v["effect"]) >= WEAK))[:4]
    show, seen = [], set()
    for k in [k for _, k in cand] + ["d014_fragilidad", "d015_solidez"]:
        if k not in seen:
            seen.add(k); show.append(k)
    for k in show:
        s = stim.get(k, {})
        pos = (s.get("pos") or ["—"])[0]
        neg = (s.get("neg") or ["—"])[0]
        print(f"  {k}\n     POS(max): {pos[:88]}\n     NEG(min): {neg[:88]}")
    print("\n  Watch the NEG lines: a clean contrast needs the LOW pole, but some "
          "still\n  name/affirm the concept (weak contrast -> weak vector -> "
          "sub-threshold dial).\n  That is a stimulus-quality lead (v3 probes), not a sign bug.")


def main():
    print("POLARITY AUDIT — control vectors")
    print(f"vectors dir: {VEC}")
    section("1) Sign vs ALPHA (8B) — a real reversal cannot flip with pressure")
    audit_across_alpha()
    section("2) Sign vs MODEL @0.15 — shared stimuli: a real swap hits all models")
    audit_across_models()
    section("3) Strong vs weak — a vector bug would still show on strong dims")
    audit_strength()
    section("4) Vector geometry — reversed/collapsed sets show here")
    audit_vectors()
    section("5) Stimulus poles — is POS the high pole and NEG the low pole?")
    audit_stimuli()
    section("VERDICT")
    print("""  Read the four tests together. If sign flips with alpha (1), is mixed
  across models (2), strong dims disagree (3), vectors are unit-norm and
  full-rank with no anti-aligned twins (4), and the spot-checked poles are
  correctly assigned (5) -> there is NO global polarity bug. The 50/50
  sign split at a single low alpha is the SIGN OF NOISE: most dials are
  simply sub-threshold at that pressure. Fix the operating point (per-dim
  alpha, via the tuner) before declaring any dial inverted or dead. Do NOT
  flip vector signs.""")


if __name__ == "__main__":
    main()
