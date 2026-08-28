"""
null_control.py — Are we measuring real structure, or does the method
manufacture dials? (José's 2026-06-14 question: "¿estamos forzando un resultado?")

The falsification. We keep the EXACT pipeline — same model, layer, whitening,
clamp, judge — and change ONE thing: each dimension is derived from ANOTHER
dimension's contrastive stimuli (a derangement: no dim keeps its own). The
vector in slot d_i now points at concept σ(i), not i. Fidelity still asks
"did dim i move?". So:

  · If the method MEASURES something, slot i no longer moves dim i -> the
    green-rate and the rank-0 rate COLLAPSE toward chance. The instrument
    discriminates real content from same-shaped noise.
  · If the method FORCES results (any strong push lights up the judge's
    reading of whatever dim we name), the null greens at ~the real rate.
    Then "25 robust dials" is just the method's baseline hit-rate and the
    whole scorecard is suspect.

A forced result has no failures. The real run fails ~75% of the time; the
honest test is whether the SHUFFLED run fails ~everywhere.

Reuses derive + fidelity as subprocesses (identical methodology, like
periodic_run) via config's EMB_TAG / EMB_STIMULI overrides. Default regime =
the 8B reference (no EMB_LLM/LAYERS set), compared against fidelity_aNNN.json.

  python -m steering.null_control                 # alpha 0.25, seed 0
  python -m steering.null_control --alphas 0.15 0.25 0.35
  python -m steering.null_control --analyze-only  # just re-print the comparison
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from . import config

LOG = config.VEC_DIR / f"null_{datetime.now():%Y%m%d_%H%M}.log"


def _log(line):
    msg = f"[{datetime.now():%H:%M:%S}] {line}"
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        _log("☕ insomniac mode on (Windows will not sleep)")
    except Exception:
        pass


def _loadj(p, tries=8):
    """Tolerant loader (network/host mounts can serve truncated reads)."""
    for _ in range(tries):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(0.4)
    raise RuntimeError(f"could not read {p}")


def _derangement(keys, seed):
    """A permutation of keys with NO fixed point (no dim keeps its stimuli)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    while True:
        perm = rng.permutation(len(keys))
        if not np.any(perm == idx):           # zero fixed points
            return {keys[i]: keys[perm[i]] for i in idx}


def build_shuffled(tag, seed):
    """Write a stimuli file where each dim carries another dim's sentences."""
    stim = _loadj(config.VEC_DIR / "caa_stimuli.json")
    have = [k for k, v in stim.items() if v.get("pos") and v.get("neg")]
    sigma = _derangement(have, seed)
    shuffled = {k: stim[sigma[k]] for k in have}     # slot k <- sentences of sigma[k]
    out = config.VEC_DIR / f"caa_stimuli_{tag}.json"
    out.write_text(json.dumps(shuffled, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"🔀 shuffled stimuli: {len(have)} dims deranged (seed={seed}) -> {out.name}")
    self_kept = sum(1 for k in have if sigma[k] == k)
    _log(f"   sanity: {self_kept} dims kept their own stimuli (must be 0)")
    return out


def build_random(tag, seed):
    """Random unit-Gaussian vectors carrying NO concept at all — the harder
    null ('does ANY push green the judge?'). Matches the real vectors' shape
    exactly; no derive needed (CPU, instant)."""
    V = np.load(config.VEC_DIR / "control_vectors_caa_white.npy")
    rng = np.random.default_rng(seed)
    R = rng.standard_normal(V.shape).astype(np.float32)
    R /= (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-12)
    out = config.VEC_DIR / f"control_vectors_caa_white_{tag}.npy"
    np.save(out, R)
    meta = config.VEC_DIR / "control_meta_caa_white.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["derive_mode"] = "random_null"
        (config.VEC_DIR / f"control_meta_caa_white_{tag}.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"🎲 random null: {tuple(V.shape)} unit-gaussian vectors (seed={seed}) -> {out.name}")
    return out


def run_step(name, env_extra, module, *extra):
    _log(f"▶ {name}: python -m {module} {' '.join(extra)}")
    t0 = time.time()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", **env_extra}
    p = subprocess.Popen([sys.executable, "-m", module, *extra],
                         cwd=config.ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    with open(LOG, "a", encoding="utf-8") as f:
        for line in p.stdout:
            print(f"    {line}", end="", flush=True)
            f.write(line)
    p.wait()
    _log(f"{'✅' if p.returncode == 0 else '❌'} {name} in {(time.time()-t0)/60:.1f} min")
    return p.returncode == 0


def _stats(rep_path):
    """green / rank-0 / sign-rate / mean|effect| / median-rank from a report."""
    d = _loadj(rep_path)
    rows = {k: v for k, v in d.items() if not k.startswith("_")}
    n = len(rows) or 1
    green = sum(1 for v in rows.values() if v.get("sign_ok") and v.get("z", 0) >= 1.5)
    rank0 = sum(1 for v in rows.values() if v.get("sign_ok") and v.get("rank", 99) == 0)
    sign = sum(1 for v in rows.values() if v.get("sign_ok"))
    mae = float(np.mean([abs(v.get("effect", 0.0)) for v in rows.values()]))
    medrank = float(np.median([v.get("rank", 0) for v in rows.values()]))
    return dict(n=n, green=green, rank0=rank0, sign=sign, mae=mae, medrank=medrank)


def _bestof(paths):
    """Best-of-alphas roll-up: a dim is robust if it is green (or rank-0) at
    ANY of its sign-correct alphas — mirrors the tuner's afinacion logic."""
    reps = [{k: v for k, v in _loadj(p).items() if not k.startswith("_")}
            for p in paths if p.exists()]
    if not reps:
        return None
    keys = set().union(*(set(r) for r in reps))
    green = rank0 = 0
    for k in keys:
        ok = [r[k] for r in reps if k in r and r[k].get("sign_ok")]
        if ok and max(x.get("z", 0) for x in ok) >= 1.5:
            green += 1
        if any(x.get("rank", 99) == 0 for x in ok):
            rank0 += 1
    return dict(n=len(keys), green=green, rank0=rank0)


def analyze(tag, alphas):
    _log(f"\n{'='*68}\n  NULL CONTROL — real vs shuffled-stimuli, same regime\n{'='*68}")
    hdr = f"  {'alpha':<7}{'set':<8}{'green≥1.5':>10}{'rank0':>8}{'sign_ok%':>10}{'mean|eff|':>11}{'medRank':>9}"
    print(hdr)
    any_done = False
    for a in alphas:
        aaa = str(a).replace('.', '')
        real_p = config.VEC_DIR / f"fidelity_a{aaa}.json"
        null_p = config.VEC_DIR / f"fidelity_{tag}_a{aaa}.json"
        for label, p in (("REAL", real_p), ("NULL", null_p)):
            if not p.exists():
                print(f"  {a:<7}{label:<8}{'(missing '+p.name+')':>48}")
                continue
            any_done = True
            s = _stats(p)
            print(f"  {a:<7}{label:<8}{s['green']:>4}/{s['n']:<5}{s['rank0']:>8}"
                  f"{100*s['sign']/s['n']:>9.0f}%{s['mae']:>11.3f}{s['medrank']:>9.0f}")
    if not any_done:
        _log("  no reports yet — run without --analyze-only first.")
        return
    print(f"""
  Read it like this:
    · NULL green ≈ 0 and NULL rank0 ≈ 0  -> the instrument DISCRIMINATES;
      shuffling content kills the dials. We are NOT forcing it.
    · NULL sign_ok% ≈ 50  -> sign is a coin-flip on mismatched vectors (expected).
    · NULL green ≈ REAL green  -> the method manufactures dials; STOP and rethink.
  The honest bar: REAL must beat NULL by a wide, obvious margin, especially on
  rank0 (the strict selectivity criterion — does the push move ITS OWN dim).""")

    if len(alphas) > 1:
        aa = [str(a).replace(".", "") for a in alphas]
        rb = _bestof([config.VEC_DIR / f"fidelity_a{a}.json" for a in aa])
        nb = _bestof([config.VEC_DIR / f"fidelity_{tag}_a{a}.json" for a in aa])
        if rb and nb:
            rg, rr, ng, nr, n = rb["green"], rb["rank0"], nb["green"], nb["rank0"], rb["n"]
            ratio = ng / max(rg, 1)
            verdict = ("DISCRIMINATES - not forcing it" if ratio <= 0.35
                       else "SUSPICIOUS - null greens too high, rethink"
                       if ratio >= 0.7 else "PARTIAL - margin exists but soft")
            print("\n  BEST-OF-{} (a dim robust at ANY of its alphas - headline):"
                  .format(len(alphas)))
            print("    REAL: {:>3} robust, {:>3} rank-0   of {}".format(rg, rr, n))
            print("    NULL: {:>3} robust, {:>3} rank-0   of {}".format(ng, nr, n))
            print("    null/real green ratio = {:.2f}  ->  {}".format(ratio, verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shuffle", "random"], default="shuffle",
                    help="shuffle = deranged real stimuli; random = gaussian noise vectors")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="artifact tag (default: 'null' for shuffle, 'rand' for random)")
    ap.add_argument("--skip-derive", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    tag = args.tag or ("rand" if args.mode == "random" else "null")

    if args.analyze_only:
        analyze(tag, args.alphas)
        return

    _log(f"🧪 NULL CONTROL [{args.mode}] — tag={tag} seed={args.seed} alphas={args.alphas}")
    _keep_awake()
    vec = config.VEC_DIR / f"control_vectors_caa_white_{tag}.npy"
    if args.mode == "random":
        if not (args.skip_derive or vec.exists()):
            build_random(tag, args.seed)
        else:
            _log(f"   random vectors reused ({vec.name})")
        env = {"EMB_TAG": tag}
    else:
        stim_file = build_shuffled(tag, args.seed)
        env = {"EMB_TAG": tag, "EMB_STIMULI": str(stim_file.resolve())}
        if args.skip_derive or vec.exists():
            _log(f"   derive skipped ({vec.name} {'exists' if vec.exists() else 'per flag'})")
        elif not run_step("derive[null]", env, "steering.derive_vectors"):
            _log("   aborted (derive failed)"); return

    for a in args.alphas:
        rep = config.VEC_DIR / f"fidelity_{tag}_a{str(a).replace('.', '')}.json"
        run_step(f"fidelity[null] α={a}", env, "steering.fidelity",
                 "--alpha", str(a), "--report", str(rep))

    analyze(tag, args.alphas)
    _log(f"🏁 null control done. log: {LOG.name}")


if __name__ == "__main__":
    main()
