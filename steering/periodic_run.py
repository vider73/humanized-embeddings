"""
periodic_run.py — NIGHT SHIFT for the periodic table of semantic concepts.

For each model in the ladder, with the SAME stimuli (caa_stimuli.json — the
controlled variable): derive CAA control vectors, run the geometry gate, and
run full fidelity at a fixed alpha. Each step is a separate process with the
model selected via environment overrides (see config.py), so VRAM is freed
between steps and one failure doesn't drag the rest down.

The 8B reference point already exists (untagged artifacts + the tuner's
fidelity_a015.json), so the ladder only runs the smaller rungs:

  tag   model                            layers   inject
  1b    meta-llama/Llama-3.2-1B-Instruct   5-9       7     (16 layers total)
  3b    meta-llama/Llama-3.2-3B-Instruct  11-17     14     (28 layers total)
  smol  HuggingFaceTB/SmolLM2-135M-…      10-14     12     (--with-smol only:
        mostly English-trained; Spanish probes are a confound — interpret
        its column with tweezers)

Afterwards: python 58_periodic_table.py   builds PERIODIC.md from whatever
rungs exist.

  python -m steering.periodic_run                # 1b + 3b (overnight-ish)
  python -m steering.periodic_run --only 1b
  python -m steering.periodic_run --with-smol
  python -m steering.periodic_run --alpha 0.15 --skip-derive

Notes: Llama-3.2 models are HF-gated like 3.1 (accept the license once).
Resumable: derive is skipped if the tagged vectors exist; fidelity resumes
per-dim inside its own report.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from . import config

LOG = config.VEC_DIR / f"periodic_{datetime.now():%Y%m%d_%H%M}.log"

LADDER = {
    "1b":   dict(model="meta-llama/Llama-3.2-1B-Instruct", layers="5-9",
                 inject="7", bits4="0"),
    "3b":   dict(model="meta-llama/Llama-3.2-3B-Instruct", layers="11-17",
                 inject="14", bits4="0"),
    "smol": dict(model="HuggingFaceTB/SmolLM2-135M-Instruct", layers="10-14",
                 inject="12", bits4="0"),
}


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
    ok = p.returncode == 0
    _log(f"{'✅' if ok else '❌'} {name} in {(time.time()-t0)/60:.1f} min")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(LADDER),
                    help="run only these rungs")
    ap.add_argument("--with-smol", action="store_true",
                    help="include SmolLM2-135M (English-prior caveat)")
    ap.add_argument("--alpha", default="0.15")
    ap.add_argument("--skip-derive", action="store_true")
    args = ap.parse_args()

    rungs = args.only or (["1b", "3b"] + (["smol"] if args.with_smol else []))
    _keep_awake()
    _log(f"🧪 PERIODIC TABLE night shift — rungs: {rungs} · alpha={args.alpha}")
    _log("   (8B reference = existing untagged vectors + fidelity_a015.json)")

    for tag in rungs:
        spec = LADDER[tag]
        env = {"EMB_TAG": tag, "EMB_LLM": spec["model"],
               "EMB_LAYERS": spec["layers"], "EMB_INJECT": spec["inject"],
               "EMB_4BIT": spec["bits4"]}
        _log(f"━━ rung [{tag}] {spec['model']} · layers {spec['layers']} "
             f"· inject {spec['inject']} ━━")

        vec_file = config.VEC_DIR / f"control_vectors_caa_white_{tag}.npy"
        if args.skip_derive or vec_file.exists():
            _log(f"   derive skipped ({vec_file.name} "
                 f"{'exists' if vec_file.exists() else 'per --skip-derive'})")
        elif not run_step(f"derive[{tag}]", env, "steering.derive_vectors"):
            _log(f"   rung [{tag}] aborted (derive failed)"); continue

        run_step(f"geometry[{tag}]", env, "steering.geometry")
        run_step(f"fidelity[{tag}]", env, "steering.fidelity",
                 "--alpha", args.alpha,
                 "--report", str(config.VEC_DIR / f"fidelity_{tag}.json"))

    _log("🏁 PERIODIC night shift done. Breakfast: python 58_periodic_table.py")


if __name__ == "__main__":
    main()
