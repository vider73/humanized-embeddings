"""
night_run.py — Night shift. Chains the complete pipeline unattended:

  1. generate_stimuli   rich phrases per pole (resumable, skips what's done)
  2. backup             saves the current vectors as *_lexical.bak.npy
  3. derive_vectors     re-derives CAA+whitening with the new stimuli
  4. geometry           measures effective rank of the new file
  5. fidelity           closed loop: does each dial move ITS dimension?

Each step runs in a SEPARATE PROCESS: VRAM is freed between steps and one
failure doesn't drag down the rest. Everything lands in vectors/night_YYYYMMDD_HHMM.log.
Keeps the PC awake (Windows) while it works.

  python -m steering.night_run
  python -m steering.night_run --skip stimuli      # if already generated
  python -m steering.night_run --skip stimuli derive
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from . import config

LOG = config.VEC_DIR / f"night_{datetime.now():%Y%m%d_%H%M}.log"


def _log(line):
    stamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[{stamp}] {line}"
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _keep_awake():
    """Keeps Windows from falling asleep mid-job. Harmless on other OSes."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        _log("☕ modo insomne activado (Windows no se dormira)")
    except Exception:
        pass


def run_step(name, module, *extra):
    """Launches `python -m module` as a child process, streaming its output to
    the log and to the console live. Returns True if it finished cleanly."""
    _log(f"▶ PASO {name}: python -m {module} {' '.join(extra)}")
    t0 = time.time()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, "-m", module, *extra],
                         cwd=config.ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    with open(LOG, "a", encoding="utf-8") as f:
        for line in p.stdout:
            print(f"    {line}", end="", flush=True)
            f.write(line)
    p.wait()
    mins = (time.time() - t0) / 60
    ok = p.returncode == 0
    _log(f"{'✅' if ok else '❌'} PASO {name} {'OK' if ok else f'FALLO (rc={p.returncode})'} en {mins:.1f} min")
    return ok


def stimuli_coverage():
    """How many dims have rich phrases? To decide whether deriving makes sense."""
    dim_names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    store = (json.loads(config.STIMULI_FILE.read_text(encoding="utf-8"))
             if config.STIMULI_FILE.exists() else {})
    full = sum(1 for d in dim_names
               if store.get(d, {}).get("pos") and store.get(d, {}).get("neg"))
    return full, len(dim_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="+", default=[],
                    choices=["stimuli", "derive", "geometry", "fidelity"],
                    help="pasos a saltar")
    ap.add_argument("--fidelity-alpha", type=float, default=0.15,
                    help="alpha para el paso de fidelity (calibralo antes en d090)")
    args = ap.parse_args()
    skip = set(args.skip)

    _log(f"🌙 TURNO DE NOCHE — log: {LOG}")
    _keep_awake()
    t0 = time.time()

    # 1 — stimuli (resumable on its own)
    if "stimuli" not in skip:
        run_step("stimuli", "steering.generate_stimuli")
    full, total = stimuli_coverage()
    _log(f"📊 cobertura de stimuli: {full}/{total} dims con frases ricas")
    if full < total:
        _log(f"⚠ {total - full} dims derivarian con moldes LEXICOS (fallback)")

    # 2+3 — backup and re-derivation
    if "derive" not in skip:
        if full == 0:
            _log("❌ sin stimuli no tiene sentido re-derivar. Abortando derive.")
        else:
            cvf = config.CONTROL_VECTORS_FILE
            if cvf.exists():
                bak = cvf.with_suffix(".lexical.bak.npy")
                shutil.copy2(cvf, bak)
                _log(f"💾 backup de vectores previos -> {bak.name}")
            if not run_step("derive", "steering.derive_vectors"):
                _log("❌ derive fallo: geometry/fidelity medirian vectores viejos. FIN.")
                _log(f"🏁 total {((time.time()-t0)/3600):.1f} h")
                return

    # 4 — geometry of the new file
    if "geometry" not in skip:
        run_step("geometry", "steering.geometry")

    # 5 — fidelity (the main course)
    if "fidelity" not in skip:
        # if we re-derived, the previous measurements belong to OTHER vectors
        # (same file name): set them aside so it doesn't "resume" them.
        rep = config.VEC_DIR / "fidelity_report.json"
        if "derive" not in skip and rep.exists():
            bak = rep.with_name(f"fidelity_report_{datetime.now():%Y%m%d_%H%M}.json")
            rep.rename(bak)
            _log(f"📁 report previo apartado -> {bak.name} (vectores nuevos, medicion limpia)")
        run_step("fidelity", "steering.fidelity",
                 "--alpha", str(args.fidelity_alpha))

    _log(f"🏁 TURNO COMPLETO en {((time.time()-t0)/3600):.1f} h. "
         f"Desayuno: el log, geometry y vectors/fidelity_report.json")


if __name__ == "__main__":
    main()
