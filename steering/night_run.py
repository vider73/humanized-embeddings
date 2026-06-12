"""
night_run.py — Turno de noche. Encadena el pipeline completo sin vigilancia:

  1. generate_stimuli   frases ricas por polo (reanudable, se salta lo hecho)
  2. backup             guarda los vectores actuales como *_lexical.bak.npy
  3. derive_vectors     re-deriva CAA+blanqueo con los stimuli nuevos
  4. geometry           mide rango efectivo del fichero nuevo
  5. fidelity           bucle cerrado: ¿cada dial mueve SU dimension?

Cada paso corre en un PROCESO SEPARADO: la VRAM se libera entre pasos y un
fallo no arrastra al resto. Todo queda en vectors/night_YYYYMMDD_HHMM.log.
Mantiene el PC despierto (Windows) mientras trabaja.

  python -m steering.night_run
  python -m steering.night_run --skip stimuli      # si ya estan generados
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
    """Evita que Windows se duerma a mitad de faena. Inocuo en otros SO."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        _log("☕ modo insomne activado (Windows no se dormira)")
    except Exception:
        pass


def run_step(name, module, *extra):
    """Lanza `python -m module` como proceso hijo, volcando su salida al log
    y a consola en vivo. Devuelve True si acabo bien."""
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
    """¿Cuantas dims tienen frases ricas? Para decidir si derivar tiene sentido."""
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

    # 1 — stimuli (reanudable por si misma)
    if "stimuli" not in skip:
        run_step("stimuli", "steering.generate_stimuli")
    full, total = stimuli_coverage()
    _log(f"📊 cobertura de stimuli: {full}/{total} dims con frases ricas")
    if full < total:
        _log(f"⚠ {total - full} dims derivarian con moldes LEXICOS (fallback)")

    # 2+3 — backup y re-derivacion
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

    # 4 — geometria del fichero nuevo
    if "geometry" not in skip:
        run_step("geometry", "steering.geometry")

    # 5 — fidelidad (el plato fuerte)
    if "fidelity" not in skip:
        # si se re-derivo, las mediciones previas son de OTROS vectores
        # (mismo nombre de fichero): apartarlas para que no las "resuma".
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
