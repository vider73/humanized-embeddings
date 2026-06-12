"""
tuner.py — EL AFINADOR. Barre fidelity a varios alphas bajo el regimen
actual (clamp/capa unica) y saca la tabla de afinacion de cada cuerda:
a que presion suena mejor cada una de las 104 dims y cual es su z maximo.

Cada alpha corre en un PROCESO separado (VRAM limpia, fallo no contagia)
y escribe su propio report (fidelity_a015.json, ...). Al final, el merge:
afinacion.md — por dim: z_kin a cada alpha, alpha optimo, veredicto.

Reanudable: fidelity se salta dims ya medidas dentro de cada report.

  python -m steering.tuner                       # alphas 0.15 0.25 0.35
  python -m steering.tuner --alphas 0.2 0.3
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from . import config
from .analyze import scorecard

LOG = config.VEC_DIR / f"tuner_{datetime.now():%Y%m%d_%H%M}.log"
TUNING_MD = config.VEC_DIR / "afinacion.md"


def _log(line):
    msg = f"[{datetime.now():%H:%M:%S}] {line}"
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        _log("☕ modo insomne activado")
    except Exception:
        pass


def run_step(name, module, *extra):
    _log(f"▶ {name}: python -m {module} {' '.join(extra)}")
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
    ok = p.returncode == 0
    _log(f"{'✅' if ok else '❌'} {name} en {(time.time()-t0)/60:.1f} min")
    return ok


def _report_path(a):
    return config.VEC_DIR / f"fidelity_a{str(a).replace('.', '')}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.15, 0.25, 0.35])
    ap.add_argument("--z-real", type=float, default=1.5)
    args = ap.parse_args()

    _log(f"🎸 AFINADOR — alphas={args.alphas} | log: {LOG.name}")
    _keep_awake()
    t0 = time.time()

    for a in args.alphas:
        run_step(f"fidelity α={a}", "steering.fidelity",
                 "--alpha", str(a), "--report", str(_report_path(a)))

    # ── merge: la tabla de afinacion ────────────────────────────────────────
    _log("🧮 calculando afinacion (kin-aware, sin GPU)...")
    per_alpha = {}
    for a in args.alphas:
        p = _report_path(a)
        if not p.exists():
            _log(f"⚠ falta {p.name}, lo salto")
            continue
        per_alpha[a] = {d["name"]: d for d in scorecard(p)}

    if not per_alpha:
        _log("❌ ningun report disponible. FIN.")
        return

    names = sorted({n for byd in per_alpha.values() for n in byd},
                   key=lambda n: n)
    rows = []
    for n in names:
        zs = {a: per_alpha[a][n]["z_kin"] for a in per_alpha if n in per_alpha[a]}
        sg = {a: per_alpha[a][n]["sign_ok"] for a in per_alpha if n in per_alpha[a]}
        ok_zs = {a: z for a, z in zs.items() if sg[a]}
        best_a, best_z = (max(ok_zs.items(), key=lambda kv: kv[1])
                          if ok_zs else (None, max(zs.values())))
        dom = next(iter(per_alpha.values()))[n]["domain"] if n in next(iter(per_alpha.values())) else "?"
        rows.append((n, dom, zs, best_a, best_z))

    rows.sort(key=lambda r: -r[4])
    reales = [r for r in rows if r[4] >= args.z_real and r[3] is not None]

    alpha_cols = sorted(per_alpha)
    head = f"{'dim':<32} {'dom':<9} " + " ".join(f"z@{a:<4}" for a in alpha_cols) + "  mejor"
    _log(f"\n=== AFINACION: {len(reales)}/{len(rows)} cuerdas reales en su alpha optimo ===")
    print(head)
    for n, dom, zs, ba, bz in rows[:40]:
        zcells = " ".join(f"{zs.get(a, float('nan')):+5.1f}" for a in alpha_cols)
        mark = "🟢" if (bz >= args.z_real and ba is not None) else "🔴"
        print(f"{mark} {n:<30} {dom:<9} {zcells}  α={ba} (z={bz:+.1f})")

    lines = ["# Afinación de la consola — alpha óptimo por cuerda", "",
             f"Régimen: {config.STEER_MODE}, capas {list(config.INJECT_LAYERS)}, "
             f"alphas barridos: {alpha_cols}.",
             f"**{len(reales)}/{len(rows)} cuerdas reales (z_kin≥{args.z_real} en su mejor alpha).**",
             "",
             "| dim | dominio | " + " | ".join(f"z@{a}" for a in alpha_cols) +
             " | α óptimo | z máx |", "|---|---|" + "---|" * (len(alpha_cols) + 2)]
    for n, dom, zs, ba, bz in rows:
        emoji = "🟢" if (bz >= args.z_real and ba is not None) else "🔴"
        zcells = " | ".join(f"{zs.get(a, float('nan')):+.1f}" for a in alpha_cols)
        lines.append(f"| {emoji} {n} | {dom} | {zcells} | {ba} | {bz:+.1f} |")
    TUNING_MD.write_text("\n".join(lines), encoding="utf-8")

    _log(f"🏁 AFINADOR completo en {(time.time()-t0)/3600:.1f} h. "
         f"Desayuno: {TUNING_MD.name} + {LOG.name}")


if __name__ == "__main__":
    main()
