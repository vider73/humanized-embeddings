"""
analyze.py — Scorecard final a partir de fidelity_report.json (sin GPU).

Corrige el sesgo del juez SIN matar a las familias: para cada dim, el modo
comun se estima solo sobre dims NO emparentadas con ella. El parentesco es
objetivo: correlaciones de tu propia tabla humana (dataset_Y). Asi divinidad
no paga por compartir alma con religiosidad y magia.

Metricas por dim:
  z_kin    : z del efecto propio tras restar el modo comun de NO-parientes
  rank_kin : cuantas dims NO emparentadas se movieron mas que ella (0 = nadie)
  sig      : el empuje movio su dim en la direccion correcta

  python -m steering.analyze
  python -m steering.analyze --kin-r 0.5 --z-real 1.5
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config

REPORT_FILE = config.VEC_DIR / "fidelity_report.json"
SCORECARD_MD = config.VEC_DIR / "scorecard.md"


def scorecard(report_path=None, kin_r=0.4):
    """Calcula el scorecard kin-aware de un report de fidelity.
    Devuelve lista de dicts ordenada por z_kin desc. Sin GPU."""
    rep = json.loads(Path(report_path or REPORT_FILE).read_text(encoding="utf-8"))
    rows = {k: v for k, v in rep.items()
            if not k.startswith("_") and "effect_vec" in v}
    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    dim_names = meta["dimension_names"]
    missing = [n for n in dim_names if n not in rows]
    if missing:
        print(f"⚠ faltan {len(missing)} dims en el report")
    used = [n for n in dim_names if n in rows]
    pos = {n: i for i, n in enumerate(dim_names)}

    E = np.array([rows[n]["effect_vec"] for n in used])        # (m, 104)

    Y = np.load(config.DATA_Y_FILE)                            # (N, 104)
    R = np.corrcoef(Y.T)
    KIN = np.abs(R) > kin_r

    out = []
    for r, n in enumerate(used):
        i = pos[n]
        nonkin_rows = [rr for rr, nn in enumerate(used)
                       if not KIN[i, pos[nn]] or nn == n]
        nonkin_rows = [rr for rr in nonkin_rows if used[rr] != n]
        cm = E[nonkin_rows].mean(0) if nonkin_rows else np.zeros(E.shape[1])
        ec = E[r] - cm
        own = float(ec[i])
        others = np.delete(ec, i)
        z = float((own - others.mean()) / (others.std() + 1e-9))
        ahead = [j for j in range(len(dim_names))
                 if j != i and not KIN[i, j] and abs(ec[j]) > abs(own)]
        kin_n = int(KIN[i].sum() - 1)
        out.append(dict(name=n, z_kin=round(z, 2), rank_kin=len(ahead),
                        sign_ok=own > 0, kin=kin_n,
                        domain=rows[n].get("domain", "?")))

    out.sort(key=lambda d: -d["z_kin"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin-r", type=float, default=0.4,
                    help="|r| de la tabla humana a partir del cual dos dims son familia")
    ap.add_argument("--z-real", type=float, default=1.5,
                    help="umbral de z_kin para declarar un dial REAL")
    ap.add_argument("--report", default=str(REPORT_FILE))
    args = ap.parse_args()

    out = scorecard(args.report, args.kin_r)
    reales = [d for d in out if d["z_kin"] >= args.z_real and d["sign_ok"]]
    r0 = [d for d in out if d["rank_kin"] == 0 and d["sign_ok"]]

    print(f"=== SCORECARD (kin |r|>{args.kin_r}) ===")
    print(f"diales REALES (z_kin>={args.z_real} y signo OK): {len(reales)}/{len(out)}")
    print(f"rank_kin=0 con signo OK: {len(r0)}\n")
    print(f"{'dim':<32} {'z_kin':>6} {'rank':>4} {'sig':>4} {'fam':>4}  dominio")
    for d in out:
        tag = "🟢" if (d["z_kin"] >= args.z_real and d["sign_ok"]) else (
              "🟡" if (d["z_kin"] >= 1.0 and d["sign_ok"]) else "🔴")
        print(f"{tag} {d['name']:<30} {d['z_kin']:+6.1f} {d['rank_kin']:>4} "
              f"{'OK' if d['sign_ok'] else 'INV':>4} {d['kin']:>4}  {d['domain']}")

    # scorecard.md para el repo
    lines = ["# Scorecard de fidelidad — diales reales de la consola",
             "",
             f"Criterio: z_kin ≥ {args.z_real} con signo correcto. "
             f"Parentesco: |r| > {args.kin_r} en la tabla humana.",
             "",
             f"**{len(reales)} diales reales de {len(out)} medidos.**",
             "",
             "| dim | z_kin | rank_kin | signo | familia | dominio |",
             "|---|---|---|---|---|---|"]
    for d in out:
        emoji = "🟢" if (d["z_kin"] >= args.z_real and d["sign_ok"]) else (
                "🟡" if (d["z_kin"] >= 1.0 and d["sign_ok"]) else "🔴")
        lines.append(f"| {emoji} {d['name']} | {d['z_kin']:+.1f} | {d['rank_kin']} "
                     f"| {'OK' if d['sign_ok'] else 'INV'} | {d['kin']} | {d['domain']} |")
    SCORECARD_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nscorecard -> {SCORECARD_MD}")


if __name__ == "__main__":
    main()
