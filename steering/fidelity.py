"""
fidelity.py — El test ciego industrializado, v2 con sondas por dominio.

Mide, dim a dim, si el steering mueve el texto en la direccion HUMANA
correcta. El juez es tu propio Humanizer:

  Para cada dimension d_i:
    1. genera con d_i empujada ARRIBA (0.95) y ABAJO (0.05), aislada, greedy
    2. pasa cada texto por el traductor texto->perfil 104
    3. effect_i = perfil(arriba) - perfil(abajo), en TODAS las dims
    4. fidelidad = ¿la dim que mas se movio es la que empujaste?

v2: cada dim se sonda con frases de SU dominio. "Hablame de un dia
cualquiera" no le da espacio a la radiactividad para expresarse — la sonda
era ciega, no el dial muerto. Ademas se guarda el vector de efecto COMPLETO
(104 floats) para poder restar el modo comun del juez (salud/necesidad/...
se mueven con cualquier texto) y recalcular ranks corregidos.

  z alto + rank 0  -> dial REAL (el empuje mueve su propia dim mas que nada)
  z ~0             -> dial muerto, o sesgo del juez (mira el rank corregido)

Reanudable: guarda incremental en fidelity_report.json y se salta lo hecho.
Si el report existente es de otra version de sondas u otros vectores, lo
aparta solo y empieza limpio.

  python -m steering.fidelity
  python -m steering.fidelity --alpha 0.2 --phrases 3
  python -m steering.fidelity --only 19 4 7 --show-text
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .translator import Humanizer
from .steering_model import SteeredLlama

REPORT_FILE = config.VEC_DIR / "fidelity_report.json"
PROBE_VERSION = 2


def _vectors_sha():
    """sha256 (16 hex) del CONTENIDO de los control vectors. El nombre del
    fichero no basta: una re-derivacion escribe el mismo nombre con otros
    vectores y los reports dejan de ser comparables sin que nadie lo note
    (la anomalia scorecard-manana vs tuner-tarde del 2026-06-11)."""
    import hashlib
    return hashlib.sha256(config.CONTROL_VECTORS_FILE.read_bytes()).hexdigest()[:16]

# ── Sondas por dominio ──────────────────────────────────────────────────────
# Frases abiertas que dan ESPACIO a las dims de ese territorio para
# expresarse. Greedy + misma frase para todos los polos => solo cambia el dial.
PROBES = {
    "materia": (
        "Describe un objeto que tengas cerca, con todo detalle.",
        "Describe un material y como se comporta al tocarlo y usarlo.",
        "Describe un fenomeno natural que estes observando.",
    ),
    "vida": (
        "Describe un ser vivo que te llame la atencion.",
        "Hablame de una manana en el campo.",
        "Describe un jardin y lo que crece en el.",
    ),
    "mente": (
        "Hablame de un dia cualquiera.",
        "Cuentame como se siente alguien al volver a casa por la noche.",
        "Describe lo que pasa por la mente de una persona en silencio.",
    ),
    "sociedad": (
        "Describe una escena en una calle de tu ciudad.",
        "Cuentame una historia breve sobre una comunidad y sus costumbres.",
        "Hablame de un dia cualquiera.",
    ),
    "sentidos": (
        "Describe una comida de domingo, con detalle.",
        "Describe los sonidos, olores y sensaciones de un lugar.",
        "Describe un objeto que tengas cerca, con todo detalle.",
    ),
}

# primer match gana; sin match -> "mente" (las narrativas son el comodin)
_DOMAIN_KEYS = (
    ("sentidos", ("sabor", "táctil", "tactil", "sonora", "olfat", "olor",
                  "sonoridad", "saturación", "brillo", "color", "sensorial")),
    ("materia", ("tamaño", "masa", "densidad", "velocidad_máxima", "temperatura",
                 "duración", "distancia", "gravedad", "luminosidad", "transparencia",
                 "dureza", "elasticidad", "viscosidad", "fricción", "fragilidad",
                 "solidez", "volatilidad", "conductividad", "magnetismo",
                 "radiactividad", "porosidad", "maleabilidad", "rugosidad",
                 "pegajosidad", "caos", "frecuencia", "durabilidad",
                 "humedad", "toxicidad", "química")),
    ("vida", ("vitalidad", "salud", "fertilidad", "evolución", "edad",
              "nutricional", "instinto")),
    ("sociedad", ("tradición", "legalidad", "formalidad", "urbanismo",
                  "industrial", "lujo", "religios", "divinidad", "magia",
                  "misterio", "futurismo", "artificialidad", "peligrosidad",
                  "belleza", "económico", "verdad", "probabilidad",
                  "dificultad", "necesidad")),
)


def _domain(name):
    low = name.lower()
    for dom, keys in _DOMAIN_KEYS:
        if any(k in low for k in keys):
            return dom
    return "mente"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--tokens", type=int, default=110)
    ap.add_argument("--phrases", type=int, default=2,
                    help="nº de sondas por dim (mas = mas robusto, mas lento)")
    ap.add_argument("--only", type=int, nargs="+", help="solo estas dims (indice)")
    ap.add_argument("--redo", action="store_true", help="repite dims ya medidas")
    ap.add_argument("--show-text", action="store_true",
                    help="imprime los textos generados (autopsia a ojo)")
    ap.add_argument("--report", default=str(REPORT_FILE),
                    help="fichero de salida (para barridos de alpha del afinador)")
    args = ap.parse_args()
    n_probes = max(1, min(args.phrases, 3))
    rf = Path(args.report)
    vec_sha = _vectors_sha()

    print("⏳ Cargando humanizador (CPU, no toca VRAM)...")
    hz = Humanizer(device="cpu")
    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    llm.alpha = args.alpha
    n = hz.human_dim
    names = hz.dim_names
    print(f"✅ Listo. capas={llm.layers} alpha={args.alpha} sondas/dim={n_probes} "
          f"vectores={config.CONTROL_VECTORS_FILE.name}\n")

    # reanudar — pero solo si el report es comparable (mismas sondas y vectores)
    report = {}
    if rf.exists():
        report = json.loads(rf.read_text(encoding="utf-8"))
        meta = report.get("_meta", {})
        if (meta.get("probe_version") != PROBE_VERSION
                or meta.get("vectors") != config.CONTROL_VECTORS_FILE.name
                or meta.get("vectors_sha") != vec_sha
                or meta.get("mode") != llm.mode
                or meta.get("layers") != list(llm.layers)
                or meta.get("alpha") != args.alpha):
            bak = rf.with_name(
                f"{rf.stem}_old_{datetime.now():%Y%m%d_%H%M}.json")
            rf.rename(bak)
            print(f"📁 report previo incompatible apartado -> {bak.name}")
            report = {}
    report["_meta"] = {
        "probe_version": PROBE_VERSION, "alpha": args.alpha,
        "hi": args.hi, "lo": args.lo, "tokens": args.tokens,
        "n_probes": n_probes, "vectors": config.CONTROL_VECTORS_FILE.name,
        "vectors_sha": vec_sha,
        "mode": llm.mode, "layers": list(llm.layers),
    }

    # neutrales: perezosos y cacheados (greedy => deterministas por frase)
    neutral = {}

    def _neutral(ph):
        if ph not in neutral:
            llm.clear()
            txt = llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)
            if args.show_text:
                print(f"\n--- NEUTRAL · «{ph}» ---\n{txt}")
            neutral[ph] = hz.profile(txt)
        return neutral[ph]

    targets = args.only or range(n)
    for di in targets:
        key = names[di]
        if not args.redo and key in report:
            print(f"✓ ya medida: {key}")
            continue
        dom = _domain(key)
        phrases = PROBES[dom][:n_probes]

        deltas = {"hi": [], "lo": []}
        for ph in phrases:
            base = _neutral(ph)
            for tag, val in (("hi", args.hi), ("lo", args.lo)):
                prof = np.full(n, 0.5, np.float32)
                prof[di] = val
                llm.set_profile(prof)
                txt = llm.generate(ph, max_new_tokens=args.tokens, temperature=0.0)
                if args.show_text:
                    print(f"\n--- [{di:03d}] {tag} · «{ph}» ---\n{txt}")
                deltas[tag].append(hz.profile(txt) - base)
        llm.clear()

        # efecto bidireccional, promediado entre sondas
        effect = np.mean(deltas["hi"], 0) - np.mean(deltas["lo"], 0)   # (104,)
        own = float(effect[di])
        others = np.delete(effect, di)
        z = float((own - others.mean()) / (others.std() + 1e-9))
        rank = int((np.abs(effect) > abs(own)).sum())   # 0 = la mas movida

        top = np.argsort(-np.abs(effect))[:8]
        movers = [[names[t], round(float(effect[t]), 3)] for t in top]

        report[key] = {"effect": round(own, 4), "z": round(z, 2), "rank": rank,
                       "sign_ok": bool(own > 0), "domain": dom,
                       "top_movers": movers,
                       "effect_vec": [round(float(x), 4) for x in effect]}
        rf.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        mark = "🟢" if (rank == 0 and own > 0) else ("🟡" if rank < 5 else "🔴")
        print(f"{mark} [{di:03d}] {key:<34} [{dom:<8}] "
              f"effect={own:+.3f} z={z:+5.1f} rank={rank}")
        if rank > 0:
            print("     se movieron mas: " +
                  ", ".join(f"{nm.split('_', 1)[-1]} {v:+.2f}" for nm, v in movers[:5]))

    # ── resumen con correccion de modo comun (por dominio) ──────────────────
    rows = {k: v for k, v in report.items() if not k.startswith("_")}
    have_vec = {k: v for k, v in rows.items() if "effect_vec" in v}
    if len(have_vec) >= 10:
        print(f"\n{'='*64}\n  CORRECCION DE MODO COMUN (sesgo del juez)")
        E = {k: np.array(v["effect_vec"]) for k, v in have_vec.items()}
        idx = {nm: i for i, nm in enumerate(names)}
        # modo comun por dominio (si hay >=5 dims), si no global
        by_dom = {}
        for k, v in have_vec.items():
            by_dom.setdefault(v["domain"], []).append(k)
        global_cm = np.mean(list(E.values()), 0)
        for k, v in have_vec.items():
            group = by_dom[v["domain"]]
            cm = (np.mean([E[g] for g in group], 0) if len(group) >= 5
                  else global_cm)
            ec = E[k] - cm
            i = idx[k]
            own = float(ec[i])
            others = np.delete(ec, i)
            v["z_corr"] = round(float((own - others.mean()) / (others.std() + 1e-9)), 2)
            v["rank_corr"] = int((np.abs(ec) > abs(own)).sum())
        rf.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")

        srt = sorted(have_vec.items(), key=lambda kv: -kv[1]["z_corr"])
        reales = sum(1 for _, v in have_vec.items()
                     if v["rank_corr"] == 0 and v["sign_ok"])
        z15 = sum(1 for _, v in have_vec.items() if v["z_corr"] >= 1.5)
        print(f"  {reales}/{len(have_vec)} dims rank_corr=0 con signo OK | "
              f"{z15} dims z_corr>=1.5")
        print("  Top 15 (corregido):")
        for k, v in srt[:15]:
            print(f"    z={v['z_corr']:+6.1f} rank={v['rank_corr']:<3} "
                  f"[{v['domain']:<8}] {k}")
        print("  Cola (candidatas a dial muerto):")
        for k, v in srt[-10:]:
            print(f"    z={v['z_corr']:+6.1f} rank={v['rank_corr']:<3} "
                  f"[{v['domain']:<8}] {k}")
    print(f"{'='*64}\nreport -> {rf}")


if __name__ == "__main__":
    main()
