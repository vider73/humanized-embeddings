"""
transcript.py — El cuaderno de perlas. Cada generacion (frase, notas, neutral,
dirigida) se anexa a un JSONL append-only. No se pierde ni una salida buena;
es la materia prima de la seccion de resultados del README.

Respeta EMB_TAG (igual que tuner.py): los barridos por tamano de modelo
escriben en transcript_<tag>.jsonl y no se pisan.

  python -m steering.transcript                 # cuenta y muestra las ultimas
  python -m steering.transcript --tail 5
  python -m steering.transcript --md perlas.md  # exporta legible para el repo
"""
import argparse
import json
import os
from datetime import datetime

from . import config

_TAG = f"_{os.environ['EMB_TAG']}" if os.environ.get("EMB_TAG") else ""
TRANSCRIPT = config.VEC_DIR / f"transcript{_TAG}.jsonl"


def notes_from_profile(profile, dim_names, layers, alpha):
    """Notas (dims != 0.5) de un perfil plano + capas + alpha comunes."""
    return [{"dim": i, "name": dim_names[i], "val": round(float(v), 3),
             "layer": list(layers), "alpha": alpha}
            for i, v in enumerate(profile) if abs(v - 0.5) > 1e-3]


def notes_from_voices(voices, dim_names, fallback_alpha=None):
    """Notas de un dict de voicing {capa: (perfil, alpha|None)}."""
    notes = []
    for L, spec in voices.items():
        prof, a = spec if isinstance(spec, tuple) else (spec, None)
        a = a if a is not None else fallback_alpha
        for i, v in enumerate(prof):
            if abs(v - 0.5) > 1e-3:
                notes.append({"dim": i, "name": dim_names[i],
                              "val": round(float(v), 3), "layer": L, "alpha": a})
    return notes


def save(source, phrase, neutral, steered, *, notes=None, mode=None,
         model=None, path=TRANSCRIPT):
    """Anexa una perla. Robusto: nunca tumba la generacion por un fallo de IO."""
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model or config.LLM_NAME,
        "mode": mode,
        "source": source,
        "phrase": phrase,
        "notes": notes or [],
        "neutral": neutral,
        "steered": steered,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                       # nunca romper la generacion
        print(f"⚠ no se pudo guardar la perla: {e}")
    return rec


def _load(path=TRANSCRIPT):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _fmt_notes(notes):
    return ", ".join(
        f"[{n['dim']:03d}]{n['name'].split('_', 1)[-1]}={n['val']}"
        f"@L{n['layer']}·α{n['alpha']}" for n in notes) or "(ninguna)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--md", metavar="FICHERO", help="exporta a markdown legible")
    args = ap.parse_args()

    recs = _load()
    print(f"📿 {len(recs)} perlas en {TRANSCRIPT.name}")

    if args.md:
        out = config.VEC_DIR / args.md
        lines = [f"# Perlas — {len(recs)} generaciones guardadas", ""]
        for r in recs:
            lines += [f"## «{r['phrase']}»  ({r['ts']}, {r['source']})",
                      f"`{_fmt_notes(r['notes'])}`  · modo={r.get('mode')}", "",
                      "**neutral**", "", r["neutral"], "",
                      "**dirigida**", "", r["steered"], "", "---", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"📝 exportado -> {out}")
        return

    for r in recs[-args.tail:]:
        print(f"\n{'='*60}\n«{r['phrase']}»  [{r['ts']}]")
        print(f"notas: {_fmt_notes(r['notes'])}")
        print(f"{'-'*60}\n{r['steered']}")


if __name__ == "__main__":
    main()
