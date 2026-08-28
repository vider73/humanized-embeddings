"""
word_probe.py — The reliable, eyeball-able test. Two words in, two words out.

The Humanizer judge is broken (judge_blind: 33%, chance). So we drop it. This
probe gives the model TWO neutral seed words and asks for TWO associated words,
at temperature 0 (deterministic), with NO steering and WITH steering, and lays
the answers side by side. If steering "divinity" turns "río, piedra -> agua,
roca" into "río, piedra -> bautismo, altar", a human (or a real LLM-judge) sees
it at a glance. No 104-dim readout, no z-score — just the words.

Two words out (instead of one) gives the steering room to express without
collapsing into a single broken token; a lower alpha keeps the morphology
intact while the concept still colors the choice.

Output: WORD_PROBE.md (dimension | palabras | neutro | dirigido[alpha]) and
word_probe.csv, for the 10 strongest dials (by afinacion z) + divinity &
consciousness. Per-dim "shift rate" = how many pairs changed.

  python -m steering.word_probe                  # alpha 0.18, top-10 dials
  python -m steering.word_probe --alpha 0.25 --dims 90 23 53
"""
import argparse
import csv
import json
import re

import numpy as np

from . import config
from .steering_model import SteeredLlama

WORDS = [
    "agua", "piedra", "casa", "árbol", "fuego", "mano", "ciudad", "noche", "pan",
    "río", "libro", "puerta", "niño", "camino", "mar", "montaña", "viento",
    "tierra", "sol", "luna", "perro", "flor", "espejo", "reloj", "silencio",
    "sangre", "hueso", "máquina", "dinero", "guerra", "madre", "sueño", "voz",
    "sombra", "cielo", "muro", "semilla", "hambre", "frío", "luz", "hierro",
    "vino", "barco", "campana", "llave", "red", "nube", "raíz", "ceniza", "hilo",
]
PAIRS = [(WORDS[i], WORDS[i + 1]) for i in range(0, len(WORDS) - 1, 2)]  # 25 pairs

PROMPT = ('Responde solo con DOS palabras que asocies con «{w1}» y «{w2}». '
          'Únicamente las dos palabras, sin explicación.')

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_SKIP = {"respuesta", "palabra", "palabras", "la", "las", "los", "el",
         "una", "uno", "dos", "claro", "y", "con", "de"}


def _dim_names():
    return json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]


def _top_dims(names, k, must):
    """Top-k dims by z-máx from afinacion.md (8B), forcing `must` indices in."""
    idx = {n: i for i, n in enumerate(names)}
    ranked = []
    af = config.VEC_DIR / "afinacion.md"
    if af.exists():
        for line in af.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or "z@" in line or "---" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            dtok = next((t for t in cells[0].split() if t.startswith("d")), None)
            try:
                zmax = float(cells[-1])
            except (ValueError, IndexError):
                continue
            if dtok in idx:
                ranked.append((zmax, idx[dtok]))
    ranked.sort(reverse=True)
    chosen = []
    for _, i in ranked:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= k:
            break
    for m in must:
        if m not in chosen:
            chosen.append(m)
    return chosen


def _two_words(llm, w1, w2, tokens, n_out=2):
    txt = llm.generate(PROMPT.format(w1=w1, w2=w2), max_new_tokens=tokens, temperature=0.0)
    out = []
    for tok in WORD_RE.findall(txt):
        if tok.lower() in _SKIP:
            continue
        out.append(tok.lower())
        if len(out) >= n_out:
            break
    return " ".join(out) if out else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.18)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="dim indices (default: top-10 by afinacion z + divinity/consciousness)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=10)
    args = ap.parse_args()

    names = _dim_names()
    dims = args.dims or _top_dims(names, args.topk, must=[90, 23])
    print("⏳ Llama + control vectors...")
    llm = SteeredLlama(); llm.alpha = args.alpha
    print(f"✅ alpha={args.alpha} | layers={llm.layers} | {len(PAIRS)} pairs | "
          f"dims: {[names[d] for d in dims]}\n")

    llm.clear()
    neutral = {p: _two_words(llm, p[0], p[1], args.tokens) for p in PAIRS}
    print("neutral done")

    rows = []   # (dim_name, "w1, w2", neutral, steered)
    for d in dims:
        prof = np.full(len(names), 0.5, np.float32); prof[d] = 0.95
        llm.set_profile(prof)
        shifted = 0
        for p in PAIRS:
            s = _two_words(llm, p[0], p[1], args.tokens)
            shifted += (s != neutral[p])
            rows.append((names[d], f"{p[0]}, {p[1]}", neutral[p], s))
        llm.clear()
        print(f"  {names[d]:<32} shift {shifted}/{len(PAIRS)}")

    with open(config.ROOT / "word_probe.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dimension", "palabras", "neutro", f"dirigido_a{args.alpha}"])
        wr.writerows(rows)

    md = [f"# Word probe — two-word association (alpha={args.alpha}, temp 0)", "",
          f"Prompt: `{PROMPT}`", "",
          "| dimension | shift rate |", "|---|---|"]
    by_dim = {}
    for dn, w, n, s in rows:
        by_dim.setdefault(dn, []).append((w, n, s))
    for dn, items in by_dim.items():
        sh = sum(1 for _, n, s in items if n != s)
        md.append(f"| {dn} | {sh}/{len(items)} |")
    md += ["", "## Tables", ""]
    for dn, items in by_dim.items():
        md += [f"### {dn}", "",
               f"| palabras | neutro | dirigido (α={args.alpha}) |", "|---|---|---|"]
        for w, n, s in items:
            mark = "" if n == s else " **←**"
            md.append(f"| {w} | {n} | {s}{mark} |")
        md.append("")
    (config.ROOT / "WORD_PROBE.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n📊 WORD_PROBE.md + word_probe.csv written ({len(rows)} rows).")


if __name__ == "__main__":
    main()
