"""
clean_table.py — Clean the concept table (dataset_Y_human) and rate its 104 axes.

The table was scored by Llama-3 one concept × axis at a time (20_score_concepts).
Auditing it after the temperature failure turned up four kinds of damage:

  junk rows      ~420 "concepts" are the labelling LLM's own chatter: preambles
                 ("here is the list of 10 words related to …:", "aquí te dejo la
                 lista …") and sign-offs ("let me know if you need more!"),
                 each scored on all 104 axes as if it were a word.
  early batch    the first 213 rows were scored by a different run: 70 % of
                 their cells are informative (non-zero, non-fallback) against
                 17 % in the rest, so they crowd the top of most axes
                 (adoración, admiración, altruismo, aislante perfecto …).
  scale words    a few words are read as positions on the scale itself:
                 "máxima" sits in the top 5 % of 56 axes, "máximos" 41.
  fallback cells when the scorer has no idea it writes a digit run near the
                 middle — 0.43221, 0.4321234, 0.4321111, 0.4211111, 0.4123457 —
                 30 % of all cells (88 % of malleability). Those are "unknown",
                 not 0.43. Also 31 cells above 1 (día = 6.375 on size).

Cleaning drops the junk rows, the early batch and the scale words, and turns
fallback and out-of-range cells into NaN. Nothing is overwritten: the clean
files sit next to the originals and keep `source_rows` (row index in the
original table) so caches built on the original rows can be subset.

Each axis is then rated on the clean rows (first match wins):
  broken         checked by hand: the top concepts do not fit the axis
  polarity-mixed checked by hand: both poles crowd the top (happiness: euforia,
                 éxtasis, miseria, desgracia) — the scorer rated how much a
                 concept is ABOUT the axis, not where it sits on it
  dead           fewer than MIN_SPARSE concepts score above 0: the anchors are
                 cosmic (a quark → the observable universe), every everyday
                 thing is 0; once the early batch goes there is nothing left
  fallback-heavy the scorer gave up (fallback) on ≥ 60 % of concepts
  sparse         fewer than MIN_INFO concepts score above 0: only the extreme
                 end is scored (radioactivity: caesium, reactor), no gradation
  ok             the rest

  python -m steering.clean_table            # write clean files + TABLE_AUDIT.md
  from steering.clean_table import load_clean
  concepts, Y, names, status = load_clean(fill="mean")
"""
import json
import re

import numpy as np

from . import config

Y_CLEAN = config.ROOT / "dataset_Y_clean.npy"
X_CLEAN = config.ROOT / "dataset_X_clean.npy"
META_CLEAN = config.ROOT / "dataset_metadata_clean.json"
AUDIT_JSON = config.ROOT / "table_audit.json"
AUDIT_MD = config.ROOT / "TABLE_AUDIT.md"

JUNK = re.compile(r"here is|here are|aquí|lista|list of|términos|terms|words related|palabras|:|"
                  r"as you requested|let me know|gracias por|acepto el reto|vaina fácil")
EARLY_BATCH = 213                   # change point in the informative-cell share (0.70 → 0.17)
SCALE_WORDS = {"máxima", "máximos", "santisima", "alta", "baja", "absoluto"}
MIN_INFO = 300
MIN_SPARSE = 50
MAX_FALLBACK = 0.6

# set by hand after reading the clean top concepts of every axis (TABLE_AUDIT.md lists them)
BROKEN = {
    "d004_temperatura": "anchors run from absolute zero to the Planck temperature: fire, sun, "
                        "lava and ovens score 0, abstract virtues 0.86; tracks string length",
    "d028_salud": "top concepts are menta, semanas, efectivo, solventes, poda — no health",
    "d065_verdad_facticidad": "top concepts are altura, radares, benceno, pistón, Burj al Arab — no pattern",
    "d068_probabilidad": "ill-posed for nouns (impossible → inevitable): top animal, verdad, horario",
    "d076_legalidad": "lexical: «mandato constitucional» pulls mando, mandante, Mandela to the top",
    "d037_sabor_amargor": "valence, not taste: abominación, odium, sodomita above any bitter food",
    "d061_soledad": "valence, not solitude: detestación, odiosidad, cianuro, Auschwitz",
    "d064_aburrimiento": "valence, not tedium: detestación, inferno, cloaca, Auschwitz, cianuro",
    "d096_dificultad_de_ejecución": "valence, not difficulty: terrorífico, infarto, Auschwitz, catástrofe",
}
POLARITY = {
    "d045_felicidad": "miseria and desgracia in the top 4",
    "d047_calma": "terrorista, satanás and pánico in the top 5",
    "d050_orgullo": "desgracia and repugnancia are the top 2",
    "d052_gratitud": "enemistad and resentimiento in the top 5",
    "d057_tristeza": "felicidades is #2",
    "d035_olor_agradabilidad": "repulsión, antipatía and asquerosidad in the top 5",
    "d086_belleza_estética": "desagrado, hediondez and asquerosidad are the top 3",
    "d051_valentía": "fobia is #2",
    "d063_odio": "respeto is #2",
    "d077_ética_bondad": "maldiciente and maldad in the top 7",
}


def is_fallback(Y):
    """Digit-run defaults the scorer writes when it has no idea."""
    s = np.char.mod("%.7f", Y)
    return np.vectorize(lambda t: t[2:5] == "432" or t[2:7] in ("42111", "41234"))(s)


def _row_flags(concepts, Y):
    junk = np.array([bool(JUNK.search(c)) for c in concepts])
    early = np.arange(len(concepts)) < EARLY_BATCH
    scale = np.array([c in SCALE_WORDS for c in concepts])
    return junk, early, scale


def _axis_rows(concepts, Yc, names, prompts):
    """Per-axis metrics and status on the clean table (NaN = missing)."""
    L = np.array([len(c) for c in concepts], float)
    rows = []
    for i, nm in enumerate(names):
        y = Yc[:, i]
        ok = ~np.isnan(y)
        info = int((ok & (y > 0)).sum())
        fb = float(1 - ok.mean())
        zero = float((y[ok] == 0).mean()) if ok.any() else 1.0
        r_len = float(np.corrcoef(L[ok], y[ok])[0, 1]) if y[ok].std() > 0 else 0.0
        order = np.argsort(-np.where(ok, y, -1))
        top = [concepts[k] for k in order[:10]]
        if nm in BROKEN:
            status, why = "broken", BROKEN[nm]
        elif nm in POLARITY:
            status, why = "polarity-mixed", POLARITY[nm]
        elif info < MIN_SPARSE:
            status, why = "dead", f"only {info} concepts score above 0"
        elif fb >= MAX_FALLBACK:
            status, why = "fallback-heavy", f"scorer gave up on {fb:.0%} of concepts"
        elif info < MIN_INFO:
            status, why = "sparse", f"only {info} concepts score above 0"
        else:
            status, why = "ok", ""
        rows.append({"dim": nm, "status": status, "why": why, "info": info, "fallback": fb,
                     "zero": zero, "r_len": r_len, "top": top,
                     "min": prompts[i]["scale"]["min"], "max": prompts[i]["scale"]["max"]})
    return rows


def build():
    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    concepts, names = meta["concepts"], meta["dimension_names"]
    Y = np.load(config.DATA_Y_FILE).astype(np.float32)
    X = np.load(config.ROOT / "dataset_X_embeddings.npy")
    prompts = json.loads((config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))
    junk, early, scale = _row_flags(concepts, Y)
    keep = ~(junk | early | scale)
    src = np.where(keep)[0]
    fb = is_fallback(Y) | (Y > 1) | (Y < 0)
    Yc = np.where(fb, np.nan, Y)[keep]
    cc = [concepts[k] for k in src]
    rows = _axis_rows(cc, Yc, names, prompts)

    np.save(Y_CLEAN, Yc.astype(np.float32))
    np.save(X_CLEAN, X[keep])
    META_CLEAN.write_text(json.dumps({
        "concepts": cc, "dimension_names": names, "embedding_model": meta["embedding_model"],
        "source_rows": src.tolist(), "axis_status": {r["dim"]: r["status"] for r in rows},
        "missing": "NaN = scorer fallback (0.432… digit runs) or value outside [0, 1]",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"n_in": len(concepts), "n_out": int(keep.sum()), "junk": int(junk.sum()),
               "early": int((early & ~junk).sum()), "scale": int((scale & ~early).sum()),
               "fallback_cells_in": float(is_fallback(Y).mean()),
               "out_of_range_cells": int(((Y > 1) | (Y < 0)).sum()),
               "missing_cells_out": float(np.isnan(Yc).mean()),
               "dropped": {"junk": [concepts[k] for k in np.where(junk)[0]],
                           "early": [concepts[k] for k in np.where(early & ~junk)[0]],
                           "scale": [concepts[k] for k in np.where(scale & ~early)[0]]}}
    AUDIT_JSON.write_text(json.dumps({"summary": summary, "axes": rows}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    report(summary, rows)
    return summary, rows


def load_clean(fill=None):
    """-> concepts, Y (n, 104), names, status. fill=None keeps NaN for missing
    cells; fill="mean" puts the axis mean of its known cells there."""
    meta = json.loads(META_CLEAN.read_text(encoding="utf-8"))
    Y = np.load(Y_CLEAN)
    if fill == "mean":
        Y = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    return meta["concepts"], Y, meta["dimension_names"], meta["axis_status"]


def report(s, rows):
    en = json.loads((config.ROOT / "dimension_names_en.json").read_text(encoding="utf-8"))
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    md = ["# Concept table audit", "",
          "Generated by `python -m steering.clean_table` from `dataset_Y_human.npy` "
          f"({s['n_in']} concepts × 104 axes). Clean files: `dataset_Y_clean.npy` (NaN = missing), "
          "`dataset_X_clean.npy`, `dataset_metadata_clean.json` (with `source_rows` into the "
          "original and `axis_status`). The originals are untouched.", "",
          "## What was wrong and what was done", "",
          "| problem | size | action |", "|---|---|---|",
          f"| LLM chatter scored as concepts (\"here is the list of 10 words related to…\", "
          f"\"let me know if you need more!\") | {s['junk']} rows | dropped |",
          f"| early scoring batch: 70 % informative cells vs 17 % in the rest, crowds the top of "
          f"most axes | {s['early']} rows | dropped |",
          f"| scale words read as positions (\"máxima\" in the top 5 % of 56 axes) | "
          f"{s['scale']} rows | dropped |",
          f"| fallback digit runs when the scorer has no idea (0.43221, 0.4321234, 0.4211111…) | "
          f"{s['fallback_cells_in']:.0%} of cells | NaN |",
          f"| values above 1 (día = 6.375 on size) | {s['out_of_range_cells']} cells | NaN |", "",
          f"Result: **{s['n_out']} concepts**, {s['missing_cells_out']:.0%} of cells missing.", "",
          "## Axes", "",
          f"- " + " · ".join(f"{k} {v}" for k, v in cnt.most_common()), "",
          "Statuses (first match wins): **broken** = checked by hand, the top concepts do not fit "
          "the axis; **polarity-mixed** = checked by hand, both poles crowd the top (the scorer "
          "rated how much a concept is *about* the axis, not where it sits); **dead** = fewer "
          f"than {MIN_SPARSE} concepts score above 0 (cosmic anchors make every everyday thing 0); "
          f"**fallback-heavy** = the scorer gave up on ≥ {MAX_FALLBACK:.0%} of concepts; "
          f"**sparse** = fewer than {MIN_INFO} concepts score above 0, only the extreme end is "
          "scored; **ok** = the rest. `info` = concepts with a known score above 0; `r len` = "
          "correlation with the concept's string length.", "",
          "| axis | status | anchors (min → max) | info | fallback | zero | r len | top concepts (clean) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {en.get(r['dim'], r['dim'])} | {r['status']} | {r['min']} → {r['max']} | "
                  f"{r['info']} | {r['fallback']:.0%} | {r['zero']:.0%} | {r['r_len']:+.2f} | "
                  f"{', '.join(r['top'][:8])} |")
    bad = [r for r in rows if r["status"] != "ok"]
    md += ["", "## Not ok, and why", ""] + [f"- **{en.get(r['dim'], r['dim'])}** ({r['status']}): "
                                            f"{r['why']}" for r in bad]
    AUDIT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"📊 {AUDIT_MD}")


if __name__ == "__main__":
    s, rows = build()
    print(f"{s['n_in']} → {s['n_out']} concepts; missing cells {s['missing_cells_out']:.0%}")
    for r in rows:
        print(f"{r['dim'][:28]:<28} {r['status']:<14} info {r['info']:4d} fb {r['fallback']:.2f} "
              f"rlen {r['r_len']:+.2f} | {', '.join(t[:12] for t in r['top'][:8])}")
