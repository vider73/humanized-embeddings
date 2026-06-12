"""
58_periodic_table.py — Assemble the periodic table of semantic concepts.

Crosses, for every model rung that has artifacts on disk, two measurements
per dimension:
  · steering capacity  — raw fidelity z + sign (fidelity_<tag>.json)
  · geometric stability — RSA between every pair of models' control-vector
    sets (mid derive-layer, unit rows), and vs the MiniLM linear readouts.

Then writes PERIODIC.md: a dims × models table (🟢 z≥1.5 & sign OK,
🟡 z≥1.0 & sign OK, 🔴 otherwise, · = not measured), per-family grouping,
robust-dial counts per rung, and the cross-model RSA matrix.

The three scenarios it exists to distinguish (see ROADMAP.md):
  A. same dials, progressively cleaner with scale
  B. new dials EMERGE with capability (the table grows "periods")
  C. early dials vanish while others appear (representational phase
     transitions — the jackpot)

  python 58_periodic_table.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

VEC = Path("steering/vectors")
OUT = Path("PERIODIC.md")

# (column label, fidelity report, control vectors file)
RUNGS = [
    ("smol-135M", VEC / "fidelity_smol.json", VEC / "control_vectors_caa_white_smol.npy"),
    ("1B",        VEC / "fidelity_1b.json",   VEC / "control_vectors_caa_white_1b.npy"),
    ("3B",        VEC / "fidelity_3b.json",   VEC / "control_vectors_caa_white_3b.npy"),
    ("8B",        VEC / "fidelity_a015.json", VEC / "control_vectors_caa_white.npy"),
]


def _unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _rsa(A, B):
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef((A @ A.T)[iu], (B @ B.T)[iu])[0, 1])


def _mark(row):
    if row is None:
        return "·"
    z, sig = row.get("z", 0.0), row.get("sign_ok", False)
    if sig and z >= 1.5:
        return "🟢"
    if sig and z >= 1.0:
        return "🟡"
    return "🔴"


def main():
    names = json.load(open("dataset_metadata.json", encoding="utf-8"))["dimension_names"]
    en = json.load(open("dimension_names_en.json", encoding="utf-8"))

    rungs = []
    for label, rep_p, cv_p in RUNGS:
        if not rep_p.exists():
            print(f"  · {label}: no report yet ({rep_p.name}) — skipped")
            continue
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        rows = {k: v for k, v in rep.items() if not k.startswith("_")}
        cv = None
        if cv_p.exists():
            M = np.load(cv_p)
            cv = _unit_rows(M[M.shape[0] // 2].astype(np.float64))
        rungs.append(dict(label=label, rows=rows, cv=cv))
        greens = sum(1 for v in rows.values()
                     if v.get("sign_ok") and v.get("z", 0) >= 1.5)
        print(f"  ✓ {label}: {len(rows)} dims measured, {greens} robust")

    if not rungs:
        print("No rungs found. Run:  python -m steering.periodic_run")
        return

    lines = [
        "# PERIODIC TABLE — semantic dials across model scale",
        "",
        "Same 104 axes, same Spanish stimuli, same judge (the Humanizer); the",
        "only variable is the target model. 🟢 robust dial (raw fidelity z≥1.5,",
        "correct sign) · 🟡 weak (z≥1.0) · 🔴 dead/inverted · `·` not measured.",
        "See `ROADMAP.md` for the three scenarios this table distinguishes.",
        "",
        "## Robust dials per rung",
        "",
    ]
    for r in rungs:
        greens = [k for k, v in r["rows"].items()
                  if v.get("sign_ok") and v.get("z", 0) >= 1.5]
        lines.append(f"- **{r['label']}**: {len(greens)} robust "
                     f"of {len(r['rows'])} measured")

    # cross-model geometry
    with_cv = [r for r in rungs if r["cv"] is not None]
    if len(with_cv) >= 2:
        lines += ["", "## Cross-model geometric stability (RSA of control-vector sets)", ""]
        header = "| | " + " | ".join(r["label"] for r in with_cv) + " |"
        lines += [header, "|" + "---|" * (len(with_cv) + 1)]
        for a in with_cv:
            cells = []
            for b in with_cv:
                cells.append("—" if a is b else f"{_rsa(a['cv'], b['cv']):.3f}")
            lines.append(f"| **{a['label']}** | " + " | ".join(cells) + " |")

    # the table itself, grouped by domain (stored in the fidelity rows)
    dom_of = {}
    for r in rungs:
        for k, v in r["rows"].items():
            dom_of.setdefault(k, v.get("domain", "?"))
    lines += ["", "## The table", "",
              "| dimension | " + " | ".join(r["label"] for r in rungs) + " |",
              "|---|" + "---|" * len(rungs)]
    for dom in ("materia", "vida", "sentidos", "mente", "sociedad", "?"):
        block = [n for n in names if dom_of.get(n, "?") == dom]
        if not block:
            continue
        lines.append(f"| **— {dom} —** |" + " |" * len(rungs))
        for n in block:
            cells = [_mark(r["rows"].get(n)) for r in rungs]
            label = f"{n} ({en.get(n, '')})"
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", f"*Generated {datetime.now():%Y-%m-%d %H:%M}. Caveats: raw z "
              "(not kin-corrected) for cross-model comparability; smol-135M is "
              "English-heavy, so Spanish probes underestimate it; per-rung "
              "regime details live in each report's `_meta`.*"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📜 table -> {OUT}")


if __name__ == "__main__":
    main()
