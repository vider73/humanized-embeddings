"""
58_periodic_table.py — Assemble the periodic table of semantic concepts.

Crosses, for every model rung that has artifacts on disk, two measurements
per dimension:
  · steering capacity  — fidelity z + sign, at each dim's RESONANT alpha
    (the best sign-correct z across that rung's tuner sweep; falls back to the
    single flat report if the rung has no sweep yet)
  · geometric stability — RSA between every pair of models' control-vector
    sets (mid derive-layer, unit rows)

Why resonant alpha and not a flat 0.15: a single low alpha is the noise floor
(polarity_audit.py) — a dial can read -0.2 at 0.15 and +4.4 at 0.35. Judging
every model at 0.15 would confuse "dial absent in this model" with "dial asleep
at this pressure", which is exactly the A/B/C question the table must resolve.
Each rung is therefore read at its own per-dim best alpha (from
fidelity[_<tag>]_aXXX.json if present).

Writes PERIODIC.md: a dims × models table (🟢 z≥1.5 & sign OK, 🟡 z≥1.0 & sign
OK, 🔴 otherwise, · = not measured), per-family grouping, robust-dial counts
per rung, and the cross-model RSA matrix.

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

# (column label, model tag [""=8B untagged], control vectors file)
RUNGS = [
    ("smol-135M", "smol", VEC / "control_vectors_caa_white_smol.npy"),
    ("1B",        "1b",   VEC / "control_vectors_caa_white_1b.npy"),
    ("3B",        "3b",   VEC / "control_vectors_caa_white_3b.npy"),
    ("8B",        "",     VEC / "control_vectors_caa_white.npy"),
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


def _reports_for(tag):
    """All sweep reports for a rung, or the flat report if no sweep exists."""
    prefix = f"fidelity_{tag}" if tag else "fidelity"
    sweep = sorted(VEC.glob(f"{prefix}_a[0-9]*.json"))   # fidelity[_<tag>]_aNNN.json
    if sweep:
        return sweep
    flat = VEC / (f"fidelity_{tag}.json" if tag else "fidelity_a015.json")
    return [flat] if flat.exists() else []


def _best_rows(tag):
    """Per-dim row chosen at its resonant alpha (max z among sign-correct
    alphas; else the max-z row, which stays red). Returns (rows, alphas, n_a)."""
    reps = _reports_for(tag)
    per_alpha = []
    for p in reps:
        d = json.loads(p.read_text(encoding="utf-8"))
        a = d.get("_meta", {}).get("alpha")
        per_alpha.append((a, {k: v for k, v in d.items() if not k.startswith("_")}))
    if not per_alpha:
        return {}, set(), 0
    keys = set().union(*(set(r) for _, r in per_alpha))
    best, alphas = {}, set()
    for k in keys:
        cands = [(a, r[k]) for a, r in per_alpha if k in r]
        ok = [c for c in cands if c[1].get("sign_ok")]
        a, row = max(ok or cands, key=lambda ar: ar[1].get("z", -9e9))
        best[k] = row
        alphas.add(a)
    return best, alphas, len(per_alpha)


def main():
    names = json.load(open("dataset_metadata.json", encoding="utf-8"))["dimension_names"]
    en = json.load(open("dimension_names_en.json", encoding="utf-8"))

    rungs = []
    for label, tag, cv_p in RUNGS:
        rows, alphas, n_a = _best_rows(tag)
        if not rows:
            print(f"  · {label}: no report yet — skipped")
            continue
        cv = None
        if cv_p.exists():
            M = np.load(cv_p)
            cv = _unit_rows(M[M.shape[0] // 2].astype(np.float64))
        rungs.append(dict(label=label, rows=rows, cv=cv, n_a=n_a, alphas=alphas))
        greens = sum(1 for v in rows.values()
                     if v.get("sign_ok") and v.get("z", 0) >= 1.5)
        swept = f"swept {sorted(a for a in alphas if a is not None)}" if n_a > 1 else "flat α (no sweep)"
        print(f"  ✓ {label}: {len(rows)} dims, {greens} robust  [{swept}]")

    if not rungs:
        print("No rungs found. Run:  python -m steering.periodic_run")
        return

    lines = [
        "# PERIODIC TABLE — semantic dials across model scale",
        "",
        "Same 104 axes, same Spanish stimuli, same judge (the Humanizer); the",
        "only variable is the target model. Each dial is read at its **resonant",
        "alpha** per model (best sign-correct z across that rung's sweep), not a",
        "flat 0.15 — see the module docstring and `polarity_audit.py` for why.",
        "🟢 robust (z≥1.5, correct sign) · 🟡 weak (z≥1.0) · 🔴 dead/inverted ·",
        "`·` not measured. See `ROADMAP.md` for the three scenarios.",
        "",
        "## Robust dials per rung",
        "",
    ]
    for r in rungs:
        greens = [k for k, v in r["rows"].items()
                  if v.get("sign_ok") and v.get("z", 0) >= 1.5]
        tag = "" if r["n_a"] > 1 else "  *(flat α — needs a tuner sweep)*"
        lines.append(f"- **{r['label']}**: {len(greens)} robust "
                     f"of {len(r['rows'])} measured{tag}")

    with_cv = [r for r in rungs if r["cv"] is not None]
    if len(with_cv) >= 2:
        lines += ["", "## Cross-model geometric stability (RSA of control-vector sets)", ""]
        lines += ["| | " + " | ".join(r["label"] for r in with_cv) + " |",
                  "|" + "---|" * (len(with_cv) + 1)]
        for a in with_cv:
            cells = ["—" if a is b else f"{_rsa(a['cv'], b['cv']):.3f}" for b in with_cv]
            lines.append(f"| **{a['label']}** | " + " | ".join(cells) + " |")

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
            lines.append(f"| {n} ({en.get(n, '')}) | " + " | ".join(cells) + " |")

    lines += ["", f"*Generated {datetime.now():%Y-%m-%d %H:%M}. Each cell = the "
              "dim's best sign-correct alpha for that model. Raw z (not "
              "kin-corrected) for cross-model comparability; smol-135M is "
              "English-heavy, so Spanish probes underestimate it; per-rung regime "
              "in each report's `_meta`. Rungs marked *(flat α)* still need a "
              "`steering.tuner` sweep and undercount robust dials.*"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📜 table -> {OUT}")


if __name__ == "__main__":
    main()
