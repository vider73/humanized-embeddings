"""
falsify_report.py — The four-corner falsification, reproducible (CPU, no GPU).

The August 2026 headline ("real vs random now discriminate, 15 vs 6") lived
only in a commit message: the corners were compared by hand. This module
recomputes it from the stored fidelity reports, adds the significance the
count comparison was missing, and writes the scorecard the README points to.

The four corners — same probes, same alpha, same layer, same clamp:

                 judge v1 (word-trained)     judge v2 (sentence-trained)
  REAL vectors   fidelity_a025.json          fidelity_j2_a025.json
  RANDOM vectors fidelity_rand_a025.json     fidelity_j2_rand_a025.json

A dial counts as REAL only if it is green on the real vectors AND dead on the
random ones (the June lesson: four greens fire on gaussian noise too — they
detect degradation, not meaning). Counts alone are weak with 104 dims and a
z>=1.5 cut, so the comparison is also made paired, dim by dim, with a
sign-flip permutation test on the z and |effect| differences.

  python -m steering.falsify_report
  python -m steering.falsify_report --real fidelity_j2_a025.json \
                                    --null fidelity_j2_rand_a025.json
  python -m steering.falsify_report --no-write          # print only
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .analyze import scorecard

SCORECARD_V2 = config.VEC_DIR / "scorecard_v2.md"
Z_GREEN = 1.5
B_PERM = 200_000

# (label, real report, null report) — the corners as they were actually run
CORNERS = (
    ("judge v2", "fidelity_j2_a025.json", "fidelity_j2_rand_a025.json"),
    ("judge v1", "fidelity_a025.json", "fidelity_rand_a025.json"),
)


def _rows(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _meta(path):
    return json.loads(Path(path).read_text(encoding="utf-8")).get("_meta", {})


def _green(v):
    """Raw fidelity green: moved its own dim, in the right direction, hard."""
    return bool(v.get("sign_ok")) and v.get("z", 0.0) >= Z_GREEN


def stats(path):
    r = _rows(path)
    n = len(r) or 1
    return dict(
        n=len(r),
        green=sum(1 for v in r.values() if _green(v)),
        rank0=sum(1 for v in r.values() if v.get("sign_ok") and v.get("rank", 99) == 0),
        top3=sum(1 for v in r.values() if v.get("sign_ok") and v.get("rank", 99) < 3),
        sign=sum(1 for v in r.values() if v.get("sign_ok")),
        mae=float(np.mean([abs(v.get("effect", 0.0)) for v in r.values()])),
    )


def kin_green(path, kin_r=0.4):
    """Greens after analyze.py's kinship correction (family-aware common mode)."""
    out = scorecard(path, kin_r)
    return {d["name"]: d for d in out}


def _perm_p(diffs, seed=0, b=B_PERM):
    """Two-sided sign-flip permutation p for a paired mean difference.

    The pairing is what makes this honest: same dimension, same probes, same
    alpha, same judge — only the vectors change. Under the null ("the vectors
    carry nothing"), the sign of each dim's difference is a coin flip.
    """
    d = np.asarray(diffs, dtype=float)
    if not len(d) or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(b, len(d))) * d).mean(1)
    return float((np.abs(null) >= abs(obs)).mean())


def compare(real_path, null_path, seed=0):
    """Paired real-vs-null comparison over the dims present in both."""
    R, N = _rows(real_path), _rows(null_path)
    keys = [k for k in R if k in N]
    dz = [R[k].get("z", 0.0) - N[k].get("z", 0.0) for k in keys]
    de = [abs(R[k].get("effect", 0.0)) - abs(N[k].get("effect", 0.0)) for k in keys]
    dg = [int(_green(R[k])) - int(_green(N[k])) for k in keys]
    survivors = [k for k in keys if _green(R[k]) and not _green(N[k])]
    both = [k for k in keys if _green(R[k]) and _green(N[k])]
    return dict(
        keys=keys, survivors=survivors, both=both,
        green_real=sum(1 for k in keys if _green(R[k])),
        green_null=sum(1 for k in keys if _green(N[k])),
        dz=float(np.mean(dz)), p_z=_perm_p(dz, seed),
        de=float(np.mean(de)), p_e=_perm_p(de, seed),
        dg=int(sum(dg)), p_g=_perm_p(dg, seed),
        R=R, N=N,
    )


def _r2():
    f = config.ROOT / "v2_r2.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="real-vector fidelity report")
    ap.add_argument("--null", default=None, help="null-vector fidelity report")
    ap.add_argument("--kin-r", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="do not write scorecard_v2.md")
    args = ap.parse_args()

    corners = ([("custom", args.real, args.null)] if args.real and args.null
               else list(CORNERS))
    have = []
    for label, rf, nf in corners:
        rp, np_ = config.VEC_DIR / rf, config.VEC_DIR / nf
        if rp.exists() and np_.exists():
            have.append((label, rp, np_))
        else:
            miss = [p.name for p in (rp, np_) if not p.exists()]
            print(f"[skip] {label}: missing {', '.join(miss)}")
    if not have:
        print("nothing to compare — run steering.fidelity / steering.null_control first")
        return

    print(f"{'corner':<22}{'n':>5}{'green':>7}{'rank0':>7}{'top3':>6}"
          f"{'sign':>6}{'mean|eff|':>11}{'kin-green':>11}")
    table = []
    for label, rp, np_ in have:
        for tag, p in (("REAL", rp), ("RAND", np_)):
            s = stats(p)
            kg = sum(1 for d in kin_green(p, args.kin_r).values()
                     if d["z_kin"] >= Z_GREEN and d["sign_ok"])
            row = dict(corner=f"{tag} x {label}", file=p.name, kin=kg, **s)
            table.append(row)
            print(f"{row['corner']:<22}{s['n']:>5}{s['green']:>7}{s['rank0']:>7}"
                  f"{s['top3']:>6}{s['sign']:>6}{s['mae']:>11.3f}{kg:>11}")

    cmps = []
    for label, rp, np_ in have:
        c = compare(rp, np_, args.seed)
        c["label"] = label
        cmps.append(c)
        print(f"\n[{label}] paired, {len(c['keys'])} dims — real minus random:")
        print(f"  green count   {c['dg']:+d}      p = {c['p_g']:.4f}")
        print(f"  mean z        {c['dz']:+.3f}   p = {c['p_z']:.4f}")
        print(f"  mean |effect| {c['de']:+.4f}  p = {c['p_e']:.4f}")
        print(f"  survive the strict rule (green real, dead random): "
              f"{len(c['survivors'])}   also green on noise: {len(c['both'])}")

    if args.no_write:
        return

    head = cmps[0]
    r2 = _r2()
    kinR = kin_green(have[0][1], args.kin_r)
    kinN = kin_green(have[0][2], args.kin_r)
    meta = _meta(have[0][1])
    lines = [
        "# Scorecard v2 — dials that survive the null control",
        "",
        f"Generated by `python -m steering.falsify_report` from "
        f"`{have[0][1].name}` vs `{have[0][2].name}`.",
        f"Regime: alpha={meta.get('alpha')} · layer {meta.get('layers')} · "
        f"mode {meta.get('mode')} · probes/dim {meta.get('n_probes')} · "
        f"vectors `{meta.get('vectors')}`.",
        "",
        "**Rule (v2):** a dial is real only if it is green (z ≥ 1.5, correct sign) "
        "on the real vectors **and dead on random gaussian vectors of the same "
        "shape**. Greens that also fire on noise detect degradation, not meaning.",
        "",
        f"**{len(head['survivors'])} dials of {len(head['keys'])} survive.** "
        f"({head['green_real']} green on real, {head['green_null']} on random, "
        f"{len(head['both'])} on both.)",
        "",
        "| corner | green | rank0 | top3 | sign_ok | mean\\|eff\\| | kin-green |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['corner']} | {row['green']}/{row['n']} | {row['rank0']} "
                     f"| {row['top3']} | {row['sign']} | {row['mae']:.3f} | {row['kin']} |")
    lines += [
        "",
        "Counts alone are weak (104 dims, a z ≥ 1.5 cut, ~7 greens expected by "
        "chance). The comparison is therefore also paired — same dim, same probe, "
        "same alpha, only the vectors change — with a sign-flip permutation test:",
        "",
        "| paired statistic (real − random) | value | p |",
        "|---|---|---|",
    ]
    for c in cmps:
        lines += [
            f"| green count · {c['label']} | {c['dg']:+d} | {c['p_g']:.4f} |",
            f"| mean z · {c['label']} | {c['dz']:+.3f} | {c['p_z']:.4f} |",
            f"| mean \\|effect\\| · {c['label']} | {c['de']:+.4f} | {c['p_e']:.4f} |",
        ]
    lines += [
        "",
        "## The surviving dials",
        "",
        "`R²` is the held-out readability of that axis for the judge that scored "
        "it (`v2_r2.json`): a dial the judge cannot read is a dial we cannot "
        "verify, whatever the vector does.",
        "",
        "| dim | z real | z random | rank | z_kin | R² held-out |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(head["survivors"], key=lambda k: -head["R"][k]["z"]):
        kz = kinR.get(k, {}).get("z_kin", float("nan"))
        lines.append(f"| 🟢 {k} | {head['R'][k]['z']:+.1f} | {head['N'][k]['z']:+.1f} "
                     f"| {head['R'][k]['rank']} | {kz:+.1f} | {r2.get(k, '—')} |")
    if head["both"]:
        lines += [
            "",
            "## Green on noise too — degradation detectors, not dials",
            "",
            "| dim | z real | z random | R² held-out |",
            "|---|---|---|---|",
        ]
        for k in sorted(head["both"], key=lambda k: -head["R"][k]["z"]):
            lines.append(f"| ⚠ {k} | {head['R'][k]['z']:+.1f} | "
                         f"{head['N'][k]['z']:+.1f} | {r2.get(k, '—')} |")
    lines += [
        "",
        f"Kinship-corrected cross-check (`analyze.py`, |r| > {args.kin_r}): "
        f"{sum(1 for d in kinR.values() if d['z_kin'] >= Z_GREEN and d['sign_ok'])} "
        f"green on real vs "
        f"{sum(1 for d in kinN.values() if d['z_kin'] >= Z_GREEN and d['sign_ok'])} "
        "on random — same verdict from a different correction.",
        "",
        "*Read `STATUS.md` for what this does and does not license us to claim.*",
    ]
    SCORECARD_V2.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nscorecard -> {SCORECARD_V2}")


if __name__ == "__main__":
    main()
