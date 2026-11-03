"""
sae_concepts.py — The honest version of sae_map: test each axis ACROSS TOPICS.

sae_map.py found 104/104 axes "native", and that was the warning sign: each
pole of the CAA stimuli is built around ONE anchor scene (economic value + =
world GDP, - = landfills; luminosity + = quasars, - = dark forests), so any
topic feature separates the poles. Distinguishable poles are not an axis.

Here the axis must hold over the 5,578 concepts of the table (dataset_Y_human,
the same scores the translator was trained on), which span every topic. Each
concept is read inside "La idea de {c}." (mean over the tokens after the
prefix). Concepts are split in two fixed halves:

  feature   best single SAE feature chosen on half A (|r| with the axis score),
            judged by its correlation on half B (held out, never seen)
  probe     ridge regression on the whole residual, fit on A, r on B: is the
            axis in the model at all, in one feature or spread over many?
  stim      the feature sae_map picked from the pole stimuli: does it follow
            the axis across concepts, or only the anchor topic?
  caa       the control vector SteeredLlama injects (layer 15): projection of
            each concept onto it vs the axis score, over all concepts

The 104 scores share big common factors (PC1 = valence), so each axis is also
tested on its UNIQUE part (score minus what the other 103 axes explain).
Verdicts: SPECIFIC-FEATURE, SPECIFIC-SPREAD, SHARED-ONLY, WEAK, FLAT (see the
report header).

  python -m steering.sae_concepts --cache X.npz # GPU once, then CPU from X.npz
  python -m steering.sae_concepts --report-only # rebuild the report (CPU)
"""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import config
from . import sae_map as S

PREFIX = "La idea de"
R_PROBE = 0.4                       # raw axis readable from the residual
R_UNIQUE_FEAT = 0.2                 # one feature tracks the axis-specific part
R_UNIQUE_PROBE = 0.3                # the residual carries the axis-specific part
MIN_STD = 0.05
MIN_FIRE = 0.01                     # feature must fire on >= 1% of half A
SPLIT_SEED = 7
CAA_LAYER = 15


def _paths(bk, clean=False):
    """clean=True: the run on the cleaned table (clean_table.py)."""
    sfx = "_clean" if clean else ""
    return (config.VEC_DIR / f"sae_concepts_{bk['tag']}{sfx}.json",
            config.ROOT / f"SAE_CONCEPTS_{bk['tag'].upper()}{sfx.upper()}.md")


def embed_concepts(concepts, bk, batch=64):
    """-> feats (n, F) float16, resid (n, H) float32, both mean-pooled over the
    tokens after PREFIX."""
    import torch
    tok, model = S._load_lm(bk)
    sae = S._load_sae(bk, model.device)
    mode = "plain"                  # chosen by sae_map's fit check (plain < decnorm FVU)
    n_pre = len(tok(PREFIX)["input_ids"])          # BOS + prefix tokens
    layer = bk["layer"]
    tok.padding_side = "right"
    feats, resid = [], []
    with torch.no_grad():
        for i in range(0, len(concepts), batch):
            texts = [f"{PREFIX} {c}." for c in concepts[i:i + batch]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
            mask = enc["attention_mask"].clone()
            mask[:, :n_pre] = 0
            m = mask.unsqueeze(-1).float()
            f = S._encode(sae, hs, mode)
            feats.append(((f * m).sum(1) / m.sum(1)).half().cpu())
            resid.append(((hs * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(feats).numpy(), torch.cat(resid).numpy()


def _corr_cols(X, y):
    """Pearson r of every column of X with y (X float32)."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    return (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)


def _ridge_r(Xa, ya, Xb, yb, lams=(1e2, 1e3, 1e4, 1e5)):
    """Ridge fit on A (lambda by 2-fold inside A), Pearson r on B."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Za, Zb = (Xa - mu) / sd, (Xb - mu) / sd

    def fit(Z, y, lam):                       # dual form: n < d
        K = Z @ Z.T
        a = np.linalg.solve(K + lam * np.eye(len(Z)), y - y.mean())
        return Z.T @ a, y.mean()

    h = len(Za) // 2
    best = max(lams, key=lambda lam: np.corrcoef(
        Za[h:] @ fit(Za[:h], ya[:h], lam)[0], ya[h:])[0, 1])
    w, b = fit(Za, ya, best)
    return float(np.corrcoef(Zb @ w + b, yb)[0, 1]), best


def _p_from_r(r, n):
    """Two-sided p for a held-out Pearson r (normal approx of Fisher z)."""
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def _bh(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    q = ps[order] * len(ps) / np.arange(1, len(ps) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def unique_parts(Y):
    """Each axis minus what the other 103 explain (least squares on z-scores).
    The 104 scores share big common factors — PC1 (~18%) is valence: dignity,
    harmony vs arsenic, sewer — so a feature that tracks a raw axis may only
    track that factor. The residual is what is specific to the axis."""
    Z = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    U = np.empty_like(Z)
    shared = np.empty(Z.shape[1])
    for i in range(Z.shape[1]):
        X = np.delete(Z, i, 1)
        b, *_ = np.linalg.lstsq(X, Z[:, i], rcond=None)
        U[:, i] = Z[:, i] - X @ b
        shared[i] = 1 - U[:, i].var() / Z[:, i].var()
    return U, shared


def _best_feature(FA, FB, alive, yA, yB):
    rA = np.where(alive, _corr_cols(FA, yA), 0.0)
    top = np.argsort(-np.abs(rA))[:5]
    rB = _corr_cols(FB[:, top], yB)
    feats = [{"f": int(f), "rA": float(rA[f]), "rB": float(r)} for f, r in zip(top, rB)]
    r = feats[0]["rB"] * np.sign(feats[0]["rA"])            # sign-consistent
    return feats, float(r), (_p_from_r(r, len(yB)) if r > 0 else 1.0)


def analyze(feats, resid, Y, names, stim_top, caa):
    n = len(Y)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    FA = np.log1p(feats[A].astype(np.float32))
    FB = np.log1p(feats[B].astype(np.float32))
    alive = (feats[A] > 0).mean(0) >= MIN_FIRE
    U, shared = unique_parts(Y)
    rows = []
    for i, k in enumerate(names):
        y, u = Y[:, i], U[:, i]
        row = {"dim": k, "std": float(y.std()), "shared": float(shared[i])}
        row["features"], row["feat_rB"], row["feat_p"] = _best_feature(FA, FB, alive, y[A], y[B])
        row["probe_rB"], row["probe_lam"] = _ridge_r(resid[A], y[A], resid[B], y[B])
        row["ufeatures"], row["ufeat_rB"], row["ufeat_p"] = _best_feature(FA, FB, alive, u[A], u[B])
        row["uprobe_rB"], _ = _ridge_r(resid[A], u[A], resid[B], u[B])
        sf = stim_top.get(k)
        if sf is not None:
            row["stim_f"] = sf
            row["stim_r"] = float(_corr_cols(np.log1p(feats[:, [sf]].astype(np.float32)), y)[0])
        if caa is not None:
            proj = resid @ caa[i]
            row["caa_r"] = float(np.corrcoef(proj, y)[0, 1])
            row["caa_ur"] = float(np.corrcoef(proj, u)[0, 1])
        rows.append(row)
        print(f"  {k:<34} shared={row['shared']:.2f} probe r_B={row['probe_rB']:+.2f} "
              f"unique: probe {row['uprobe_rB']:+.2f} feat {row['ufeat_rB']:+.2f} "
              f"caa {row.get('caa_ur', float('nan')):+.2f}")
    for key in ("feat", "ufeat"):
        for r, qq in zip(rows, _bh([r[f"{key}_p"] for r in rows])):
            r[f"{key}_q"] = float(qq)
    # a feature is the native dial of ONE axis at most: unique parts still
    # correlate a little, so the same feature can top several of them
    owner = {}
    for r in rows:
        f = r["ufeatures"][0]["f"]
        if f not in owner or r["ufeat_rB"] > owner[f]["ufeat_rB"]:
            owner[f] = r
    for r in rows:
        f = r["ufeatures"][0]["f"]
        r["ufeat_claimed_by"] = sum(x["ufeatures"][0]["f"] == f for x in rows)
        r["verdict"] = ("FLAT" if r["std"] < MIN_STD
                        else "SPECIFIC-FEATURE" if r["ufeat_rB"] >= R_UNIQUE_FEAT
                        and r["ufeat_q"] < 0.05 and owner[f] is r
                        else "SPECIFIC-SPREAD" if r["uprobe_rB"] >= R_UNIQUE_PROBE
                        else "SHARED-ONLY" if r["probe_rB"] >= R_PROBE
                        else "WEAK")
    return rows, {"n": n, "n_A": len(A), "n_B": len(B)}


def fetch_labels(fs, bk):
    labels_json = S._paths(bk)[1]
    labels = (json.loads(labels_json.read_text(encoding="utf-8"))
              if labels_json.exists() else {})
    for f in sorted(set(fs)):
        if str(f) in labels:
            continue
        try:
            with urllib.request.urlopen(
                    f"https://www.neuronpedia.org/api/feature/{bk['np_source']}/{f}",
                    timeout=20) as r:
                j = json.loads(r.read().decode("utf-8"))
            ex = j.get("explanations") or []
            labels[str(f)] = ex[0]["description"].strip() if ex else ""
        except Exception as e:
            labels[str(f)] = f"(no label: {type(e).__name__})"
        time.sleep(0.15)
    labels_json.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


def report(data, bk, labels):
    rows = data["rows"]
    _, out_md = _paths(bk, data.get("clean", False))

    def lab(f):
        s = labels.get(str(f), "")
        return f"#{f} {s}" if s and not s.startswith("(no label") else f"#{f}"

    verdicts = ("SPECIFIC-FEATURE", "SPECIFIC-SPREAD", "SHARED-ONLY", "WEAK", "FLAT")
    surv = [r for r in rows if r["dim"][:4] in S.SURVIVORS]
    rest = [r for r in rows if r["dim"][:4] not in S.SURVIVORS]

    def count(rs):
        return " · ".join(f"{v} {sum(r['verdict'] == v for r in rs)}" for v in verdicts)

    def mean(rs, key, absval=True):
        v = [abs(r[key]) if absval else r[key] for r in rs if key in r and not np.isnan(r[key])]
        return f"{np.mean(v):.2f}" if v else "—"

    md = [f"# Concept test — do the 104 axes hold across topics in "
          f"{data['model'].split('/')[-1]}?", "",
          f"Generated by `python -m steering.sae_concepts` · layer {data['layer']} · "
          f"{bk['desc']} · {data['split']['n']} concepts from "
          f"{'the CLEANED table (`TABLE_AUDIT.md`; missing cells filled with the axis mean)' if data.get('clean') else '`dataset_Y_human.npy`'}, read as "
          f"«{PREFIX} …», split {data['split']['n_A']} / {data['split']['n_B']} "
          f"(seed {SPLIT_SEED}); everything chosen on half A, judged on half B.", "",
          "**Why this exists.** `sae_map.py` called all 104 axes native: each pole of the "
          "CAA stimuli is one anchor topic (economic value + = world GDP, − = landfills), "
          "and any topic feature separates two topics. A first concept pass then looked "
          "good for the wrong reason: the 104 scores share big common factors (PC1, 18% "
          "of the variance, is valence — dignity, harmony vs arsenic, sewer; each axis is "
          "on average 53% predictable from the other 103), and a couple of generic "
          "\"emotional\" features tracked that factor for dozens of axes. So the verdict "
          "below is on each axis' **unique part**: the score minus what the other 103 axes "
          "explain.", "",
          "| verdict | meaning |", "|---|---|",
          f"| SPECIFIC-FEATURE | one SAE feature tracks the axis-specific part (held-out r ≥ "
          f"{R_UNIQUE_FEAT}, BH q < 0.05) — a native dial |",
          f"| SPECIFIC-SPREAD | no single feature, but the residual carries the "
          f"axis-specific part (ridge r ≥ {R_UNIQUE_PROBE}) |",
          f"| SHARED-ONLY | the raw axis is readable (ridge r ≥ {R_PROBE}) but only through "
          "what it shares with other axes |",
          "| WEAK | neither |",
          f"| FLAT | scores barely vary (std < {MIN_STD}), untestable |", "",
          "## Summary", "",
          f"- All 104: {count(rows)}",
          f"- The 11 Llama survivors ⭐: {count(surv)}",
          f"- The other {len(rest)}: {count(rest)}",
          f"- Raw axis from the residual: mean ridge r_B {mean(rows, 'probe_rB', False)}; "
          f"axis-specific part: {mean(rows, 'uprobe_rB', False)}.",
          f"- Stimulus-picked features (`sae_map`) vs the raw axis over concepts: mean |r| "
          f"{mean(rows, 'stim_r')} — they tracked the anchor topic, not the axis.",
          f"- The CAA control vectors SteeredLlama injects, vs the axis-specific part: mean "
          f"|r| survivors {mean(surv, 'caa_ur')}, others {mean(rest, 'caa_ur')} "
          f"(raw axis: {mean(surv, 'caa_r')} / {mean(rest, 'caa_r')}).", "",
          "## Per axis", "",
          "| axis | verdict | shared | ridge r_B raw / unique | best feature for the unique part "
          "(r_A → r_B) | caa r raw / unique |",
          "|---|---|---|---|---|---|"]
    order = {v: i for i, v in enumerate(verdicts)}
    for r in sorted(rows, key=lambda r: (order[r["verdict"]], -r["ufeat_rB"])):
        star = " ⭐" if r["dim"][:4] in S.SURVIVORS else ""
        b = r["ufeatures"][0]
        md.append(f"| {r['dim']}{star} | {r['verdict']} | {r['shared']:.2f} | "
                  f"{r['probe_rB']:+.2f} / {r['uprobe_rB']:+.2f} | {lab(b['f'])} "
                  f"({b['rA']:+.2f} → {b['rB']:+.2f}) | "
                  f"{r.get('caa_r', float('nan')):+.2f} / {r.get('caa_ur', float('nan')):+.2f} |")
    md += ["", "`shared` = fraction of the axis explained by the other 103 axes.",
           "⭐ = one of the 11 dials that survived the null control on Llama-3.1-8B.",
           "Axis scores come from an LLM acting as a measuring instrument (README), so "
           "they carry its noise; the unique part carries proportionally more of it, which "
           "lowers every unique r."]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(S.BACKENDS), default="llama")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cache", help="npz with pooled activations: written after encoding, "
                                    "read instead of the GPU when it exists (~450 MB)")
    ap.add_argument("--clean", action="store_true",
                    help="use the cleaned table (clean_table.py); the cache stays the "
                         "original rows and is subset by source_rows")
    args = ap.parse_args()
    bk = S.BACKENDS[args.backend]
    out_json, _ = _paths(bk, args.clean)

    if not args.report_only:
        meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
        concepts, names = meta["concepts"], meta["dimension_names"]
        Y = np.clip(np.load(config.DATA_Y_FILE), 0, 1)
        stim_json = S._paths(bk)[0]
        stim_top = ({r["dim"]: r["top"][0]["f"]
                     for r in json.loads(stim_json.read_text(encoding="utf-8"))["rows"]}
                    if stim_json.exists() else {})
        caa = None
        if args.backend == "llama" and config.CONTROL_VECTORS_FILE.exists():
            cv = np.load(config.CONTROL_VECTORS_FILE)
            layers = json.loads(config.CONTROL_META_FILE.read_text(encoding="utf-8"))["target_layers"]
            caa = cv[layers.index(CAA_LAYER)].astype(np.float32)
            caa /= np.linalg.norm(caa, axis=1, keepdims=True)
        if args.cache and Path(args.cache).exists():
            z = np.load(args.cache)
            feats, resid = z["feats"], z["resid"]
            print(f"📦 activations from {args.cache}")
        else:
            print(f"⏳ {bk['model']} on {len(concepts)} concepts")
            t0 = time.time()
            feats, resid = embed_concepts(concepts, bk)
            print(f"✅ encoded in {time.time() - t0:.0f}s")
            if args.cache:
                np.savez(args.cache, feats=feats, resid=resid)
        if args.clean:
            from .clean_table import META_CLEAN, load_clean
            src = json.loads(META_CLEAN.read_text(encoding="utf-8"))["source_rows"]
            _, Y, _, _ = load_clean(fill="mean")
            feats, resid = feats[src], resid[src]
        rows, split = analyze(feats, resid, Y, names, stim_top, caa)
        data = {"model": bk["model"], "layer": bk["layer"], "split": split, "rows": rows,
                "clean": args.clean}
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"💾 {out_json}")
    else:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    fs = [r["ufeatures"][0]["f"] for r in data["rows"]] + \
         [r["stim_f"] for r in data["rows"] if "stim_f" in r]
    report(data, bk, fetch_labels(fs, bk))


if __name__ == "__main__":
    main()
