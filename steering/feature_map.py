"""
feature_map.py — The model's own features laid out in the 104-axis human space.

The other direction of sae_concepts.py. There we asked "does the model carry
MY axis?"; here we ask "where do ITS features fall in my space?".

Inputs
  labels     Neuronpedia's auto-interp descriptions of the 32k Llama Scope
             layer-15 features (gpt-4o-mini, English), from the public export
             explanations-only/llama3.1-8b_llamascope-res-32k.zip
  judge v2   semantic_translator_v2.pth: label text -> 104-dim profile. Trained
             on Spanish sentences; MiniLM is multilingual, so English labels
             land in the same space, but only axes with held-out R² >= R2_MIN
             (v2_r2.json) are read as measurements
  behaviour  sae_concepts' cached activations: each feature's correlation with
             every axis over the 5,578 concepts (what the feature DOES, not
             what its label SAYS)
  geometry   the SAE decoder rows: t-SNE of their directions for the layout,
             k-means for regions

Outputs
  FEATURE_MAP.md                      agreement label↔behaviour per axis,
                                      regions, empty axes, unnamed features
  steering/vectors/feature_map_llama.json   per-feature data for the map page

  python -m steering.feature_map --cache <concepts_llama.npz>
"""
import argparse
import json
import time

import numpy as np

from . import config
from . import sae_map as S
from . import sae_concepts as C

LABELS_FILE = config.VEC_DIR / "np_labels_llama_L15_all.json"
JUDGE_V2 = config.ROOT / "semantic_translator_v2.pth"
R2_FILE = config.ROOT / "v2_r2.json"
R2_MIN = 0.3
N_REGIONS = 48
MIN_FIRE = 0.01
OUT_JSON = config.VEC_DIR / "feature_map_llama.json"
OUT_MD = config.ROOT / "FEATURE_MAP.md"
OUT_NPZ = config.VEC_DIR / "feature_map_llama.npz"


def humanize(texts, batch=512):
    import torch
    from sentence_transformers import SentenceTransformer
    from .translator import SemanticTranslator
    emb = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    E = emb.encode(texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=False)
    tr = SemanticTranslator(E.shape[1], 104)
    tr.load_state_dict(torch.load(JUDGE_V2, map_location="cpu", weights_only=True))
    tr.eval()
    with torch.no_grad():
        return tr(torch.tensor(E, dtype=torch.float32)).numpy()


def behaviour(cache, Y):
    """(r_raw, r_unique, alive): each feature's Pearson r with every axis (raw
    score and unique part) over the concepts, from sae_concepts' cache."""
    z = np.load(cache)
    F = np.log1p(z["feats"].astype(np.float32))
    alive = (z["feats"] > 0).mean(0) >= MIN_FIRE
    U, _ = C.unique_parts(Y)

    def corr(Fm, T):
        Fc = Fm - Fm.mean(0)
        Tc = T - T.mean(0)
        num = Fc.T @ Tc
        den = np.sqrt((Fc ** 2).sum(0))[:, None] * np.sqrt((Tc ** 2).sum(0))[None, :]
        return num / (den + 1e-12)

    return corr(F, Y), corr(F, U), alive


def layout(W_dec, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Wn = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    P = PCA(50, random_state=seed).fit_transform(Wn)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    xy = TSNE(2, init="pca", metric="cosine", perplexity=40, random_state=seed).fit_transform(Pn)
    regions = KMeans(N_REGIONS, n_init=4, random_state=seed).fit_predict(Pn)
    return xy, regions


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="sae_concepts --cache npz")
    args = ap.parse_args()
    t0 = time.time()

    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    names = meta["dimension_names"]
    en = json.loads((config.ROOT / "dimension_names_en.json").read_text(encoding="utf-8"))
    Y = np.clip(np.load(config.DATA_Y_FILE), 0, 1)
    r2 = json.loads(R2_FILE.read_text(encoding="utf-8"))
    r2v = np.array([r2.get(n, -1.0) for n in names])
    readable = r2v >= R2_MIN

    labels = {int(k): v for k, v in json.loads(LABELS_FILE.read_text(encoding="utf-8")).items()}
    sae = S._load_sae(S.BACKENDS["llama"], "cpu")
    W_dec = sae["W_dec"].numpy()
    n_feat = len(W_dec)
    has = np.array([bool(labels.get(i)) for i in range(n_feat)])
    texts = [labels.get(i, "") for i in range(n_feat)]

    print(f"⏳ humanizing {has.sum()} labels with judge v2 ({readable.sum()} readable axes)")
    prof = np.full((n_feat, 104), np.nan, np.float32)
    prof[has] = humanize([t for t, h in zip(texts, has) if h])
    print(f"   done {time.time() - t0:.0f}s; behaviour over concepts...")
    r_raw, r_uni, alive = behaviour(args.cache, Y)
    print(f"   alive on concepts: {alive.sum()}; layout (t-SNE + k-means)...")
    xy, regions = layout(W_dec)
    print(f"   done {time.time() - t0:.0f}s")

    # label profile as z-scores across features: "this feature is unusually X"
    pz = (prof - np.nanmean(prof, 0)) / (np.nanstd(prof, 0) + 1e-9)

    # 1. agreement per axis: does a feature whose LABEL says X also FIRE on X concepts?
    both = has & alive
    agree = []
    for i in range(104):
        a = spearman(pz[both, i], r_raw[both, i])
        k = max(20, int(0.01 * both.sum()))
        top_lab = np.argsort(-pz[both, i])[:k]           # features the labels call most X
        hit = float(np.mean(r_raw[both][top_lab, i]))   # their mean behaviour r on X
        agree.append({"dim": names[i], "en": en.get(names[i], names[i]), "r2": float(r2v[i]),
                      "readable": bool(readable[i]), "rho": a, "toplab_r": hit})

    # 2. axis coverage: how many features are strongly about each axis, by label
    #    (readable axes) and by behaviour (unique part)
    cover_lab = (pz[has] > 3).sum(0)
    cover_beh = (np.abs(r_uni[alive]) > 0.15).sum(0)

    # 3. regions: dominant readable axes by mean label z
    reg = []
    for g in range(N_REGIONS):
        m = (regions == g) & has
        mz = np.nanmean(pz[m], 0)
        mz[~readable] = -np.inf
        top = np.argsort(-mz)[:3]
        ex = [texts[j].strip() for j in np.where(m)[0][:6]]
        reg.append({"region": g, "size": int((regions == g).sum()),
                    "axes": [(names[t], float(mz[t])) for t in top], "examples": ex})

    # 4. unnamed features: labelled, alive, but no readable axis stands out
    maxz = np.where(readable[None, :], pz, -np.inf).max(1)
    unnamed = np.where(has & alive & (maxz < 1.0))[0]

    # per-feature records for the page (compact)
    conc = json.loads(C._paths(S.BACKENDS["llama"])[0].read_text(encoding="utf-8"))
    dials = {r["ufeatures"][0]["f"]: r["dim"] for r in conc["rows"]
             if r["verdict"] == "SPECIFIC-FEATURE"}
    ridx = np.where(readable)[0]
    feats = []
    for j in range(n_feat):
        rec = {"i": j, "x": round(float(xy[j, 0]), 2), "y": round(float(xy[j, 1]), 2),
               "g": int(regions[j]), "l": texts[j].strip()[:90]}
        if has[j]:
            t = ridx[np.argsort(-pz[j, ridx])[:3]]
            rec["la"] = [[int(k), round(float(pz[j, k]), 1)] for k in t]
        if alive[j]:
            k = int(np.argmax(np.abs(r_uni[j])))
            rec["ba"] = [k, round(float(r_uni[j, k]), 2)]
        if j in dials:
            rec["dial"] = dials[j]
        feats.append(rec)
    OUT_JSON.write_text(json.dumps({
        "axes": [{"id": n, "en": en.get(n, n), "r2": float(r2v[i]), "readable": bool(readable[i])}
                 for i, n in enumerate(names)],
        "regions": reg, "features": feats}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")

    # full matrices for the interactive page (steering/feature_atlas.py)
    np.savez_compressed(OUT_NPZ, xy=xy.astype(np.float32), regions=regions.astype(np.uint8),
                        label_z=pz.astype(np.float16), beh_r=r_uni.astype(np.float16),
                        alive=alive, has=has, readable=readable)
    (OUT_JSON.with_name("feature_map_llama_summary.json")).write_text(json.dumps({
        "agree": agree, "cover_lab": cover_lab.tolist(), "cover_beh": cover_beh.tolist(),
        "n_unnamed": int(len(unnamed)), "n_alive": int(alive.sum()), "n_labelled": int(has.sum()),
        "dials": {str(k): v for k, v in dials.items()}}, ensure_ascii=False), encoding="utf-8")

    # report
    ag = sorted(agree, key=lambda a: -a["rho"])
    ra = [a for a in agree if a["readable"]]
    md = ["# Feature map — the model's own features in the 104-axis space", "",
          f"Generated by `python -m steering.feature_map` · {has.sum()} Neuronpedia labels of "
          f"Llama Scope L15R-8x (Llama-3.1-8B layer 15) humanized with judge v2 · behaviour "
          f"from {len(Y)} concepts ({alive.sum()} features fire on ≥ {MIN_FIRE:.0%} of them) · "
          f"layout = t-SNE of decoder directions, {N_REGIONS} k-means regions.", "",
          "Two independent readings of each feature: what its **label says** (judge v2 on the "
          "English description, as a z-score against all features) and what it **does** "
          "(correlation of its activation with the axis over the concept table). Judge v2 "
          f"reads only {readable.sum()} axes at held-out R² ≥ {R2_MIN}; the others are shown "
          "but not trusted.", "",
          "## Do labels and behaviour agree?", "",
          f"Spearman ρ across the {both.sum()} features that have a label and fire on "
          "concepts; `top-label r` = mean behaviour r of the 1% of features the labels call "
          "most X (chance ≈ 0).", "",
          f"- Readable axes: mean ρ **{np.mean([a['rho'] for a in ra]):+.2f}**, mean "
          f"top-label r {np.mean([a['toplab_r'] for a in ra]):+.2f}",
          f"- Unreadable axes: mean ρ {np.mean([a['rho'] for a in agree if not a['readable']]):+.2f}, "
          f"mean top-label r {np.mean([a['toplab_r'] for a in agree if not a['readable']]):+.2f}", "",
          "| axis | judge R² | ρ label↔behaviour | top-label r | features by label (z>3) "
          "| features by behaviour (unique \\|r\\|>0.15) |", "|---|---|---|---|---|---|"]
    for a in ag:
        i = names.index(a["dim"])
        mark = "" if a["readable"] else " ·"
        md.append(f"| {a['en']}{mark} | {a['r2']:.2f} | {a['rho']:+.2f} | {a['toplab_r']:+.2f} | "
                  f"{cover_lab[i]} | {cover_beh[i]} |")
    md += ["", "· = judge v2 cannot read this axis (R² < 0.3); its label column is noise.", "",
           "## Regions of the model's feature space", "",
           "Each region = a k-means cluster of decoder directions; its readable axes are the "
           "ones its labels lean to most (mean z).", "",
           "| region | features | leaning (readable axes, mean z) | sample labels |",
           "|---|---|---|---|"]
    for r in sorted(reg, key=lambda r: -r["axes"][0][1]):
        ax = ", ".join(f"{en.get(n, n)} {z:+.1f}" for n, z in r["axes"])
        md.append(f"| {r['region']} | {r['size']} | {ax} | {' · '.join(r['examples'][:3])} |")
    empty = [en.get(names[i], names[i]) for i in np.argsort(cover_beh)[:12]]
    md += ["", "## Where the model has (almost) nothing", "",
           "Axes with the fewest features tracking their unique part over concepts: "
           + ", ".join(empty) + ".", "",
           "## Features the 104 axes cannot name", "",
           f"{len(unnamed)} labelled features fire on concepts but lean to no readable axis "
           "(max z < 1). A sample:", ""]
    rng = np.random.default_rng(0)
    for j in rng.choice(unnamed, size=min(25, len(unnamed)), replace=False):
        md.append(f"- #{j} {texts[j].strip()}")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {OUT_MD} · 💾 {OUT_JSON} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
