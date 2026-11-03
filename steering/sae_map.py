"""
sae_map.py — Do the 104 human axes exist as the model's OWN features?

Bottom-up check of the humanized space. Instead of asking "does the model carry
MY axis?" (CAA on 104 human-defined dimensions, 11 survivors on Llama), we let a
sparse autoencoder (Llama Scope on Llama-3.1-8B layer 15, or Gemma Scope on
Gemma-2-2B layer 12) split the activations into tens of thousands of native features, then ask which native
features separate the two poles of each axis.

For each of the 104 dims, the pole sentences (caa_stimuli v1 + v2, merged and
deduplicated) are run through the model; layer-L residuals are encoded by the
SAE and mean-pooled per sentence (BOS excluded). Then, per dim:

  contrast   Cohen's d of every feature, pos vs neg pole, with a firing floor
             (the feature must fire on >= MIN_FIRE of its winning pole)
  null       label-permutation max |d| (N_PERM shuffles): what the best feature
             looks like when the poles mean nothing
  caa_cos    cosine of the raw mean-difference direction (the CAA vector, in
             Gemma's space) with every decoder row; chance max ~ 0.09
  hubs       features that are top-3 for many dims are generic (sentiment,
             negation, register), not the axis

Verdict per dim: NATIVE (significant, top feature not a hub), HUB-ONLY
(significant but only through generic features) or ABSENT.

  python -m steering.sae_map --smoke          # 3 dims, 20 perms, checks SAE fit
  python -m steering.sae_map                  # all 104 on Llama (GPU, minutes)
  python -m steering.sae_map --backend gemma  # same on Gemma-2-2B + Gemma Scope
  python -m steering.sae_map --labels         # + Neuronpedia labels (network)
  python -m steering.sae_map --report-only    # rebuild SAE_MAP_<BACKEND>.md
"""
import argparse
import json
import time
import urllib.request

import numpy as np

from . import config

# Two backends. Llama is the default: same model and same layer (15) where
# SteeredLlama injects. Llama Scope was trained on the BASE model; the smoke run
# measures how well it reconstructs the Instruct's activations before trusting it.
BACKENDS = {
    "llama": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "load_4bit": True, "layer": 15,
        "sae_repo": "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
        "sae_file": "Llama3_1-8B-Base-L15R-8x/checkpoints/final.safetensors",
        "sae_hparams": "Llama3_1-8B-Base-L15R-8x/hyperparams.json",
        "np_source": "llama3.1-8b/15-llamascope-res-32k", "tag": "llama",
        "desc": "Llama Scope L15R-8x (32k, JumpReLU)",
    },
    "gemma": {
        "model": "google/gemma-2-2b", "load_4bit": False, "layer": 12,
        "sae_repo": "google/gemma-scope-2b-pt-res",
        "sae_file": "layer_12/width_16k/average_l0_82/params.npz",
        "np_source": "gemma-2-2b/12-gemmascope-res-16k", "tag": "gemma",
        "desc": "Gemma Scope layer 12 (16k, L0 82)",
    },
}
MIN_FIRE = 0.25
N_PERM = 1000
FDR_Q = 0.05                                # Benjamini–Hochberg over the 104 dims
HUB_DIMS = 5                                # top-3 in more than this many dims = hub
VERDICTS = ("NATIVE", "HUB-ONLY", "ABSENT", "BLIND")
MIN_CAPTURE = 0.5                           # below: the SAE barely sees the axis -> BLIND
SURVIVORS = ["d009", "d022", "d062", "d035", "d077", "d090", "d045", "d029",
             "d044", "d096", "d084"]        # the 11 Llama dials (scorecard_v2.md)



def _paths(bk, smoke=False):
    """(map json, labels json, report md) for a backend."""
    t = bk["tag"] + ("_smoke" if smoke else "")
    return (config.VEC_DIR / f"sae_map_{t}.json", config.VEC_DIR / f"np_labels_{bk['tag']}.json",
            config.ROOT / f"SAE_MAP_{t.upper()}.md")


def _stimuli():
    merged = {}
    for f in ("caa_stimuli.json", "caa_stimuli_v2.json"):
        for k, v in json.loads((config.VEC_DIR / f).read_text(encoding="utf-8")).items():
            m = merged.setdefault(k, {"pos": [], "neg": []})
            for pole in ("pos", "neg"):
                m[pole] += [s for s in v[pole] if s not in m[pole]]
    return merged


def _load_sae(bk, device):
    """Unified SAE dict: W_enc (H,F), b_enc, W_dec (F,H), b_dec, threshold, scale."""
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(bk["sae_repo"], bk["sae_file"])

    def t(a):
        return torch.as_tensor(a).to(device=device, dtype=torch.float32)

    if p.endswith(".npz"):                                  # Gemma Scope
        z = np.load(p)
        sae = {k: t(z[k]) for k in z.files}
        sae["scale"] = 1.0
    else:                                                   # Llama Scope (lm-saes)
        from safetensors.torch import load_file
        z = load_file(p)
        hp = json.loads(open(hf_hub_download(bk["sae_repo"], bk["sae_hparams"])).read())
        sae = {"W_enc": t(z["encoder.weight"]).T.contiguous(), "b_enc": t(z["encoder.bias"]),
               "W_dec": t(z["decoder.weight"]).T.contiguous(), "b_dec": t(z["decoder.bias"]),
               "threshold": t(hp["jump_relu_threshold"]),
               # "dataset-wise" norm: inputs rescaled so their mean norm is sqrt(d_model)
               "scale": hp["d_model"] ** 0.5 / hp["dataset_average_activation_norm"]["in"]}
    sae["dec_norm"] = sae["W_dec"].norm(dim=1)
    return sae


def _encode(sae, x, mode):
    """JumpReLU. mode 'plain': pre > threshold; 'decnorm': pre*|W_dec| > threshold
    (lm-saes trains with sparsity_include_decoder_norm). The smoke run picks the
    mode that reconstructs best."""
    pre = (x * sae["scale"]) @ sae["W_enc"] + sae["b_enc"]
    gate = pre * sae["dec_norm"] if mode == "decnorm" else pre
    return torch_relu(pre) * (gate > sae["threshold"])


def torch_relu(x):
    return x.clamp_min(0)


def _fit(sae, x, mode):
    """(FVU, L0) of the SAE on token activations x (n, H)."""
    f = _encode(sae, x, mode)
    rec = (f @ sae["W_dec"] + sae["b_dec"]) / sae["scale"]
    fvu = ((x - rec) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    return float(fvu), float((f > 0).sum(1).float().mean())


def _load_lm(bk):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bk["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if bk["load_4bit"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if "gemma" in bk["model"]:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(bk["model"], **kw)
    model.eval()
    return tok, model


def embed_all(sentences, bk, batch=32):
    """-> feats (n, F) mean-pooled SAE acts, resid (n, H) mean-pooled residual,
    W_dec, fit {mode, fvu, l0 per mode} measured on the first batches' tokens."""
    import torch
    tok, model = _load_lm(bk)
    sae = _load_sae(bk, model.device)
    layer = bk["layer"]
    feats, resid, sample = [], [], []
    mode = None
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
            mask = enc["attention_mask"].clone()
            mask[torch.arange(len(mask)), (mask.cumsum(1) == 1).float().argmax(1)] = 0  # BOS
            if mode is None:
                sample.append(hs[mask.bool()])
                if sum(len(s) for s in sample) < 2000 and i + batch < len(sentences):
                    continue
                x = torch.cat(sample)
                fit = {m: _fit(sae, x, m) for m in ("plain", "decnorm")}
                mode = min(fit, key=lambda m: fit[m][0])
                print("   SAE fit (FVU, L0): " + ", ".join(
                    f"{m}={v[0]:.3f}/{v[1]:.0f}" for m, v in fit.items()) + f" -> {mode}")
                # re-run the held batches with the chosen mode
                for j in range(0, i + batch, batch):
                    e = tok(sentences[j:j + batch], return_tensors="pt",
                            padding=True).to(model.device)
                    h = model(**e, output_hidden_states=True).hidden_states[layer + 1].float()
                    mk = e["attention_mask"].clone()
                    mk[torch.arange(len(mk)), (mk.cumsum(1) == 1).float().argmax(1)] = 0
                    m3 = mk.unsqueeze(-1).float()
                    feats.append(((_encode(sae, h, mode) * m3).sum(1) / m3.sum(1)).cpu())
                    resid.append(((h * m3).sum(1) / m3.sum(1)).cpu())
                continue
            m3 = mask.unsqueeze(-1).float()
            feats.append(((_encode(sae, hs, mode) * m3).sum(1) / m3.sum(1)).cpu())
            resid.append(((hs * m3).sum(1) / m3.sum(1)).cpu())
    fit = {"mode": mode, "scale": float(sae["scale"]),
           **{m: {"fvu": v[0], "l0": v[1]} for m, v in fit.items()}}
    return (torch.cat(feats).numpy(), torch.cat(resid).numpy(),
            sae["W_dec"].cpu().numpy(), fit)


def cohen_d(A, B):
    sa, sb = A.std(0, ddof=1), B.std(0, ddof=1)
    sp = np.sqrt(((len(A) - 1) * sa ** 2 + (len(B) - 1) * sb ** 2) / (len(A) + len(B) - 2))
    return (A.mean(0) - B.mean(0)) / (sp + 1e-3)


def gated_d(A, B):
    """d with the firing floor: a feature only counts on the pole where it fires."""
    d = cohen_d(A, B)
    fire_a, fire_b = (A > 0).mean(0), (B > 0).mean(0)
    ok = np.where(d > 0, fire_a >= MIN_FIRE, fire_b >= MIN_FIRE)
    return np.where(ok, d, 0.0)


def perm_null_max(X, perms, n_a, chunk=100):
    """max |gated d| for every permutation, all at once (torch, GPU if present).
    Same statistic as gated_d: row p of M marks which samples go to pole A."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    F2 = X ** 2
    fire = (X > 0).float()
    n, n_b = len(X), len(X) - n_a
    out = []
    for i in range(0, len(perms), chunk):
        P = torch.as_tensor(perms[i:i + chunk], device=dev)
        M = torch.zeros(len(P), n, device=dev)
        M.scatter_(1, P[:, :n_a], 1.0)
        Mb = 1.0 - M
        sa, sb = M @ X, Mb @ X
        qa, qb = M @ F2, Mb @ F2
        ma, mb = sa / n_a, sb / n_b
        va = (qa - n_a * ma ** 2) / (n_a - 1)
        vb = (qb - n_b * mb ** 2) / (n_b - 1)
        sp = torch.sqrt(((n_a - 1) * va + (n_b - 1) * vb).clamp_min(0) / (n - 2))
        d = (ma - mb) / (sp + 1e-3)
        fa, fb = (M @ fire) / n_a, (Mb @ fire) / n_b
        ok = torch.where(d > 0, fa >= MIN_FIRE, fb >= MIN_FIRE)
        out.append(torch.where(ok, d, torch.zeros_like(d)).abs().max(1).values.cpu())
    return torch.cat(out).numpy()


def analyze(feats, resid, W_dec, index, names, n_perm, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Wn = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    rows = []
    for k in names:
        ip, ineg = index[k]
        A, B = feats[ip], feats[ineg]
        d = gated_d(A, B)
        top = np.argsort(-np.abs(d))[:10]
        allx = np.concatenate([A, B])
        perms = np.stack([rng.permutation(len(allx)) for _ in range(n_perm)])
        null = perm_null_max(allx, perms, len(A))
        caa = resid[ip].mean(0) - resid[ineg].mean(0)
        cos = Wn @ (caa / np.linalg.norm(caa))
        ctop = np.argsort(-np.abs(cos))[:5]
        # how much of the axis the SAE sees: decoding is linear, so the pole
        # difference of the reconstructions is (mean feats diff) @ W_dec / scale
        rec = (A.mean(0) - B.mean(0)) @ W_dec / scale
        capture = float(rec @ caa / (np.linalg.norm(rec) * np.linalg.norm(caa) + 1e-9))
        rows.append({
            "dim": k, "n_pos": len(ip), "n_neg": len(ineg),
            "top": [{"f": int(f), "d": float(d[f]),
                     "fire_pos": float((A[:, f] > 0).mean()),
                     "fire_neg": float((B[:, f] > 0).mean())} for f in top],
            "max_d": float(np.abs(d).max()),
            "null_p95": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= np.abs(d).max()).sum()) / (1 + n_perm)),
            "caa_top": [{"f": int(f), "cos": float(cos[f])} for f in ctop],
            "capture": capture,
        })
        print(f"  {k:<34} max|d|={rows[-1]['max_d']:.2f}  null95={rows[-1]['null_p95']:.2f}"
              f"  p={rows[-1]['p']:.3f}  caa_cos={abs(cos[ctop[0]]):.2f}  capture={capture:.2f}")
    # hubs: features in the top-3 of many dims
    count = {}
    for r in rows:
        for t in r["top"][:3]:
            count[t["f"]] = count.get(t["f"], 0) + 1
    hubs = sorted(f for f, c in count.items() if c > HUB_DIMS)
    # BH q-values: 104 tests at p<0.05 would hand ~5 dims a NATIVE by chance
    ps = np.array([r["p"] for r in rows])
    order = np.argsort(ps)
    q = ps[order] * len(ps) / np.arange(1, len(ps) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    for i, r in zip(order, q):
        rows[i]["q"] = float(min(r, 1.0))
    for r in rows:
        sig = r["q"] < FDR_Q
        clean = [t for t in r["top"][:3] if t["f"] not in hubs and abs(t["d"]) > r["null_p95"]]
        r["verdict"] = ("NATIVE" if sig and clean else "HUB-ONLY" if sig
                        else "ABSENT" if r["capture"] >= MIN_CAPTURE else "BLIND")
        r["native"] = [t["f"] for t in clean]
    return rows, {int(f): count[f] for f in hubs}


def fetch_labels(rows, bk):
    labels_json = _paths(bk)[1]
    labels = (json.loads(labels_json.read_text(encoding="utf-8"))
              if labels_json.exists() else {})
    want = {t["f"] for r in rows for t in r["top"][:3]} | {t["f"] for r in rows for t in r["caa_top"][:1]}
    src = bk["np_source"]
    for f in sorted(want):
        if str(f) in labels:
            continue
        try:
            with urllib.request.urlopen(
                    f"https://www.neuronpedia.org/api/feature/{src}/{f}", timeout=20) as r:
                j = json.loads(r.read().decode("utf-8"))
            ex = j.get("explanations") or []
            labels[str(f)] = ex[0]["description"].strip() if ex else ""
        except Exception as e:                      # keep going; label stays missing
            labels[str(f)] = f"(no label: {type(e).__name__})"
        time.sleep(0.15)
    labels_json.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"🏷️  {labels_json} ({len(labels)} labels)")


def report(data, bk):
    rows, hubs = data["rows"], {int(k): v for k, v in data["hubs"].items()}
    _, labels_json, out_md = _paths(bk, data.get("smoke"))
    labels = (json.loads(labels_json.read_text(encoding="utf-8"))
              if labels_json.exists() else {})
    fit = data["fit"]
    chance = (2 * np.log(data["n_features"]) / data["d_model"]) ** 0.5

    def lab(f):
        s = labels.get(str(f), "")
        return f"#{f} {s}" if s else f"#{f}"

    by = {v: [r for r in rows if r["verdict"] == v] for v in VERDICTS}
    surv = [r for r in rows if r["dim"][:4] in SURVIVORS]
    rest = [r for r in rows if r["dim"][:4] not in SURVIVORS]

    def share(rs, v):
        return f"{sum(r['verdict'] == v for r in rs)}/{len(rs)}"

    md = [f"# SAE map — do the 104 axes exist as {data['model'].split('/')[-1]}'s own features?",
          "",
          f"Generated by `python -m steering.sae_map --backend {bk['tag']}` · model "
          f"`{data['model']}` (layer {data['layer']}) · {bk['desc']} · stimuli caa v1+v2 "
          f"merged ({data['n_sentences']} sentences) · {data['n_perm']} label permutations "
          "per dim" + (" · **SMOKE RUN**" if data.get("smoke") else "") + ".", "",
          f"SAE fit on this model's activations (non-BOS tokens of the stimuli): "
          f"FVU **{fit[fit['mode']]['fvu']:.3f}**, L0 {fit[fit['mode']]['l0']:.0f} "
          f"(JumpReLU gate `{fit['mode']}`). FVU = fraction of variance the SAE fails "
          "to reconstruct; above ~0.3 the features below describe the model poorly.", "",
          "> ⚠️ **Read this as a test of the stimuli, not of the axes.** Each pole of the "
          "CAA stimuli is built around one anchor topic (economic value + = world GDP, "
          "− = landfills), so any topic feature separates the poles and every axis comes "
          "out NATIVE. The features picked here correlate |r| ≈ 0.06 with the axis over "
          "the 5,578 concepts of the table. The axis test is `steering/sae_concepts.py` "
          f"→ `SAE_CONCEPTS_{bk['tag'].upper()}.md`.", "",
          "Bottom-up: the SAE splits the residual stream into the model's own features; "
          "an axis is **NATIVE** when some non-generic feature separates its two poles "
          "beyond what shuffled poles produce (max |d| against its permutation null, "
          f"Benjamini–Hochberg q < {FDR_Q} over all axes), **HUB-ONLY** "
          "when only generic features (top-3 in more than "
          f"{HUB_DIMS} dims) do. Otherwise **ABSENT** if the SAE does see the axis "
          f"(`capture` ≥ {MIN_CAPTURE}: cosine between the pole difference of the "
          "reconstructions and of the raw activations) and **BLIND** if it does not — "
          "then the SAE cannot tell us anything about that axis. `caa cos` = best cosine between "
          "the axis' mean-difference direction and any decoder row "
          f"(chance ≈ {chance:.2f}).", "",
          "## Summary", "",
          f"- NATIVE **{len(by['NATIVE'])}** · HUB-ONLY {len(by['HUB-ONLY'])} · "
          f"ABSENT {len(by['ABSENT'])} · BLIND {len(by['BLIND'])} (of {len(rows)})",
          f"- The 11 Llama survivors: NATIVE {share(surv, 'NATIVE')}, "
          f"HUB-ONLY {share(surv, 'HUB-ONLY')}, ABSENT {share(surv, 'ABSENT')}, BLIND {share(surv, 'BLIND')}",
          f"- The other {len(rest)} (dead on Llama): NATIVE {share(rest, 'NATIVE')}, "
          f"HUB-ONLY {share(rest, 'HUB-ONLY')}, ABSENT {share(rest, 'ABSENT')}, BLIND {share(rest, 'BLIND')}",
          "", "## Hub features (generic, shared by many axes)", "",
          "| feature | axes where it is top-3 |", "|---|---|"]
    for f, c in sorted(hubs.items(), key=lambda x: -x[1]):
        md.append(f"| {lab(f)} | {c} |")
    md += ["", "## Per axis", "",
           "| axis | verdict | max\\|d\\| / null95 | q | best native feature (d) | caa cos "
           "| capture |", "|---|---|---|---|---|---|---|"]
    order = {v: i for i, v in enumerate(VERDICTS)}
    for r in sorted(rows, key=lambda r: (order[r["verdict"]], -r["max_d"])):
        star = " ⭐" if r["dim"][:4] in SURVIVORS else ""
        best = next((t for t in r["top"] if t["f"] in r["native"]), None)
        bcell = f"{lab(best['f'])} ({best['d']:+.2f})" if best else "—"
        md.append(f"| {r['dim']}{star} | {r['verdict']} | {r['max_d']:.2f} / "
                  f"{r['null_p95']:.2f} | {r['q']:.3f} | {bcell} | "
                  f"{abs(r['caa_top'][0]['cos']):.2f} | {r['capture']:.2f} |")
    md += ["", "⭐ = one of the 11 dials that survived the null control on Llama-3.1-8B.",
           "Labels come from Neuronpedia (auto-interpretability on mostly English text); "
           "the stimuli are Spanish, so a label can read off-topic while the feature "
           "still splits the poles."]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), default="llama")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    bk = BACKENDS[args.backend]
    out_json = _paths(bk, args.smoke)[0]

    if not args.report_only:
        stim = _stimuli()
        names = list(stim)[:3] if args.smoke else list(stim)
        sents, index = [], {}
        for k in names:
            ip = list(range(len(sents), len(sents) + len(stim[k]["pos"])))
            sents += stim[k]["pos"]
            ineg = list(range(len(sents), len(sents) + len(stim[k]["neg"])))
            sents += stim[k]["neg"]
            index[k] = (ip, ineg)
        print(f"⏳ {bk['model']} + {bk['desc']} on {len(sents)} sentences, {len(names)} dims")
        t0 = time.time()
        feats, resid, W_dec, fit = embed_all(sents, bk)
        print(f"✅ encoded in {time.time() - t0:.0f}s; active features/sentence "
              f"{(feats > 0).sum(1).mean():.0f}")
        n_perm = 20 if args.smoke else N_PERM
        rows, hubs = analyze(feats, resid, W_dec, index, names, n_perm, fit["scale"])
        data = {"model": bk["model"], "layer": bk["layer"], "n_perm": n_perm,
                "smoke": args.smoke, "fit": fit, "n_sentences": len(sents),
                "n_features": int(W_dec.shape[0]), "d_model": int(W_dec.shape[1]),
                "rows": rows, "hubs": hubs}
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"💾 {out_json}")
    else:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    if args.labels:
        fetch_labels(data["rows"], bk)
    report(data, bk)


if __name__ == "__main__":
    main()
