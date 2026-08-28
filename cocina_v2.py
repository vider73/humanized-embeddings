# -*- coding: utf-8 -*-
"""
cocina_v2.py — Rebuild the foundation FROM SCRATCH, with rigor. Sequential,
resumable, one overnight run on the 4090.

Fixes the root causes found 2026-06-15 (see chat):
  · the judge was trained on single-WORD embeddings but used on PARAGRAPHS
    -> here every training input is a SENTENCE (matches inference).
  · noisy single-shot labels -> temp 0 (deterministic), JSON-schema checked,
    hard-clamped to [0,1] (kills the old Y>1 bug at the source).
  · contaminated / uneven CAA poles -> poles are GENERATED as poles, balanced,
    and self-validated against their own label.
  · training hygiene -> shuffled split, early stopping, BEST-model save,
    per-dimension R² reported (so unreadable dims are visible, not hidden).

Dimensions are REUSED from master_dimensions_prompts.json (José's taxonomy);
pruning/redefining them is a separate, human-in-the-loop step.

Stages (each writes a checkpoint; a rerun skips finished stages):
  1 generate : K pole sentences per dim, both poles            -> v2_sentences.json
  2 label    : balanced subset scored on all 104 dims (LLM)    -> v2_labeled.json
  3 embed    : sentences -> multilingual MiniLM X, clamped Y   -> v2_X.npy/v2_Y.npy/v2_meta.json
  4 stimuli  : clean CAA poles from stage 1 (self-validated)   -> vectors/caa_stimuli_v2.json
  5 derive   : control vectors (reuses steering.derive_vectors)-> vectors/control_vectors_caa_white_v2.npy
  6 train    : rigorous translator on SENTENCES                -> semantic_translator_v2.pth + v2_r2.json

  python cocina_v2.py --smoke      # tiny end-to-end (~3 min) — RUN THIS FIRST
  python cocina_v2.py              # full overnight run
  python cocina_v2.py --only 5 6   # rerun specific stages
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering import config

ROOT = config.ROOT
LOG = ROOT / f"cocina_v2_{datetime.now():%Y%m%d_%H%M}.log"
SENT = ROOT / "v2_sentences.json"
LAB = ROOT / "v2_labeled.json"
XF, YF, METAF = ROOT / "v2_X.npy", ROOT / "v2_Y.npy", ROOT / "v2_meta.json"
STIM = config.VEC_DIR / "caa_stimuli_v2.json"
TRANS = ROOT / "semantic_translator_v2.pth"
R2F = ROOT / "v2_r2.json"

SYS = "Eres un analista semántico preciso. Sigues las instrucciones al pie de la letra."
_PIPE = None


def log(m):
    s = f"[{datetime.now():%H:%M:%S}] {m}"
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        log("insomnia on")
    except Exception:
        pass


def llm(prompt, max_new=600, temp=0.0):
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        log(f"loading {config.LLM_NAME} ...")
        _PIPE = pipeline("text-generation", model=config.LLM_NAME,
                         model_kwargs={"dtype": torch.bfloat16, "low_cpu_mem_usage": True},
                         device_map="auto")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    out = _PIPE(msgs, max_new_tokens=max_new, do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                pad_token_id=_PIPE.tokenizer.eos_token_id)
    return out[0]["generated_text"][-1]["content"]


def load_dims():
    """[(name, min_pole, max_pole)] aligned to dataset_metadata dimension order."""
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    master = json.loads((ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))
    by_id = {d.get("id", d.get("name")): d for d in master}
    dims = []
    for n in names:
        sc = by_id.get(n, {}).get("scale", {})
        dims.append((n, sc.get("min", ""), sc.get("max", "")))
    return dims


# ── stage 1: generate pole sentences ─────────────────────────────────────────
def _gen(name, pole_desc, which, k):
    label = name.split("_", 1)[-1]
    prompt = (f"Escribe EXACTAMENTE {k} frases en español, variadas y naturales, "
              f"que ejemplifiquen un grado {which} de «{label}» ({pole_desc}). "
              f"Frases ricas y distintas entre sí, de temas diversos. "
              f"Una frase por línea, sin numerar, sin comillas, sin explicación.")
    txt = llm(prompt, max_new=40 * k)
    lines = [re.sub(r'^\s*[\d\.\-\*\)"]+\s*', "", ln).strip().strip('"')
             for ln in txt.splitlines()]
    lines = [ln for ln in lines if len(ln.split()) >= 3]
    return lines[:k]


def stage_generate(dims, k):
    data = json.loads(SENT.read_text(encoding="utf-8")) if SENT.exists() else {}
    for i, (name, mn, mx) in enumerate(dims):
        if name in data and data[name].get("pos") and data[name].get("neg"):
            continue
        pos = _gen(name, mx, "MÁXIMO", k)
        neg = _gen(name, mn, "MÍNIMO", k)
        data[name] = {"pos": pos, "neg": neg}
        SENT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  gen {i+1}/{len(dims)} {name}: pos {len(pos)} neg {len(neg)}")
    return data


# ── stage 2: label a balanced subset on all dims ─────────────────────────────
def _parse_scores(txt, block):
    try:
        s, e = txt.find("{"), txt.rfind("}") + 1
        d = json.loads(txt[s:e])
    except Exception:
        return {}
    out = {}
    for k in block:
        key = k.split("_", 1)[-1]
        v = d.get(key, d.get(k))
        if isinstance(v, (int, float)):
            out[k] = float(min(1.0, max(0.0, v)))   # hard clamp [0,1]
    return out


def stage_label(dims, sentences, n_label, block_size):
    names = [d[0] for d in dims]
    done = json.loads(LAB.read_text(encoding="utf-8")) if LAB.exists() else []
    seen = {r["text"] for r in done}
    todo = [s for s in sentences if s not in seen][:max(0, n_label - len(done))]
    blocks = [names[i:i + block_size] for i in range(0, len(names), block_size)]
    for j, text in enumerate(todo):
        scores = {}
        for block in blocks:
            labels = ", ".join(b.split("_", 1)[-1] for b in block)
            prompt = (f"Puntúa la frase en cada dimensión de 0.000 a 1.000 "
                      f"(usa todo el rango, decimales). Frase: «{text}». "
                      f"Dimensiones: {labels}. Responde SOLO un objeto JSON "
                      f"{{dimension: valor}}.")
            for _ in range(2):
                scores.update(_parse_scores(llm(prompt, max_new=40 * len(block)), block))
                if all(b in scores for b in block):
                    break
        if len(scores) >= 0.9 * len(names):
            done.append({"text": text, "scores": scores})
            LAB.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        if (j + 1) % 10 == 0:
            log(f"  labeled {len(done)}/{n_label}")
    return done


# ── stage 3: embed + assemble X/Y ────────────────────────────────────────────
def stage_embed(dims):
    from sentence_transformers import SentenceTransformer
    names = [d[0] for d in dims]
    rows = json.loads(LAB.read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    Y = np.array([[float(min(1.0, max(0.0, r["scores"].get(n, 0.5)))) for n in names]
                  for r in rows], dtype=np.float32)
    emb = SentenceTransformer(config.EMBEDDING_MODEL)
    X = emb.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(XF, X); np.save(YF, Y)
    METAF.write_text(json.dumps({"dimension_names": names, "n": len(texts),
                                 "embedding_model": config.EMBEDDING_MODEL,
                                 "texts": texts}, ensure_ascii=False), encoding="utf-8")
    log(f"  X {X.shape} Y {Y.shape}  (Y max {Y.max():.2f}, >1: {(Y>1).sum()})")


# ── stage 4: clean stimuli (self-validated) ──────────────────────────────────
def stage_stimuli(dims):
    sents = json.loads(SENT.read_text(encoding="utf-8"))
    labeled = {r["text"]: r["scores"] for r in json.loads(LAB.read_text(encoding="utf-8"))} \
        if LAB.exists() else {}
    out = {}
    dropped = 0
    for name, _, _ in dims:
        s = sents.get(name, {})
        lab = name.split("_", 1)[-1]
        pos, neg = [], []
        for p in s.get("pos", []):
            v = labeled.get(p, {}).get(name)
            if v is None or v >= 0.55:
                pos.append(p)
            else:
                dropped += 1
        for nseq in s.get("neg", []):
            v = labeled.get(nseq, {}).get(name)
            if v is None or v <= 0.45:
                neg.append(nseq)
            else:
                dropped += 1
        if pos and neg:
            out[name] = {"pos": pos, "neg": neg}
    STIM.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  stimuli: {len(out)} dims  (dropped {dropped} pole sentences that failed self-check)")


# ── stage 5: derive control vectors (reuse the validated tool) ───────────────
def stage_derive():
    env = {**os.environ, "PYTHONUTF8": "1", "EMB_TAG": "v2",
           "EMB_STIMULI": str(STIM.resolve())}
    log("  derive (steering.derive_vectors, EMB_TAG=v2) ...")
    p = subprocess.run([sys.executable, "-m", "steering.derive_vectors"],
                       cwd=ROOT, env=env)
    if p.returncode != 0:
        log("  derive FAILED")
    return p.returncode == 0


# ── stage 6: rigorous translator ─────────────────────────────────────────────
def stage_train(epochs, patience):
    import torch
    import torch.nn as nn
    X = np.load(XF); Y = np.load(YF)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))                       # SHUFFLE before split
    X, Y = X[idx], Y[idx]
    n_tr = int(len(X) * 0.85)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr = torch.tensor(X[:n_tr], device=dev); ytr = torch.tensor(Y[:n_tr], device=dev)
    xte = torch.tensor(X[n_tr:], device=dev); yte = torch.tensor(Y[n_tr:], device=dev)
    net = nn.Sequential(
        nn.Linear(X.shape[1], 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, Y.shape[1]), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, best_state, bad = 1e9, None, 0
    bs = 64
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(xtr[b]), ytr[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xte), yte).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 20 == 0:
            log(f"  ep {ep} val_mse {vl:.5f} best {best:.5f}")
        if bad >= patience:
            log(f"  early stop @ep {ep}"); break
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), TRANS)
    # per-dim R² on held-out
    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    yte_np = yte.cpu().numpy()
    r2 = {}
    for j in range(yte_np.shape[1]):
        ss_res = ((yte_np[:, j] - pred[:, j]) ** 2).sum()
        ss_tot = ((yte_np[:, j] - yte_np[:, j].mean()) ** 2).sum() + 1e-9
        r2[json.loads(METAF.read_text(encoding="utf-8"))["dimension_names"][j]] = round(1 - ss_res / ss_tot, 3)
    R2F.write_text(json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
    good = sum(1 for v in r2.values() if v >= 0.3)
    log(f"  best val_mse {best:.5f} | dims with R²>=0.3 on held-out: {good}/{len(r2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=int, nargs="+", default=None)
    args = ap.parse_args()

    if args.smoke:
        K, N_LABEL, BLOCK, EPOCHS, PAT, NDIMS = 4, 12, 26, 60, 12, 4
    else:
        # ~1200 sentences × 3 blocks ≈ 3600 LLM calls -> fits one night
        K, N_LABEL, BLOCK, EPOCHS, PAT, NDIMS = 24, 1200, 35, 1500, 40, None

    keep_awake()
    dims = load_dims()
    if NDIMS:
        dims = dims[:NDIMS]
    log(f"COCINA v2 {'[SMOKE]' if args.smoke else '[FULL]'} | dims {len(dims)} | "
        f"K {K} | label {N_LABEL} | log {LOG.name}")

    run = set(args.only) if args.only else {1, 2, 3, 4, 5, 6}
    t0 = time.time()

    if 1 in run:
        log("STAGE 1 generate"); stage_generate(dims, K)
    sents_all = []
    if SENT.exists():
        s = json.loads(SENT.read_text(encoding="utf-8"))
        for v in s.values():
            sents_all += v.get("pos", []) + v.get("neg", [])
        sents_all = list(dict.fromkeys(sents_all))      # dedup, keep order
        import random
        random.Random(0).shuffle(sents_all)             # balance label subset across all dims
    if 2 in run:
        log("STAGE 2 label"); stage_label(dims, sents_all, N_LABEL, BLOCK)
    if 3 in run:
        log("STAGE 3 embed"); stage_embed(dims)
    if 4 in run:
        log("STAGE 4 stimuli"); stage_stimuli(dims)
    if 5 in run:
        log("STAGE 5 derive"); stage_derive()
    if 6 in run:
        log("STAGE 6 train"); stage_train(EPOCHS, PAT)

    log(f"DONE in {(time.time()-t0)/60:.1f} min. "
        f"Artifacts: v2_*.json/npy, caa_stimuli_v2.json, "
        f"control_vectors_caa_white_v2.npy, semantic_translator_v2.pth, v2_r2.json")


if __name__ == "__main__":
    main()
