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

Stage 2 rebuilt 2026-08-28 (index-keyed + anchored + batched + targeted
retries) after the June run starved at 10/1200 labels in 7h — see the
stage-2 banner below for the specifics.

Stage 2 fixed again 2026-09-17: the prompt printed example scores and the
labeler copied them (21% of the v2 matrix was one of four numbers written in
the prompt). No number appears in the prompt any more, a collapsed block is
re-asked instead of believed, and every run audits its own value histogram —
coverage said 1200/1200 with zero rejects while 65% of the matrix was echo or
rail. Artifacts are tagged (--tag): a re-cook writes v3_* and leaves v2 intact,
so the judge behind a published result stays reproducible.
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
TAG = "v2"
SENT = LAB = XF = YF = METAF = STIM = TRANS = R2F = PART = REJ = AUDIT = None


def set_tag(tag, sentences=None):
    """Point every artifact at <tag>_*.

    A re-cook must not overwrite the judge a published result was measured
    with: v2 stays on disk, v3 is written beside it, and the two can be
    compared instead of one replacing the other. `sentences` reuses an
    earlier stage-1 file — same sentences, new labels — so a labeling fix
    reads as an A/B instead of a fresh draw."""
    global TAG, SENT, LAB, XF, YF, METAF, STIM, TRANS, R2F, PART, REJ, AUDIT, LOG
    TAG = tag
    LOG = ROOT / f"cocina_{tag}_{datetime.now():%Y%m%d_%H%M}.log"
    SENT = Path(sentences) if sentences else ROOT / f"{tag}_sentences.json"
    LAB = ROOT / f"{tag}_labeled.json"
    XF, YF = ROOT / f"{tag}_X.npy", ROOT / f"{tag}_Y.npy"
    METAF = ROOT / f"{tag}_meta.json"
    STIM = config.VEC_DIR / f"caa_stimuli_{tag}.json"
    TRANS = ROOT / f"semantic_translator_{tag}.pth"
    R2F = ROOT / f"{tag}_r2.json"
    PART = ROOT / f"{tag}_label_partial.json"
    REJ = ROOT / f"{tag}_label_rejects.json"
    AUDIT = ROOT / f"{tag}_label_audit.json"


set_tag("v2")

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


def _ensure_pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        log(f"loading {config.LLM_NAME} ...")
        _PIPE = pipeline("text-generation", model=config.LLM_NAME,
                         model_kwargs={"dtype": torch.bfloat16, "low_cpu_mem_usage": True},
                         device_map="auto")
    return _PIPE


def _free_pipe():
    """Release the labeler LLM so later stages (MiniLM, 4-bit derive) fit."""
    global _PIPE
    if _PIPE is not None:
        import gc
        import torch
        _PIPE = None
        gc.collect()
        torch.cuda.empty_cache()
        log("labeler LLM released (VRAM freed)")


def llm(prompt, max_new=600, temp=0.0):
    pipe = _ensure_pipe()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    out = pipe(msgs, max_new_tokens=max_new, do_sample=temp > 0,
               temperature=max(temp, 1e-5),
               pad_token_id=pipe.tokenizer.eos_token_id)
    return out[0]["generated_text"][-1]["content"]


def llm_batch(prompts, max_new):
    """Greedy generation for several prompts in one padded forward pass —
    the stage-2 throughput fix (generation on one 4090 is bandwidth-bound;
    batching multiplies aggregate tokens/s almost linearly)."""
    pipe = _ensure_pipe()
    tok = pipe.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                    # decoder-only: pad on the left
    msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": p}]
            for p in prompts]
    outs = pipe(msgs, max_new_tokens=max_new, do_sample=False,
                batch_size=len(msgs), pad_token_id=tok.eos_token_id)
    res = []
    for o in outs:
        if isinstance(o, list):
            o = o[0]
        res.append(o["generated_text"][-1]["content"])
    return res


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
# Rebuilt 2026-08-28. The June version burned up to 6 calls x 1400 tokens per
# sentence and DISCARDED the whole sentence when <90% of dims parsed (name-keyed
# JSON with accents/underscores rarely matched) -> 10/1200 in 7h. Now:
#   . index-keyed JSON (accent-proof), pole ANCHORS in the prompt (the
#     calibrated-instrument idea from 20_score, which v2 had dropped)
#   . batched generation (llm_batch), tight max_new (~14 tok/dim, not 40)
#   . retries ask ONLY for the dims still missing; partial work persists in
#     <tag>_label_partial.json; a sentence is rejected (<tag>_label_rejects
#     .json) only after all retry rounds — nothing is silently thrown away.
#
# 2026-09-17 — the echo fix. The prompt used to print its own example scores
# ('ej. {"0": 0.482, "13": 0.061}' and "tres decimales variados (0.137,
# 0.482, 0.815...)"). The labeler copied them: 0.061 landed on 7.5% of all
# cells, 0.482 on 4.8%, 0.815 on 4.5%, 0.137 on 4.3% — 21.1% of the matrix,
# 825 of 1200 sentences carrying >=20 such cells, 333 distinct values in
# 124,800, one sentence scored with a SINGLE value across all 104 dims.
# Per-dim label variance predicts held-out R2 at r=+0.46, so the echo was
# capping the judge. Now: the format is shown structurally with no number to
# copy, a block that comes back collapsed is re-asked instead of believed,
# and every run audits its own histogram — coverage cannot see this class of
# failure, which is exactly why the August run looked perfect.
BATCH = 8
RETRY_ROUNDS = 3
MIN_DISTINCT_BLOCK = 3        # a block flatter than this is a collapse, not a score
AUDIT_AFTER = 40              # sentences before the first histogram check
MAX_VALUE_SHARE = 10.0        # % of cells one value may hold before it is a warning
MIN_DISTINCT_PER_SENT = 45    # of 104 (the echoing v2 run sat at 33.2)


def _short(s, n=70):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def _label_prompt(text, idxs, dims):
    lines = []
    for i in idxs:
        name, mn, mx = dims[i]
        lab = name.split("_", 1)[-1].replace("_", " ")
        lines.append(f"{i}. {lab} — 0: {_short(mn)} | 1: {_short(mx)}")
    # NOT ONE DIGIT of score below: every number this prompt used to print
    # came back as a label. The JSON shape is shown with placeholders instead.
    return ("Puntúa la FRASE en cada dimensión numerada, entre su polo mínimo "
            "y su polo máximo.\n"
            "Reglas:\n"
            "· Cada puntuación es un decimal de tres cifras, medido para ESA "
            "dimensión y ESA frase.\n"
            "· Recorre todo el rango; reserva los valores extremos para polos "
            "absolutos.\n"
            "· Valor alto SOLO si la frase evoca claramente ese polo máximo.\n"
            "· Si la dimensión no trata de la frase, puntúa bajo — pero nunca "
            "cero, y nunca el mismo valor bajo dos veces seguidas.\n"
            "· Puntúa cada dimensión por separado: una lista de valores "
            "repetidos no es una medición.\n\n"
            f"FRASE: «{text}»\n\nDIMENSIONES:\n" + "\n".join(lines) +
            "\n\nResponde SOLO un objeto JSON, sin texto extra, con TODAS las "
            "claves numéricas mostradas y esta forma exacta:\n"
            '{"<índice>": <decimal>, "<índice>": <decimal>, ...}')


def _parse_indexed(txt, want, strict=True):
    """{index: score} from model output; JSON first, regex rescue after.
    Tolerates fences, prose, comma decimals. Hard clamp [0,1].

    `strict` (every round but the last) drops a block that comes back
    COLLAPSED — one number repeated down a whole family of dimensions. That
    is the labeler giving up, not measuring, so the retry asks again. On the
    last round whatever parsed is kept: a flat block beats a lost sentence."""
    got = {}
    s, e = txt.find("{"), txt.rfind("}")
    if s != -1 and e > s:
        try:
            for k, v in json.loads(txt[s:e + 1]).items():
                try:
                    ki, vf = int(str(k).strip()), float(v)
                except (ValueError, TypeError):
                    continue
                if ki in want:
                    got[ki] = min(1.0, max(0.0, vf))
        except Exception:
            pass
    if len(got) < len(want):
        for k, v in re.findall(
                r'["\']?(\d{1,3})["\']?\s*[:=]\s*([01]?[.,]\d+|[01])\b', txt):
            ki = int(k)
            if ki in want and ki not in got:
                got[ki] = min(1.0, max(0.0, float(v.replace(",", "."))))
    if (strict and len(got) >= 8
            and len({round(v, 3) for v in got.values()}) < MIN_DISTINCT_BLOCK):
        return {}
    return got


def label_audit(rows):
    """Value histogram of a labeled set — the check coverage cannot do.

    The August run reported 1200/1200 labeled and zero rejects while 21% of its
    cells were numbers copied from the prompt and one sentence came back with a
    single distinct value across all 104 dims. Counting rows says nothing about
    whether they were measured; counting VALUES does."""
    vals = np.array([v for r in rows for v in r["scores"].values()], dtype=np.float64)
    if not vals.size:
        return {}
    uniq, counts = np.unique(np.round(vals, 3), return_counts=True)
    order = np.argsort(-counts)[:8]
    per_row = [len({round(v, 3) for v in r["scores"].values()}) for r in rows]
    return {
        "n_rows": len(rows), "n_cells": int(vals.size),
        "distinct_values": int(uniq.size),
        "pct_floor_le_002": round(float((vals <= 0.02).mean() * 100), 2),
        "pct_ceiling_ge_098": round(float((vals >= 0.98).mean() * 100), 2),
        "top_values": [[float(uniq[i]), int(counts[i]),
                        round(float(counts[i] / vals.size * 100), 2)] for i in order],
        "worst_value_share": round(float(counts[order[0]] / vals.size * 100), 2),
        "distinct_per_sentence": round(float(np.mean(per_row)), 1),
        "flattest_sentence": int(min(per_row)),
    }


def _audit_log(rows, where):
    a = label_audit(rows)
    if not a:
        return a
    log(f"  [audit {where}] {a['n_cells']} cells · {a['distinct_values']} distinct "
        f"· floor {a['pct_floor_le_002']}% · ceiling {a['pct_ceiling_ge_098']}% "
        f"· {a['distinct_per_sentence']}/104 distinct per sentence "
        f"(flattest {a['flattest_sentence']})")
    log("  [audit] top values: " +
        ", ".join(f"{v}x{p}%" for v, _, p in a["top_values"][:5]))
    if a["worst_value_share"] > MAX_VALUE_SHARE:
        log(f"  !! AUDIT: one value holds {a['worst_value_share']}% of all cells "
            f"(limit {MAX_VALUE_SHARE}%) — the labeler is collapsing, not scoring.")
    if a["distinct_per_sentence"] < MIN_DISTINCT_PER_SENT:
        log(f"  !! AUDIT: {a['distinct_per_sentence']}/104 distinct values per "
            f"sentence (limit {MIN_DISTINCT_PER_SENT}) — labels are too flat.")
    return a


def stage_label(dims, sentences, n_label, block_size):
    global BATCH
    done = json.loads(LAB.read_text(encoding="utf-8")) if LAB.exists() else []
    part = {k: {int(i): v for i, v in d.items()}
            for k, d in (json.loads(PART.read_text(encoding="utf-8"))
                         if PART.exists() else {}).items()}
    seen = {r["text"] for r in done}
    todo = [s for s in sentences if s not in seen][:max(0, n_label - len(done))]
    # interleaved blocks: each one mixes all families (consecutive slices put
    # 26 physics dims in front of an emotional sentence -> template-fill spam)
    n_blocks = -(-len(dims) // block_size)
    blocks = [list(range(b, len(dims), n_blocks)) for b in range(n_blocks)]
    t0, ndone0 = time.time(), len(done)

    def flush():
        LAB.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        PART.write_text(json.dumps({k: {str(i): v for i, v in d.items()}
                                    for k, d in part.items()},
                                   ensure_ascii=False), encoding="utf-8")

    open_texts = list(todo)
    audited = False
    for rnd in range(1 + RETRY_ROUNDS):
        tasks = []
        for text in open_texts:
            have = part.get(text, {})
            for blk in blocks:
                miss = [i for i in blk if i not in have]
                if miss:
                    tasks.append((text, miss))
        if not tasks:
            break
        if rnd:
            log(f"  retry {rnd}: {len(tasks)} incomplete blocks")
        i = 0
        while i < len(tasks):
            chunk = tasks[i:i + BATCH]
            prompts = [_label_prompt(t, m, dims) for t, m in chunk]
            mx = max(len(m) for _, m in chunk) * 14 + 40
            try:
                outs = llm_batch(prompts, mx)
            except Exception as ex:       # only VRAM pressure halves the batch
                oom = "out of memory" in str(ex).lower() or "cuda" in str(ex).lower()
                if oom and BATCH > 1:
                    BATCH = max(1, BATCH // 2)
                    log(f"  batch OOM -> BATCH={BATCH}")
                    import torch
                    torch.cuda.empty_cache()
                    continue
                raise
            for (t, m), out in zip(chunk, outs):
                part.setdefault(t, {}).update(
                    _parse_indexed(out, set(m), strict=rnd < RETRY_ROUNDS))
            i += BATCH
            newly = [t for t in dict.fromkeys(t for t, _ in chunk)
                     if t in part and len(part[t]) >= len(dims)]
            for t in newly:
                done.append({"text": t,
                             "scores": {dims[k][0]: v
                                        for k, v in sorted(part.pop(t).items())}})
                open_texts.remove(t)
            if newly:
                flush()
                rate = (len(done) - ndone0) / max(time.time() - t0, 1) * 3600
                log(f"  labeled {len(done)}/{n_label}  "
                    f"({rate:.0f}/h, {len(open_texts)} open)")
                # early histogram: an echoing or collapsing labeler is
                # visible in the first few dozen sentences. Better to see it
                # now than after a 3.5h run that reports perfect coverage.
                if not audited and len(done) >= AUDIT_AFTER:
                    audited = True
                    _audit_log(done, f"first {len(done)}")
    # after all retry rounds: José's 90% gate, but visible instead of silent
    rej = []
    for t in open_texts:
        have = part.pop(t, {})
        if len(have) >= 0.9 * len(dims):
            done.append({"text": t, "scores": {dims[k][0]: v
                                               for k, v in sorted(have.items())}})
        else:
            rej.append({"text": t, "coverage": len(have)})
    if rej:
        REJ.write_text(json.dumps(rej, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    flush()
    a = _audit_log(done, "final")
    if a:
        AUDIT.write_text(json.dumps(a, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log(f"  stage 2 done: {len(done)} labeled, {len(rej)} rejected, "
        f"{(time.time() - t0) / 60:.1f} min")
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
    env = {**os.environ, "PYTHONUTF8": "1", "EMB_TAG": TAG,
           "EMB_STIMULI": str(STIM.resolve())}
    log(f"  derive (steering.derive_vectors, EMB_TAG={TAG}) ...")
    p = subprocess.run([sys.executable, "-m", "steering.derive_vectors"],
                       cwd=ROOT, env=env)
    if p.returncode != 0:
        log("  derive FAILED")
    return p.returncode == 0


# ── stage 6: rigorous translator ─────────────────────────────────────────────
def stage_train(epochs, patience):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)                    # reproducible judge builds
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
    # save with the "net." prefix SemanticTranslator expects, so the v2 judge
    # is drop-in loadable by steering/translator.py (EMB_TRANSLATOR override)
    torch.save({f"net.{k}": v for k, v in net.state_dict().items()}, TRANS)
    # per-dim R² on held-out
    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    yte_np = yte.cpu().numpy()
    r2 = {}
    for j in range(yte_np.shape[1]):
        ss_res = ((yte_np[:, j] - pred[:, j]) ** 2).sum()
        ss_tot = ((yte_np[:, j] - yte_np[:, j].mean()) ** 2).sum() + 1e-9
        r2[json.loads(METAF.read_text(encoding="utf-8"))["dimension_names"][j]] = round(float(1 - ss_res / ss_tot), 3)
    R2F.write_text(json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
    good = sum(1 for v in r2.values() if v >= 0.3)
    log(f"  best val_mse {best:.5f} | dims with R²>=0.3 on held-out: {good}/{len(r2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="v2",
                    help="artifact prefix; a re-cook uses a NEW tag so the "
                         "previous judge stays on disk (default: v2)")
    ap.add_argument("--sentences", default=None,
                    help="reuse an existing stage-1 file (e.g. v2_sentences"
                         ".json): same sentences, new labels = clean A/B")
    ap.add_argument("--n-label", type=int, default=None,
                    help="override how many sentences stage 2 labels (bench the "
                         "histogram on a few dozen before committing hours)")
    ap.add_argument("--audit", default=None,
                    help="print the value histogram of a labeled file and exit")
    args = ap.parse_args()

    if args.audit:
        rows = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        a = label_audit(rows)
        print(f"{args.audit}: {a['n_rows']} rows · {a['n_cells']} cells")
        print(f"  distinct values      {a['distinct_values']}")
        print(f"  worst value share    {a['worst_value_share']}%")
        print(f"  floor <=0.02         {a['pct_floor_le_002']}%")
        print(f"  ceiling >=0.98       {a['pct_ceiling_ge_098']}%")
        print(f"  distinct per sentence {a['distinct_per_sentence']}/104 "
              f"(flattest {a['flattest_sentence']})")
        print("  top: " + ", ".join(f"{v}x{p}%" for v, _, p in a["top_values"]))
        return

    set_tag(args.tag, args.sentences)

    if args.smoke:
        K, N_LABEL, BLOCK, EPOCHS, PAT, NDIMS = 4, 12, 26, 60, 12, 4
    else:
        # ~1200 sentences × 8 interleaved blocks, batched -> ~5h measured
        K, N_LABEL, BLOCK, EPOCHS, PAT, NDIMS = 24, 1200, 13, 1500, 40, None

    if args.n_label:
        N_LABEL = args.n_label
    keep_awake()
    dims = load_dims()
    if NDIMS:
        dims = dims[:NDIMS]
    log(f"COCINA [{TAG}] {'[SMOKE]' if args.smoke else '[FULL]'} | "
        f"dims {len(dims)} | K {K} | label {N_LABEL} | log {LOG.name}")

    run = set(args.only) if args.only else {1, 2, 3, 4, 5, 6}
    if args.sentences and 1 in run:
        log(f"STAGE 1 skipped — reusing {SENT.name} (same sentences, new labels)")
        run.discard(1)
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
        _free_pipe()                       # make room for MiniLM + 4-bit derive
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
