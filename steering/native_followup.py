"""
native_followup.py — Two loose ends of native_vectors: temperature and fluency.

1. Temperature. Its "native" feature (#3809, "references to events, suggestions,
   and comparisons") pushed the wrong way (0.41). The cause is the table, not the
   steering: d004's scale runs from absolute zero to the Planck temperature, so
   fire, sun, lava and ovens all score 0.0 (92 % of concepts are 0) while abstract
   virtues get 0.86 and LLM preamble rows ("here is the list of 10 terms…") get
   1.0 — the column tracks string length (r +0.53), and #3809 tracked that.
   The tactile axis d043 (liquid nitrogen → red-hot metal) is sane on physical
   things (fire 0.83, lava 0.93, ice 0.0) and its unique part is tracked by real
   heat features. Candidates, judged on d004's anchors like before:
     fire       SAE #3217 "references to fire and flames" (top on d043's unique part)
     fire_cold  #3217 minus #4167 "cold temperatures and frozen states", a two-pole dial
     probe43    ridge direction on d043's unique part (as `probe` in native_vectors)
   native (#3809) and the random seeds are reused from native_eval_llama.json:
   same probes, same α, greedy, so regenerating them gives the same texts.

2. Fluency. At α 0.3–0.4 every source garbles some words. For every text of the
   replication (plus the neutral, unsteered answers to the same probes) and the
   new temperature texts: per-token NLL of the answer given the prompt under the
   UNSTEERED Llama and under Qwen3-8B (another family, so not a self-preference),
   the share of answer tokens Llama finds very surprising (NLL > 8 nats, what a
   garbled word looks like) and distinct-2 (repetition loops).

  python -m steering.native_followup --cache <concepts_llama.npz>
  python -m steering.native_followup --report-only

The report appends two sections to NATIVE_VECTORS.md; rerunning
`native_vectors --report-only` drops them, so run this --report-only after it.
"""
import argparse
import gc
import json
import time

import numpy as np

from . import config
from . import native_vectors as N
from . import sae_concepts as C
from . import sae_map as S

TEMP_SOURCES = ("fire", "fire_cold", "probe43")
F_FIRE, F_COLD = 3217, 4167
OUT_JSON = config.VEC_DIR / "native_followup_llama.json"
SURPRISE = 8.0


# --- temperature ---------------------------------------------------------------
def temperature_vectors(cache, names):
    i, j = names.index("d004_temperatura"), names.index("d043_temperatura_táctil")
    W = S._load_sae(S.BACKENDS["llama"], "cpu")["W_dec"].numpy().astype(np.float64)
    fire, cold = W[F_FIRE] / np.linalg.norm(W[F_FIRE]), W[F_COLD] / np.linalg.norm(W[F_COLD])
    fc = fire - cold
    U, _ = C.unique_parts(np.clip(np.load(config.DATA_Y_FILE), 0, 1))
    ridge = N._Ridge(np.load(cache)["resid"].astype(np.float64))
    out = {s: np.zeros((104, W.shape[1]), np.float32) for s in TEMP_SOURCES}
    out["fire"][i] = fire
    out["fire_cold"][i] = fc / np.linalg.norm(fc)
    out["probe43"][i] = ridge.direction(U[:, j])
    old = np.load(N.VEC_NPZ)["native"][i]
    cos = {s: {t: float(out[s][i] @ out[t][i]) for t in TEMP_SOURCES} | {"native": float(out[s][i] @ old)}
           for s in TEMP_SOURCES}
    for s in TEMP_SOURCES:
        print(f"  {s:<10} " + "  ".join(f"cos·{t} {v:+.2f}" for t, v in cos[s].items()))
    return i, out, cos


# --- fluency -------------------------------------------------------------------
def _prompt_of(key, names, n_p):
    """Text key -> list of the probes its answers respond to."""
    i = int(key.split("|")[1] if key.startswith("neutral") else key.split("|")[2])
    return N.probes_for(names[i], n_p)


def _nll(model_id, pairs, enable_thinking=None, batch=12):
    """Per-answer mean NLL and share of tokens above SURPRISE, under model_id
    without any steering. pairs = [(prompt, answer)]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, spike = [], []
    t0 = time.time()
    for b in range(0, len(pairs), batch):
        ids, starts = [], []
        for p, a in pairs[b:b + batch]:
            pre = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, **kw)
            pi = tok(pre, add_special_tokens=False)["input_ids"]
            ai = tok(a, add_special_tokens=False)["input_ids"] or [pad]
            ids.append(pi + ai)
            starts.append(len(pi))
        L = max(map(len, ids))
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for k, s in enumerate(ids):
            x[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        x, m = x.to(model.device), m.to(model.device)
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits[:, :-1].float(), -1)
        tl = -lp.gather(-1, x[:, 1:, None])[..., 0]
        for k, s in enumerate(ids):
            t = tl[k, starts[k] - 1:len(s) - 1].cpu().numpy()
            nll.append(float(t.mean()))
            spike.append(float((t > SURPRISE).mean()))
        if b % (batch * 50) == 0:
            print(f"  {model_id.split('/')[-1]} {b}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return nll, spike


def _distinct2(text):
    w = text.lower().split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg) if bg else 1.0


def fluency(texts, names, n_p):
    keys = sorted(texts)
    pairs, owner = [], []
    for key in keys:
        for k, (p, a) in enumerate(zip(_prompt_of(key, names, n_p), texts[key])):
            pairs.append((p, a))
            owner.append((key, k))
    ll, ls = _nll(S.BACKENDS["llama"]["model"], pairs)
    ql, _ = _nll(N.JUDGE, pairs, enable_thinking=False)
    out = {}
    for (key, k), a, b, c, (_, txt) in zip(owner, ll, ls, ql, pairs):
        out.setdefault(key, {"llama": [], "spike": [], "qwen": [], "d2": []})
        out[key]["llama"].append(a)
        out[key]["spike"].append(b)
        out[key]["qwen"].append(c)
        out[key]["d2"].append(_distinct2(txt))
    return out


# --- report --------------------------------------------------------------------
def report(fu, ev, names):
    en = json.loads((config.ROOT / "dimension_names_en.json").read_text(encoding="utf-8"))
    alphas, axes, n_p = ev["alphas"], ev["axes"], ev["n_probes"]
    rands = [s for s in ev["sources"] if s.startswith("rand")]
    fl = fu["fluency"]
    sc = ev["scores"]

    def fm(keys, m):
        return float(np.mean([v for k in keys for v in fl[k][m]]))

    neutral = [f"neutral|{i}" for i in axes]
    base = {m: fm(neutral, m) for m in ("llama", "spike", "qwen", "d2")}
    md = ["", "## Fluency cost", "",
          "Every answer scored by the **unsteered** Llama and by Qwen3-8B: mean NLL per answer "
          f"token (lower = more fluent), share of tokens Llama finds very surprising (NLL > "
          f"{SURPRISE:.0f} nats, how a garbled word shows up) and distinct-2 (1 = no repeated word "
          f"pairs). `neutral` = the same probes with no push. Generated by "
          "`python -m steering.native_followup`.", "",
          "| α | source | score | Llama NLL | Qwen NLL | surprising tokens | distinct-2 |",
          "|---|---|---|---|---|---|---|",
          f"| 0 | neutral | — | {base['llama']:.2f} | {base['qwen']:.2f} | "
          f"{100 * base['spike']:.1f} % | {base['d2']:.2f} |"]
    for a in alphas:
        for s in list(N.BASE_SOURCES) + ["rand"]:
            srcs = rands if s == "rand" else [s]
            keys = [f"{a}|{x}|{i}|{p}" for x in srcs for i in axes for p in ("up", "down")]
            score = np.mean([sc[f"{a}|{x}|{i}"] for x in srcs for i in axes])
            md.append(f"| {a} | {s} | {score:.2f} | {fm(keys, 'llama'):.2f} | {fm(keys, 'qwen'):.2f} | "
                      f"{100 * fm(keys, 'spike'):.1f} % | {fm(keys, 'd2'):.2f} |")
    md += ["", "Per axis, native vector: Llama NLL above neutral (mean of up and down) and judge "
           "score, by α.", "",
           "| axis | " + " | ".join(f"ΔNLL α{a}" for a in alphas) + " | " +
           " | ".join(f"score α{a}" for a in alphas) + " |",
           "|---|" + "---|" * (2 * len(alphas))]
    for i in axes:
        b = fm([f"neutral|{i}"], "llama")
        d = [fm([f"{a}|native|{i}|{p}" for p in ("up", "down")], "llama") - b for a in alphas]
        s = [float(np.mean(sc[f"{a}|native|{i}"])) for a in alphas]
        md.append(f"| {en.get(names[i], names[i])} | " + " | ".join(f"{x:+.2f}" for x in d) +
                  " | " + " | ".join(f"{x:.2f}" for x in s) + " |")

    t = fu["temperature"]
    i = t["axis"]
    md += ["", "## Temperature, second try", "",
           "The first native feature for temperature (#3809, \"references to events, suggestions, "
           "and comparisons\") pushed the wrong way. The table is to blame: d004's scale runs from "
           "absolute zero to the Planck temperature, so fire, sun, lava and ovens all score 0.0 "
           "(92 % of concepts sit at 0), abstract virtues get 0.86 and LLM preamble rows (\"here "
           "is the list of 10 terms…\") 1.0. The column tracks string length (r +0.53), and #3809 "
           "tracked that. The tactile axis d043 (liquid nitrogen → red-hot metal) is sane on "
           "physical things (fire 0.83, lava 0.93, ice 0.0), and its unique part is tracked by "
           f"real heat features. Candidates: `fire` = SAE #{F_FIRE} \"references to fire and "
           f"flames\"; `fire_cold` = #{F_FIRE} minus #{F_COLD} \"cold temperatures and frozen "
           "states\"; `probe43` = ridge on d043's unique part. Judged on d004's anchors as before; "
           f"native and rand reused from the replication. {len(alphas) * n_p} trials per source.", "",
           "| source | " + " | ".join(f"α {a}" for a in alphas) + " | all | vs rand p | Llama ΔNLL |",
           "|---|" + "---|" * (len(alphas) + 3)]
    tsc, tfl = t["scores"], t["fluency"]
    b = fm([f"neutral|{i}"], "llama")
    rnd = np.concatenate([np.mean([sc[f"{a}|{r}|{i}"] for r in rands], 0) for a in alphas])
    for s in ["native"] + list(TEMP_SOURCES) + ["rand"]:
        if s == "rand":
            per = [np.mean([sc[f"{a}|{r}|{i}"] for r in rands], 0) for a in alphas]
            keys = [f"{a}|{r}|{i}|{p}" for a in alphas for r in rands for p in ("up", "down")]
            src_fl = fl
        elif s == "native":
            per = [np.asarray(sc[f"{a}|native|{i}"]) for a in alphas]
            keys = [f"{a}|native|{i}|{p}" for a in alphas for p in ("up", "down")]
            src_fl = fl
        else:
            per = [np.asarray(tsc[f"{a}|{s}|{i}"]) for a in alphas]
            keys = [f"{a}|{s}|{i}|{p}" for a in alphas for p in ("up", "down")]
            src_fl = tfl
        allv = np.concatenate(per)
        p = "—" if s == "rand" else f"{N._sign_p(allv - rnd):.4f}"
        dn = np.mean([v for k in keys for v in src_fl[k]["llama"]]) - b
        md.append(f"| {s} | " + " | ".join(f"{x.mean():.2f}" for x in per) +
                  f" | **{allv.mean():.2f}** | {p} | {dn:+.2f} |")
    best = max(TEMP_SOURCES, key=lambda s: np.mean([tsc[f"{a}|{s}|{i}"] for a in alphas]))
    a_mid = alphas[len(alphas) // 2]
    md += ["", f"Best candidate `{best}`, α={a_mid}, first two probes:", ""]
    for k in range(2):
        md += ["> ⬆ " + t["texts"][f"{a_mid}|{best}|{i}|up"][k].replace("\n", " ")[:240], ">",
               "> ⬇ " + t["texts"][f"{a_mid}|{best}|{i}|down"][k].replace("\n", " ")[:240], ""]
    md += ["Cosines between the candidates and the old native direction: " +
           "; ".join(f"{s}·native {t['cos'][s]['native']:+.2f}" for s in TEMP_SOURCES) +
           f"; fire·probe43 {t['cos']['fire']['probe43']:+.2f}.", ""]

    txt = N.OUT_MD.read_text(encoding="utf-8")
    cut = txt.find("\n## Fluency cost")
    if cut >= 0:
        txt = txt[:cut]
    N.OUT_MD.write_text(txt.rstrip("\n") + "\n" + "\n".join(md), encoding="utf-8")
    print(f"📊 {N.OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="sae_concepts --cache npz (pooled activations)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    ev = json.loads(N.EVAL_JSON.read_text(encoding="utf-8"))
    if args.report_only:
        report(json.loads(OUT_JSON.read_text(encoding="utf-8")), ev, names)
        return
    alphas, n_p = ev["alphas"], ev["n_probes"]
    print("⏳ temperature vectors")
    i, tv, cos = temperature_vectors(args.cache, names)
    print("⏳ generating temperature candidates")
    ttexts = N.generate([i], tv, names, alphas, n_p, 90)
    ttexts.pop(f"neutral|{i}")
    print("⏳ judging temperature candidates")
    tscores = N.judge_qwen(ttexts, [i], names, alphas, list(TEMP_SOURCES), n_p)
    print("⏳ fluency")
    fl = fluency(ev["texts"] | ttexts, names, n_p)
    tfl = {k: fl.pop(k) for k in ttexts}
    fu = {"fluency": fl, "temperature": {"axis": i, "cos": cos, "texts": ttexts,
                                          "scores": tscores, "fluency": tfl}}
    OUT_JSON.write_text(json.dumps(fu, ensure_ascii=False, indent=1), encoding="utf-8")
    report(fu, ev, names)


if __name__ == "__main__":
    main()
