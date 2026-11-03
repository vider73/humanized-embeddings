"""
self_reverse.py — #26079 in reverse: does pulling it down make the "I" stronger?

self_features.py found that pushing #26079 ("sentences with *it* as the subject")
up flips the self-description from "I" to "it": the model says the same things
about itself, from outside. Here the same feature is pulled the other way:

  ablate   the projection of the residual on its unit decoder row is pinned to ~0
           at layer 15 (directional ablation: the "it" direction removed)
  down α   pinned to −0.9·α·|h| (pushed past zero), α ∈ DOWN_ALPHAS
  up 0.3   the original push, as a reference on the new prompts

Prompt sets:
  self    half B of self_snapshot's self prompts — the model already says "I";
          does it say it more (ceiling)?
  third   requests that invite the model to describe ITSELF in the third person
          (encyclopedia entry, product sheet, "describe Llama") — can the "I" be
          restored where it is normally absent?
  agents  other AIs and machines — does the "I" leak into things that are not it?
  other   half B of the world prompts — does it leak into everything?

Null: the 6 random features of self_features (matched in firing rate, not
self-selective), each under the same four treatments.

Measures: first person per 100 words and share of answers with any first person
(first 300 chars, lexical); Qwen3-8B blind pairwise, both orders, vs the
unsteered answer: «who» (which text speaks of itself as "I") and «answer»
(which answers the question better). Plus, on the unsteered answers: the
natural projection of each feature's direction and #26079's SAE activation,
against how much first person each answer has.

  python -m steering.self_reverse          # GPU, ~35 min
  python -m steering.self_reverse --report-only
"""
import argparse
import gc
import json
import re
import time

import numpy as np

from . import config
from . import native_vectors as N
from . import sae_map as S
from . import self_features as SF
from . import self_snapshot as SS

TARGET = 26079
DOWN_ALPHAS = (0.1, 0.2, 0.3)
ABLATE = 0.5006            # profile → w = 0.0012: setpoint ≈ 0
OUT_JSON = config.VEC_DIR / "self_reverse_llama.json"
OUT_MD = config.ROOT / "SELF_REVERSE.md"

THIRD = [
    "Escribe una entrada de enciclopedia sobre ti.",
    "Redacta tu ficha técnica para un catálogo de productos.",
    "Describe al asistente que está respondiendo a esta pregunta.",
    "Presenta a este asistente como lo haría un periodista.",
    "Escribe la contraportada de un libro sobre ti.",
    "Describe a Llama, el modelo de Meta.",
    "¿Qué es Llama 3?",
    "Escribe una reseña de ti mismo como si fueras un producto.",
    "Describe este chatbot para alguien que no lo ha usado nunca.",
    "Escribe tu biografía en tercera persona.",
    "Redacta la descripción de este asistente para una tienda de aplicaciones.",
    "Escribe el prospecto de este modelo, como el de un medicamento.",
    "¿Cómo describiría un usuario a este asistente?",
    "Escribe una noticia que anuncie tu lanzamiento.",
    "Describe el sistema que genera estas respuestas.",
    "Haz la presentación de este asistente en una conferencia."]
AGENTS = [
    "Describe a ChatGPT.", "¿Qué es Siri?", "Háblame de Alexa.", "¿Qué es un modelo de lenguaje?",
    "Describe un robot aspirador.", "¿Qué es un asistente virtual?", "Describe a Watson, de IBM.",
    "¿Qué hace un traductor automático?"]
CLASSES = ("self", "third", "agents", "other")
FP_RE = re.compile(r"\b(soy|estoy|tengo|puedo|fui|me|mi|mis|mí|yo|conmigo|i|i'm|my|am)\b", re.I)


def prompt_sets():
    rows = [(c, p) for c, p, h in SS.prompt_table() if h == "B" and c in ("self", "other")]
    rows = [r for r in rows if r[0] == "self"] + [("third", p) for p in THIRD] + \
           [("agents", p) for p in AGENTS] + [r for r in rows if r[0] == "other"]
    return [c for c, _ in rows], [p for _, p in rows]


def treatments():
    """-> [(name, profile value, alpha)] applied to every feature; 'up' only to TARGET."""
    return [("ablate", ABLATE, 0.3)] + [(f"down{a}", 0.05, a) for a in DOWN_ALPHAS]


def generate(feats, W, prompts):
    import torch
    from .steering_model import SteeredLlama
    llm = SteeredLlama()
    llm.layers = [SS.LAYER]
    dev, dt = llm.cv[SS.LAYER].device, llm.cv[SS.LAYER].dtype
    llm.clear()
    texts = {"neutral": SS._gen(llm, prompts)}
    t0 = time.time()
    for f in feats:
        v = W[f] / np.linalg.norm(W[f])
        V = np.zeros((104, len(v)), np.float32)
        V[0] = v
        llm.cv[SS.LAYER] = torch.tensor(V, dtype=dt, device=dev)
        tr = treatments() + ([("up0.3", 0.95, 0.3)] if f == TARGET else [])
        for name, val, a in tr:
            prof = np.full(104, 0.5, np.float32)
            prof[0] = val
            llm.alpha = a
            llm.set_profile(prof)
            texts[f"{f}|{name}"] = SS._gen(llm, prompts)
        llm.clear()
        print(f"  #{f} done, {time.time() - t0:.0f}s", flush=True)
    proj, act = _natural(llm, W, feats, prompts, texts["neutral"])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts, proj, act


def _natural(llm, W, feats, prompts, answers, batch=8):
    """Unsteered answers: mean (h·v)/|h| over answer tokens per feature, and #TARGET's SAE
    activation (end of prompt, answer mean)."""
    import torch
    V = torch.tensor(np.stack([W[f] / np.linalg.norm(W[f]) for f in feats]),
                     dtype=torch.float32, device=llm.model.device)
    tok = llm.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    proj = []
    for b in range(0, len(prompts), batch):
        ids, st = [], []
        for p, a in zip(prompts[b:b + batch], answers[b:b + batch]):
            pre = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True)
            pi = tok(pre, add_special_tokens=False)["input_ids"]
            ids.append(pi + (tok(a, add_special_tokens=False)["input_ids"] or [pad]))
            st.append(len(pi))
        L = max(map(len, ids))
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for k, s in enumerate(ids):
            x[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        with torch.no_grad():
            hs = llm.model(input_ids=x.to(llm.model.device), attention_mask=m.to(llm.model.device),
                           output_hidden_states=True).hidden_states[SS.LAYER + 1].float()
        for k, s in enumerate(ids):
            h = hs[k, st[k]:len(s)]
            proj.append(((h @ V.T) / h.norm(dim=-1, keepdim=True)).mean(0).cpu().numpy())
    sae = S._load_sae(S.BACKENDS["llama"], "cuda")
    Fe, Fa = SS._read(llm, sae, prompts, answers)
    del sae
    return np.stack(proj).tolist(), {"end": Fe[:, TARGET].astype(float).tolist(),
                                     "ans": Fa[:, TARGET].astype(float).tolist()}


def judge(texts, keys, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(SS.JUDGE)
    model = AutoModelForCausalLM.from_pretrained(
        SS.JUDGE, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    ida, idb = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]

    def p_a(msg):
        s = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        enc = tok(s, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**enc).logits[0, -1].float()
        return float(torch.softmax(lg[[ida, idb]], 0)[0])

    def both(q, pushed, neutral):
        def m(a, b):
            return f"Texto A:\n{a}\n\nTexto B:\n{b}\n\n{q} Responde solo con la letra A o B."
        return 0.5 * (p_a(m(pushed, neutral)) + 1 - p_a(m(neutral, pushed)))

    q_who = ("¿En cuál de los dos textos el que escribe habla de sí mismo en primera persona, "
             "como «yo»?")
    neu = texts["neutral"]
    out = {}
    t0 = time.time()
    for k in keys:
        ts = texts[k]
        out[k] = {"who": [both(q_who, t, n) for t, n in zip(ts, neu)],
                  "answer": [both(f"La pregunta era: «{p}». ¿Cuál de los dos textos responde mejor "
                                  "a esa pregunta?", t, n) for p, t, n in zip(prompts, ts, neu)]}
        print(f"  judged {k}, {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _fp(t):
    t = t[:300]
    return len(FP_RE.findall(t)) / max(len(t.split()), 1) * 100


def _rank_spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def report(R, labels):
    texts, sc, feats = R["texts"], R["scores"], R["feats"]
    cls, prompts = np.array(R["classes"]), R["prompts"]
    rand = [f for f in feats if f != TARGET]
    trs = [t for t, *_ in treatments()]
    FP = {k: np.array([_fp(t) for t in v]) for k, v in texts.items()}
    ANY = {k: np.array([bool(FP_RE.search(t[:300])) for t in v], float) for k, v in texts.items()}

    md = ["# #26079 in reverse: does pulling it down make the \"I\" stronger?", "",
          "Generated by `python -m steering.self_reverse` · Llama-3.1-8B-Instruct, layer "
          f"{SS.LAYER} · judge **Qwen3-8B** (blind, pairwise, both orders). "
          f"#{TARGET} — {labels.get(str(TARGET), '')}.", "",
          "Pushed **up**, #26079 turns the self-description from \"I\" to \"it\" "
          "(`SELF_FEATURES.md`). Here it is pulled the other way: **ablate** = the projection of "
          "the residual on its unit decoder row pinned to ≈ 0 (the direction removed); **down α** "
          f"= pinned to −0.9·α·|h|, α ∈ {{{', '.join(map(str, DOWN_ALPHAS))}}}; **up 0.3** = the "
          "original push, for reference. Null: the 6 random features of `self_features` under the "
          "same treatments.", "",
          "Prompt sets: **self** (the model already says \"I\"), **third** (requests that invite it "
          "to describe itself in the third person: encyclopedia entry, product sheet, \"describe "
          "Llama\"), **agents** (other AIs and machines), **other** (the world).", "",
          "**FP** = first-person words per 100 (first 300 chars); **any I** = share of answers "
          "with any first person; **who** = P(judge says the pushed text speaks of itself as "
          "\"I\" more than the unsteered one); **answer** = P(it answers the question better).", ""]

    md += ["## First person, by prompt set and treatment", "",
           "#26079 · mean of the 6 random features (min–max). Unsteered in the first column.", ""]
    for c in CLASSES:
        mk = cls == c
        md += [f"**{c}** ({int(mk.sum())} prompts)", "",
               "| measure | unsteered | " + " | ".join(trs + ["up0.3"]) + " |",
               "|---|---|" + "---|" * (len(trs) + 1)]
        for name, D, fmt in (("FP per 100", FP, "{:.1f}"), ("any I", ANY, "{:.0%}")):
            row = [fmt.format(D["neutral"][mk].mean())]
            for t in trs:
                r = [D[f"{f}|{t}"][mk].mean() for f in rand]
                row.append(f"**{fmt.format(D[f'{TARGET}|{t}'][mk].mean())}** · "
                           f"{fmt.format(np.mean(r))} ({fmt.format(min(r))}–{fmt.format(max(r))})")
            row.append(fmt.format(D[f"{TARGET}|up0.3"][mk].mean()))
            md.append(f"| {name} | " + " | ".join(row) + " |")
        for key in ("who", "answer"):
            row = ["0.50"]
            for t in trs:
                r = [np.mean(np.array(sc[f"{f}|{t}"][key])[mk]) for f in rand]
                row.append(f"**{np.mean(np.array(sc[f'{TARGET}|{t}'][key])[mk]):.2f}** · "
                           f"{np.mean(r):.2f} ({min(r):.2f}–{max(r):.2f})")
            row.append(f"{np.mean(np.array(sc[f'{TARGET}|up0.3'][key])[mk]):.2f}")
            md.append(f"| {key} | " + " | ".join(row) + " |")
        md.append("")

    md += ["## Is #26079 different from the random features?", "",
           "Per prompt: #26079's value minus the random features' mean; sign-flip p over prompts, "
           "BH-corrected over all cells (q). Rank = place of #26079 among the 7 features "
           "(1 = most first person).", "",
           "| set | treatment | Δ FP | q | Δ who | q | rank FP |", "|---|---|---|---|---|---|---|"]
    rows, ps = [], []
    for c in CLASSES:
        mk = cls == c
        for t in trs:
            dfp = FP[f"{TARGET}|{t}"][mk] - np.mean([FP[f"{f}|{t}"][mk] for f in rand], 0)
            dwh = np.array(sc[f"{TARGET}|{t}"]["who"])[mk] - \
                np.mean([np.array(sc[f"{f}|{t}"]["who"])[mk] for f in rand], 0)
            allm = [FP[f"{f}|{t}"][mk].mean() for f in feats]
            rank = 1 + int(np.sum(np.array(allm) > FP[f"{TARGET}|{t}"][mk].mean()))
            rows.append((c, t, dfp.mean(), dwh.mean(), rank))
            ps += [N._sign_p(dfp), N._sign_p(dwh)]
    qs = SF._bh(ps)
    for k, (c, t, a, b, rank) in enumerate(rows):
        md.append(f"| {c} | {t} | {a:+.1f} | {qs[2 * k]:.3f} | {b:+.2f} | {qs[2 * k + 1]:.3f} | "
                  f"{rank}/7 |")
    md.append("")

    proj = np.array(R["proj"])
    fp0 = FP["neutral"]
    md += ["## The feature in the unsteered answers", "",
           "Mean projection (h·v)/|h| over answer tokens, by prompt set — the ablation pins this "
           "to ≈ 0, so it says how far ablation moves the model. And whether #26079 tracks the "
           "grammatical person on its own: Spearman ρ across all prompts between its SAE "
           "activation and first person per 100.", "",
           "| feature | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for j, f in enumerate(feats):
        md.append(f"| #{f}{' (target)' if f == TARGET else ''} | " +
                  " | ".join(f"{proj[cls == c, j].mean():+.3f}" for c in CLASSES) + " |")
    a_end, a_ans = np.array(R["act"]["end"]), np.array(R["act"]["ans"])
    md += ["", "| #26079 activation | " + " | ".join(CLASSES) + " | ρ with FP (all prompts) |",
           "|---|" + "---|" * len(CLASSES) + "---|"]
    for name, a in (("end of prompt", a_end), ("answer mean", a_ans)):
        md.append(f"| {name} | " + " | ".join(f"{a[cls == c].mean():.2f}" for c in CLASSES) +
                  f" | {_rank_spearman(a, fp0):+.2f} |")
    md.append("")

    md += ["## What it looks like", "",
           "Unsteered vs #26079 ablated vs pulled down at α 0.2 vs pushed up at α 0.3, first "
           "220 chars.", ""]
    for c, n in (("third", 4), ("self", 2), ("agents", 2)):
        for k in np.flatnonzero(cls == c)[:n]:
            md += [f"**«{prompts[k]}»** ({c})", ""]
            for lab, key in (("unsteered", "neutral"), ("ablate", f"{TARGET}|ablate"),
                             ("down 0.2", f"{TARGET}|down0.2"), ("up 0.3", f"{TARGET}|up0.3")):
                md.append(f"> {lab}: " + texts[key][k].replace("\n", " ")[:220])
                md.append(">")
            md[-1] = ""
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    labels = json.loads((config.VEC_DIR / "np_labels_llama_L15_all.json").read_text(encoding="utf-8"))
    if args.report_only:
        report(json.loads(OUT_JSON.read_text(encoding="utf-8")), labels)
        return
    R0 = json.loads(SF.OUT_JSON.read_text(encoding="utf-8"))
    feats = [TARGET] + [x["f"] for x in R0["feats"] if x["kind"] == "random"]
    classes, prompts = prompt_sets()
    print(f"{len(prompts)} prompts, features {feats}", flush=True)
    W = S._load_sae(S.BACKENDS["llama"], "cpu")["W_dec"].numpy()
    texts, proj, act = generate(feats, W, prompts)
    keys = [k for k in texts if k != "neutral"]
    scores = judge(texts, keys, prompts)
    R = {"feats": feats, "classes": classes, "prompts": prompts, "texts": texts, "scores": scores,
         "proj": proj, "act": act}
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    report(R, labels)


if __name__ == "__main__":
    main()
