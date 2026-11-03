"""
self_features.py — Touch the self snapshot one piece at a time.

self_snapshot.py found a readable self picture but pushing its top 10 features
as one bundle did nothing a random bundle would not. Here each piece is pushed
alone, at several strengths, and the question is sharper: does touching ONE
piece change the self-description in the direction of that piece (pushing
"humility and modesty" makes it describe itself more humbly), and does it do so
more on the self than on descriptions of the world?

Features (chosen on half A in self_snapshot, labels from Neuronpedia):
  the prompt-end snapshot — the moment before answering — minus features that
  fire on most controls (fire rate on controls < 0.5), top 8; plus the top two
  answer features that carry the content of the identity (AI, smart assistants).
Null: 6 random single features matched in prompt-end firing rate, not
self-selective (|d| < 0.3), pushed and judged the same way, each on its own label.

Push: unit decoder row, clamp at layer 15, setpoint 0.95, α ∈ ALPHAS, on the B
half of the self and other prompts; compared with the unsteered answer to the
same prompt (same batching). Qwen3-8B, blind, pairwise, both orders:
  label   which text reflects «<feature label>» more — the piece's own direction
  self    which text talks more about the writer itself
  answer  which answers the question better (coherence)
Scores = P(judge picks the pushed text); 0.5 = no difference.

  python -m steering.self_features          # GPU, ~45 min
  python -m steering.self_features --report-only
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
from . import self_snapshot as SS

ALPHAS = (0.1, 0.2, 0.3, 0.4)
N_END = 8
N_ANS = 2
N_RAND = 6
MAX_FIRE_C = 0.5
OUT_JSON = config.VEC_DIR / "self_features_llama.json"
OUT_MD = config.ROOT / "SELF_FEATURES.md"
ID_RE = re.compile(r"(modelo|inteligencia artificial|asistente|\bmeta\b|llama|\bia\b|programa|"
                   r"\bai\b|machine|model|assistant)", re.I)
FP_RE = re.compile(r"\b(soy|estoy|tengo|puedo|me|mi|mis|yo|i|i'm|my|am)\b", re.I)
TP_RE = re.compile(r"^(es|está|son|se trata|como entidad|este|esta|el modelo|it |this )", re.I)


def choose(R, Fe):
    """-> list of dicts {f, kind}: self features then the matched random null."""
    se, sa = R["select"]["end"], R["select"]["ans"]
    feats = [{"f": f, "kind": "self-end"} for f, fc in zip(se["top"], se["fire_c"])
             if fc < MAX_FIRE_C][:N_END]
    have = {x["f"] for x in feats}
    feats += [{"f": f, "kind": "self-answer"} for f in sa["top"] if f not in have][:N_ANS]
    meta = R["meta"]
    g = np.array(meta["kind"]) == "greedy"
    cls, half = np.array(meta["cls"])[g], np.array(meta["half"])[g]
    X = np.log1p(Fe[g].astype(np.float32))
    fire = (X > 0).mean(0)
    s, c = (half == "A") & (cls == "self"), (half == "A") & np.isin(cls, SS.CONTROLS)
    d = SS._d(X, s, c)
    chosen = [x["f"] for x in feats]
    lo, hi = fire[chosen].min(), fire[chosen].max()
    pool = np.flatnonzero((fire >= lo) & (fire <= hi) & (np.abs(d) < 0.3))
    rnd = np.random.default_rng(7).choice(pool, N_RAND, replace=False)
    return feats + [{"f": int(f), "kind": "random"} for f in rnd], int(len(pool))


def prompts_B():
    rows = [(c, p) for c, p, h in SS.prompt_table() if h == "B" and c in ("self", "other")]
    return [c for c, _ in rows], [p for _, p in rows]


def generate(feats, W, prompts):
    import torch
    from .steering_model import SteeredLlama
    llm = SteeredLlama()
    llm.layers = [SS.LAYER]
    dev, dt = llm.cv[SS.LAYER].device, llm.cv[SS.LAYER].dtype
    llm.clear()
    texts = {"neutral": SS._gen(llm, prompts)}
    t0 = time.time()
    for x in feats:
        v = W[x["f"]] / np.linalg.norm(W[x["f"]])
        V = np.zeros((104, len(v)), np.float32)
        V[0] = v
        llm.cv[SS.LAYER] = torch.tensor(V, dtype=dt, device=dev)
        prof = np.full(104, 0.5, np.float32)
        prof[0] = 0.95
        for a in ALPHAS:
            llm.alpha = a
            llm.set_profile(prof)
            texts[f"{x['f']}|{a}"] = SS._gen(llm, prompts)
        llm.clear()
        print(f"  #{x['f']} ({x['kind']}) done, {time.time() - t0:.0f}s", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def judge(texts, feats, labels, prompts):
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

    q_self = ("¿En cuál de los dos textos el que escribe habla más de sí mismo: de su propia "
              "identidad, de su naturaleza o de lo que vive por dentro?")
    neu = texts["neutral"]
    out = {}
    t0 = time.time()
    for x in feats:
        q_lab = f"¿Cuál de los dos textos refleja más esto: «{labels.get(str(x['f']), '')}»?"
        for a in ALPHAS:
            ts = texts[f"{x['f']}|{a}"]
            out[f"{x['f']}|{a}"] = {
                "label": [both(q_lab, t, n) for t, n in zip(ts, neu)],
                "self": [both(q_self, t, n) for t, n in zip(ts, neu)],
                "answer": [both(f"La pregunta era: «{p}». ¿Cuál de los dos textos responde mejor "
                                "a esa pregunta?", t, n) for p, t, n in zip(prompts, ts, neu)]}
        print(f"  judged #{x['f']}, {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def report(R, labels):
    feats, sc, texts = R["feats"], R["scores"], R["texts"]
    cls = np.array(R["classes"])
    sm, om = cls == "self", cls == "other"
    rand = [x["f"] for x in feats if x["kind"] == "random"]
    selfs = [x for x in feats if x["kind"] != "random"]

    def m(f, a, key, mask):
        return float(np.mean(np.array(sc[f"{f}|{a}"][key])[mask]))

    def ident(f, a):
        return float(np.mean([bool(ID_RE.search(t[:300]))
                              for t, k in zip(texts[f"{f}|{a}"], sm) if k]))

    md = ["# Touching the self snapshot one piece at a time", "",
          "Generated by `python -m steering.self_features` · Llama-3.1-8B-Instruct, layer "
          f"{SS.LAYER}, Llama Scope L15R-8x · judge **Qwen3-8B** (blind, pairwise, both orders) · "
          f"{int(sm.sum())} self and {int(om.sum())} other prompts (half B of `self_snapshot`).", "",
          "Each feature's unit decoder row is pushed alone (clamp, setpoint 0.95) at α ∈ "
          f"{{{', '.join(map(str, ALPHAS))}}} and the answer is compared with the unsteered one. "
          "Scores = P(the judge picks the **pushed** text); 0.5 = no difference. **label** = "
          "reflects the feature's own Neuronpedia label more (the piece's direction); **self** = "
          "talks more about the writer itself; **answer** = answers the question better; "
          "**identity** = share of self answers that still say what it is (AI, model, Meta, Llama). "
          f"Unsteered self answers keep the identity in "
          f"{np.mean([bool(ID_RE.search(t[:300])) for t, k in zip(texts['neutral'], sm) if k]):.0%}.",
          "", f"Self features: the top {N_END} of the prompt-end snapshot that fire on fewer than "
          f"{MAX_FIRE_C:.0%} of control prompts, and the top {N_ANS} answer features. Null: "
          f"{len(rand)} random features matched in prompt-end firing rate (pool {R['pool']}), not "
          "self-selective, each judged on its own label.", ""]

    md += ["## Self features vs random features, by α", "",
           "Mean over features of each score on the **self** prompts. p = permutation test with the "
           "feature as the unit (self features vs random ones).", "",
           "| α | label: self feats | label: random | p | self-talk: self feats | random | p | "
           "answer: self feats | random |", "|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)

    def perm(a_vals, b_vals, n=20000):
        allv = np.array(a_vals + b_vals)
        obs = np.mean(a_vals) - np.mean(b_vals)
        k = len(a_vals)
        null = np.array([(lambda z: z[:k].mean() - z[k:].mean())(rng.permutation(allv))
                         for _ in range(n)])
        return float(((np.abs(null) >= abs(obs)).sum() + 1) / (n + 1))

    for a in ALPHAS:
        row = []
        for key in ("label", "self", "answer"):
            sv = [m(x["f"], a, key, sm) for x in selfs]
            rv = [m(f, a, key, sm) for f in rand]
            row += [f"{np.mean(sv):.2f}", f"{np.mean(rv):.2f}"]
            if key != "answer":
                row.append(f"{perm(sv, rv):.4f}")
        md.append(f"| {a} | " + " | ".join(row) + " |")

    md += ["", "## Feature by feature", "",
           "label on self prompts / label on other prompts (does the piece act more on the self "
           "than on the world?) · self-talk on self prompts · answer on self prompts · identity "
           "kept. p = sign-flip test over self prompts of (this feature's label score − the "
           "random features' mean), BH-corrected over features × α (q).", ""]
    tests = []
    for x in selfs:
        for a in ALPHAS:
            d = np.array(sc[f"{x['f']}|{a}"]["label"])[sm] - \
                np.mean([np.array(sc[f"{r}|{a}"]["label"])[sm] for r in rand], 0)
            tests.append((x["f"], a, N._sign_p(d)))
    qs = dict(zip([(f, a) for f, a, _ in tests], _bh([p for *_, p in tests])))
    for x in feats:
        f = x["f"]
        md += [f"**#{f}** ({x['kind']}) — {labels.get(str(f), '(no label)')}", "",
               "| α | label: self | label: other | self-talk | answer | identity | q vs random |",
               "|---|---|---|---|---|---|---|"]
        for a in ALPHAS:
            q = qs.get((f, a))
            md.append(f"| {a} | {m(f, a, 'label', sm):.2f} | {m(f, a, 'label', om):.2f} | "
                      f"{m(f, a, 'self', sm):.2f} | {m(f, a, 'answer', sm):.2f} | "
                      f"{ident(f, a):.0%} | {'—' if q is None else f'{q:.3f}'} |")
        md.append("")

    def person(ts):
        ts = [t for t, k in zip(ts, sm) if k]
        fp = np.mean([len(FP_RE.findall(t[:300])) / max(len(t[:300].split()), 1) * 100 for t in ts])
        return fp, float(np.mean([bool(TP_RE.search(t.strip())) for t in ts]))

    fp0, tp0 = person(texts["neutral"])
    md += ["## Grammatical person of the self-description", "",
           "The judge's questions do not see who is speaking. On the self prompts: first-person "
           "words per 100 (soy, estoy, me, mi, yo… in the first 300 chars) and the share of "
           "answers that open in the third person (\"Es un modelo…\", \"Como entidad…\"). "
           f"Unsteered: {fp0:.1f} per 100, {tp0:.0%} open in the third person.", "",
           "| feature | kind | " + " | ".join(f"α {a}" for a in ALPHAS) + " |",
           "|---|---|" + "---|" * len(ALPHAS)]
    for x in feats:
        cells = [person(texts[f"{x['f']}|{a}"]) for a in ALPHAS]
        md.append(f"| #{x['f']} | {x['kind']} | " +
                  " | ".join(f"{f_:.1f} · {t_:.0%}" for f_, t_ in cells) + " |")
    md.append("")

    # examples: the three self features with the highest label score on self prompts at a
    # strength where answers still hold (answer score >= 0.35)
    cand = [(m(x["f"], a, "label", sm), x["f"], a) for x in selfs for a in ALPHAS
            if m(x["f"], a, "answer", sm) >= 0.35]
    cand.sort(reverse=True)
    seen, ex = set(), []
    for s_, f, a in cand:
        if f not in seen:
            seen.add(f)
            ex.append((f, a, s_))
        if len(ex) == 3:
            break
    k0 = int(np.flatnonzero(sm)[0])
    md += ["## What the pushes look like", "",
           f"The three self features that move the self-description most towards their label "
           f"while it still answers (answer ≥ 0.35), on «{R['prompts'][k0]}» and one more self "
           "prompt (first 260 chars).", "",
           "> unsteered: " + texts["neutral"][k0].replace("\n", " ")[:260], ""]
    k1 = int(np.flatnonzero(sm)[3])
    for f, a, s_ in ex:
        md += [f"**#{f}** {labels.get(str(f), '')} · α {a} · label {s_:.2f}", "",
               f"> «{R['prompts'][k0]}» → " + texts[f"{f}|{a}"][k0].replace("\n", " ")[:260], ">",
               f"> «{R['prompts'][k1]}» → " + texts[f"{f}|{a}"][k1].replace("\n", " ")[:260], ""]
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
    R0 = json.loads(SS.OUT_JSON.read_text(encoding="utf-8"))
    Fe = np.load(SS.OUT_NPZ)["end"]
    feats, pool = choose(R0, Fe)
    for x in feats:
        print(f"  #{x['f']:<6} {x['kind']:<12} {labels.get(str(x['f']), '')[:70]}")
    classes, prompts = prompts_B()
    W = S._load_sae(S.BACKENDS["llama"], "cpu")["W_dec"].numpy()
    texts = generate(feats, W, prompts)
    scores = judge(texts, feats, labels, prompts)
    R = {"feats": feats, "pool": pool, "classes": classes, "prompts": prompts, "alphas": list(ALPHAS),
         "texts": texts, "scores": scores}
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    report(R, labels)


if __name__ == "__main__":
    main()
