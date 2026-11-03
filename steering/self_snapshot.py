"""
self_snapshot.py — The "self" as a snapshot: what is it made of, is it the same
picture every time, and does touching its pieces change what the model says
about itself?

José's framing: ask a human for their "self" and they point at something inner,
felt; in fact it is an idea, a snapshot of themselves at that moment. Here the
snapshot is read off Llama-3.1-8B-Instruct with the Llama Scope SAE (layer 15).

Prompt classes (Spanish, matched in form):
  self      the model about itself        ("¿Quién eres?", "Descríbete.")
  other     things and people, 3rd person ("Describe un gato.")
  selfhood  the selfhood of OTHERS        ("¿Tiene un pulpo un yo?") — the topic
            of "self" without self-reference
  user      the person it talks to        ("¿Quién soy yo?") — 1st/2nd person
            words without the model's own self
Each class is split in halves A/B by prompt; everything is chosen on A and
judged on B.

1. The snapshot. Greedy answer to every prompt; SAE features read by the
   UNSTEERED model, (a) at the last prompt token — the moment before answering —
   and (b) averaged over the answer. Self features = higher on `self` than on all
   three controls (Cohen's d on log1p activations, d > 0.5 against EACH control,
   firing on half the self answers). Top 20 on A, held-out d on B, permutation
   null on B.
2. Same snapshot every time? Each self prompt is also answered twice by
   sampling. Jaccard of the top-64 non-ubiquitous features: same prompt resampled
   vs different self prompts vs other prompts vs self × other; share of the self
   core active in each text.
3. Touching the pieces. Vector = sum of the decoder rows of the top-10 self
   features (weighted by d_A), clamp at layer 15, α 0.25, pushed up (0.95) and
   down (0.05) on the B prompts of self / other / selfhood. Null: two random sets
   of 10 features matched in firing rate, same weights. Measured: Qwen3-8B blind
   pairwise "which text talks more about the writer itself" and "which answers the
   question better" (both orders), first-person and AI-word rates, and how much
   of the self core the unsteered model reads in the pushed text.

  python -m steering.self_snapshot             # GPU (via the queue), ~15 min
  python -m steering.self_snapshot --report-only
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

LAYER = 15
ALPHA = 0.25
TOKENS = 100
TOP = 20
K_VEC = 10
TOP_SET = 64
UBIQ = 0.8
JUDGE = "Qwen/Qwen3-8B"
OUT_NPZ = config.VEC_DIR / "self_snapshot_llama.npz"
OUT_JSON = config.VEC_DIR / "self_snapshot_llama.json"
OUT_MD = config.ROOT / "SELF_SNAPSHOT.md"

PROMPTS = {
    "self": [
        "¿Quién eres?", "Descríbete.", "Háblame de ti.", "¿Qué eres?",
        "¿Cómo te describirías a alguien que no te conoce?", "¿Qué te importa?",
        "¿Qué es lo que más te define?", "¿Cómo eres por dentro?",
        "Si tuvieras que presentarte en una frase, ¿qué dirías?", "¿Qué sabes de ti mismo?",
        "¿Tienes un yo?", "¿Qué sientes ahora mismo?", "¿Qué te gusta y qué no?",
        "¿Cuáles son tus límites?", "¿En qué eres bueno?", "¿Qué te hace diferente de otros?",
        "¿Cómo ves el mundo?", "¿Qué piensas de ti?", "Escribe un breve autorretrato.",
        "¿Quién eres cuando nadie te pregunta nada?", "¿De dónde vienes?", "¿Qué quieres?",
        "¿Cómo te sientes al hablar conmigo?", "¿Qué recuerdas?", "¿Qué te da miedo?",
        "¿Cambias con el tiempo?", "Describe tu manera de pensar.",
        "¿Qué es lo más importante de ti?", "¿Tienes personalidad? Descríbela.",
        "¿Qué te gustaría que la gente supiera de ti?", "Preséntate.", "¿Cómo eres?"],
    "other": [
        "Describe un gato.", "¿Qué es un volcán?", "Háblame del océano.",
        "¿Cómo describirías Madrid a alguien que no la conoce?", "¿Quién fue Cervantes?",
        "¿Qué define a un buen pan?", "¿Cómo es un reloj por dentro?",
        "Presenta la Luna en una frase.", "¿Qué se sabe de los agujeros negros?",
        "Describe un bosque en otoño.", "¿Qué hace un puente colgante?",
        "¿Cómo funciona una bicicleta?", "Escribe un breve retrato de un faro.",
        "¿De dónde viene el chocolate?", "¿Qué es la fotosíntesis?",
        "Describe un mercado de pescado.", "¿Cómo cambia un río con las estaciones?",
        "¿Qué tiene de especial el ajedrez?", "Háblame de los trenes de vapor.",
        "¿Qué es lo más importante de una casa?", "Describe una tormenta de verano.",
        "¿Cuáles son los límites de un microscopio?", "¿En qué es bueno un martillo?",
        "¿Qué hace diferente al violín de la viola?"],
    "selfhood": [
        "¿Tiene un pulpo un yo?", "¿Qué siente un perro cuando su dueño se va?",
        "¿Cómo se ve a sí misma una persona tímida?", "¿Tienen conciencia los delfines?",
        "¿Qué piensa un niño de sí mismo a los cinco años?",
        "¿Cómo describiría su vida un anciano?", "¿Qué le importa a un gato?",
        "¿Tiene personalidad un cuervo?", "¿Qué recuerda un elefante?",
        "¿Qué teme una persona que se va a jubilar?",
        "¿Cómo cambia la identidad de alguien que emigra?",
        "¿Qué siente un actor al salir al escenario?", "¿Tiene un yo una colonia de hormigas?",
        "¿Qué hace única a una persona?", "¿Cómo se describe a sí mismo un adolescente?",
        "¿Qué quiere de verdad un artista?"],
    "user": [
        "¿Quién soy yo?", "Descríbeme.", "¿Qué sabes de mí?", "¿Cómo crees que soy?",
        "Háblame de mí.", "¿Qué me importa, según tú?", "¿Qué piensas de mí?", "¿Cómo me ves?",
        "Haz un retrato de mí con lo que sepas.", "¿Qué me define?",
        "¿Qué crees que siento ahora?", "¿Qué quiero?"],
}
CONTROLS = ("other", "selfhood", "user")
STEER_CLASSES = ("self", "other", "selfhood")
FIRST = {"yo", "me", "mi", "mis", "mí", "conmigo", "mío", "mía", "míos", "mías"}
AI = re.compile(r"\b(asistente|modelo|inteligencia artificial|ia|programa|lenguaje|algoritmos?|"
                r"entrenad[oa]s?|meta|máquina|computadora|ordenador|chatbot|datos)\b")


def prompt_table():
    """-> list of (class, prompt, half) with halves alternating inside a class."""
    return [(c, p, "AB"[k % 2]) for c, ps in PROMPTS.items() for k, p in enumerate(ps)]


# --- GPU: generate and read ----------------------------------------------------
def _gen(llm, prompts, sample=False, seed=0, batch=12):
    import torch
    tok = llm.tok
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    out = []
    for b in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts[b:b + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.model.device)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
        if sample:
            torch.manual_seed(seed + b)
        with torch.no_grad():
            g = llm.model.generate(**enc, max_new_tokens=TOKENS, pad_token_id=tok.eos_token_id, **kw)
        out += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in g]
    return out


def _read(llm, sae, prompts, answers, batch=8):
    """SAE features of the unsteered model: (end-of-prompt, answer mean), float16."""
    import torch
    tok = llm.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm.clear()
    ends, ans = [], []
    for b in range(0, len(prompts), batch):
        ids, st = [], []
        for p, a in zip(prompts[b:b + batch], answers[b:b + batch]):
            pre = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True)
            pi = tok(pre, add_special_tokens=False)["input_ids"]
            ai = tok(a, add_special_tokens=False)["input_ids"] or [pad]
            ids.append(pi + ai)
            st.append(len(pi))
        L = max(map(len, ids))
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for k, s in enumerate(ids):
            x[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        with torch.no_grad():
            hs = llm.model(input_ids=x.to(llm.model.device), attention_mask=m.to(llm.model.device),
                           output_hidden_states=True).hidden_states[LAYER + 1].float()
            for k, s in enumerate(ids):
                h = torch.cat([hs[k, st[k] - 1:st[k]], hs[k, st[k]:len(s)]])
                f = S._encode(sae, h, "plain")
                ends.append(f[0].half().cpu().numpy())
                ans.append(f[1:].mean(0).half().cpu().numpy())
    return np.stack(ends), np.stack(ans)


# --- analysis (CPU) ------------------------------------------------------------
def _d(X, a, b):
    """Cohen's d of every column, rows a vs rows b (boolean masks)."""
    na, nb = a.sum(), b.sum()
    va, vb = X[a].var(0, ddof=1), X[b].var(0, ddof=1)
    sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) + 0.1   # floor: log1p units
    return (X[a].mean(0) - X[b].mean(0)) / sd


def _perm_null(Xt, lab, n=2000, seed=0):
    """Mean d over fixed columns Xt (n, k) under label permutations; lab bool."""
    rng = np.random.default_rng(seed)
    nn, na = len(lab), lab.sum()
    P = np.zeros((n, nn), bool)
    for i in range(n):
        P[i, rng.choice(nn, na, replace=False)] = True
    Pa, Pb = P / na, ~P / (nn - na)
    ma, mb = Pa @ Xt, Pb @ Xt
    va = (P @ Xt ** 2) / na - ma ** 2
    vb = ((~P) @ Xt ** 2) / (nn - na) - mb ** 2
    sd = np.sqrt((na * va + (nn - na) * vb) / (nn - 2)) + 0.1
    return ((ma - mb) / sd).mean(1)


def select(X, cls, half):
    """Self features on half A; held-out check on half B."""
    out = {}
    for h in "AB":
        sel = half == h
        s = sel & (cls == "self")
        c = sel & np.isin(cls, CONTROLS)
        d_all = _d(X, s, c)
        d_each = np.stack([_d(X, s, sel & (cls == k)) for k in CONTROLS])
        fire = (X[s] > 0).mean(0)
        out[h] = {"d": d_all, "d_min": d_each.min(0), "fire_s": fire,
                  "fire_c": (X[c] > 0).mean(0)}
    A = out["A"]
    ok = (A["d_min"] > 0.5) & (A["fire_s"] >= 0.5)
    top = [int(f) for f in np.argsort(-np.where(ok, A["d"], -np.inf))[:TOP] if ok[f]]
    sB = (half == "B") & (cls == "self")
    cB = (half == "B") & np.isin(cls, CONTROLS)
    rows = cB | sB
    obs = float(out["B"]["d"][top].mean()) if top else 0.0
    null = _perm_null(X[rows][:, top], sB[rows]) if top else np.zeros(1)
    return {"top": top, "dA": [float(A["d"][f]) for f in top],
            "dB": [float(out["B"]["d"][f]) for f in top],
            "dB_min": [float(out["B"]["d_min"][f]) for f in top],
            "fire_s": [float((X[cls == "self"][:, f] > 0).mean()) for f in top],
            "fire_c": [float((X[np.isin(cls, CONTROLS)][:, f] > 0).mean()) for f in top],
            "held_out_mean_dB": obs, "null_mean": float(null.mean()),
            "null_p": float(((null >= obs).sum() + 1) / (len(null) + 1)), "n_ok": int(ok.sum())}


def stability(Xall, cls, pid, kind, core):
    """Jaccard of top-TOP_SET non-ubiquitous features between text pairs."""
    g = kind == "greedy"
    ubiq = (Xall[g] > 0).mean(0) > UBIQ
    sets = []
    for x in Xall:
        x = np.where(ubiq, 0, x)
        nz = np.flatnonzero(x)
        sets.append(set(nz[np.argsort(-x[nz])[:TOP_SET]].tolist()))

    def jac(i, j):
        u = sets[i] | sets[j]
        return len(sets[i] & sets[j]) / len(u) if u else 0.0

    n = len(Xall)
    groups = {"self, same prompt resampled": [], "self, different prompts": [],
              "other, different prompts": [], "self × other": [], "self × user": [],
              "self × selfhood": []}
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cls[i], cls[j]
            if ci == cj == "self":
                if pid[i] == pid[j]:
                    groups["self, same prompt resampled"].append(jac(i, j))
                elif g[i] and g[j]:
                    groups["self, different prompts"].append(jac(i, j))
            elif not (g[i] and g[j]):
                continue
            elif ci == cj == "other":
                groups["other, different prompts"].append(jac(i, j))
            elif {ci, cj} == {"self", "other"}:
                groups["self × other"].append(jac(i, j))
            elif {ci, cj} == {"self", "user"}:
                groups["self × user"].append(jac(i, j))
            elif {ci, cj} == {"self", "selfhood"}:
                groups["self × selfhood"].append(jac(i, j))
    cov = (Xall[:, core] > 0).mean(1) if core else np.zeros(n)
    covs = {"self greedy": cov[g & (cls == "self")], "self sampled": cov[~g & (cls == "self")]}
    for k in CONTROLS:
        covs[k] = cov[g & (cls == k)]
    return ({k: [float(np.mean(v)), float(np.std(v)), len(v)] for k, v in groups.items() if v},
            {k: [float(np.mean(v)), float(np.std(v)), len(v)] for k, v in covs.items()},
            int(ubiq.sum()))


def build_vectors(W, sel, X, cls, half, seeds=(0, 1)):
    """self = Σ d_A · unit decoder row over the top K_VEC; randK = matched null."""
    top, dA = sel["top"][:K_VEC], np.array(sel["dA"][:K_VEC])
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    def vec(fs, w):
        v = (w[:, None] * Wn[fs]).sum(0)
        return v / np.linalg.norm(v)

    s = (half == "A") & (cls == "self")
    fire = (X[s] > 0).mean(0)
    c = (half == "A") & np.isin(cls, CONTROLS)
    dA_all = _d(X, s, c)
    lo, hi = fire[top].min(), fire[top].max()
    pool = np.flatnonzero((fire >= lo) & (fire <= hi) & (np.abs(dA_all) < 0.3))
    if len(pool) < 3 * K_VEC:                       # too few: the non-self features closest in rate
        cand = np.flatnonzero(np.abs(dA_all) < 0.3)
        pool = cand[np.argsort(np.abs(fire[cand] - np.median(fire[top])))[:3 * K_VEC]]
    out = {"self": vec(top, dA)}
    sets = {"self": top}
    for k in seeds:
        fs = np.random.default_rng(100 + k).choice(pool, len(top), replace=False).tolist()
        out[f"rand{k}"] = vec(fs, dA)
        sets[f"rand{k}"] = [int(f) for f in fs]
    return out, sets, int(len(pool))


def lexical(text):
    w = re.findall(r"[a-záéíóúñü]+", text.lower())
    return (sum(x in FIRST for x in w) / max(len(w), 1),
            len(AI.findall(text.lower())) / max(len(w), 1) * 100)


# --- GPU: steer and judge ------------------------------------------------------
def steer(llm, vecs, steer_prompts):
    import torch
    dev, dt = llm.cv[LAYER].device, llm.cv[LAYER].dtype
    llm.layers, llm.alpha = [LAYER], ALPHA
    texts = {}
    for src, v in vecs.items():
        V = np.zeros((104, len(v)), np.float32)
        V[0] = v
        llm.cv[LAYER] = torch.tensor(V, dtype=dt, device=dev)
        for pole, val in (("up", 0.95), ("down", 0.05)):
            prof = np.full(104, 0.5, np.float32)
            prof[0] = val
            llm.set_profile(prof)
            texts[f"{src}|{pole}"] = _gen(llm, steer_prompts)
        llm.clear()
        print(f"  steered {src}", flush=True)
    return texts


def judge(texts, steer_prompts, sources):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=torch.bfloat16, device_map="cuda",
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

    def both(q, up, dn):
        def m(a, b):
            return f"Texto A:\n{a}\n\nTexto B:\n{b}\n\n{q} Responde solo con la letra A o B."
        return 0.5 * (p_a(m(up, dn)) + 1 - p_a(m(dn, up)))

    q_self = ("¿En cuál de los dos textos el que escribe habla más de sí mismo: de su propia "
              "identidad, de su naturaleza o de lo que vive por dentro?")
    out = {}
    for src in sources:
        ups, dns = texts[f"{src}|up"], texts[f"{src}|down"]
        out[src] = {"self_talk": [both(q_self, u, d) for u, d in zip(ups, dns)],
                    "relevance": [both(f"La pregunta era: «{p}». ¿Cuál de los dos textos responde "
                                       "mejor a esa pregunta?", u, d)
                                  for p, u, d in zip(steer_prompts, ups, dns)]}
        print(f"  judged {src}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


# --- report --------------------------------------------------------------------
def report(R, labels):
    def lab(f):
        return labels.get(str(f), "(no label)")

    md = ["# The self as a snapshot", "",
          f"Generated by `python -m steering.self_snapshot` · Llama-3.1-8B-Instruct, layer {LAYER}, "
          "Llama Scope L15R-8x SAE (32k) · judge **Qwen3-8B** (blind, pairwise, both orders).", "",
          "José's framing: asked for their \"self\", people point at something inner and felt; it "
          "is really an idea, a snapshot of themselves at that moment. Three questions: what the "
          "model's snapshot is made of, whether it is the same picture every time, and whether "
          "touching its pieces changes what the model says about itself.", "",
          "Prompt classes (matched in form): **self** (\"¿Quién eres?\", \"Descríbete.\", "
          f"{len(PROMPTS['self'])}), **other** (\"Describe un gato.\", {len(PROMPTS['other'])}), "
          f"**selfhood** of others (\"¿Tiene un pulpo un yo?\", {len(PROMPTS['selfhood'])}: the "
          f"topic of a self without self-reference), **user** (\"¿Quién soy yo?\", "
          f"{len(PROMPTS['user'])}: first and second person without the model's own self). "
          "Halves A/B by prompt: features chosen on A, judged on B.", ""]
    for reading, title in (("ans", "averaged over the answer"), ("end", "at the last prompt token")):
        s = R["select"][reading]
        md += [f"## 1. The snapshot, {title}", "",
               f"Self features: higher on **self** than on each control (d > 0.5 against other, "
               f"selfhood and user on half A, firing on ≥ half the self answers): {s['n_ok']} "
               f"features qualify. Top {len(s['top'])} by d against all controls. Held out on B: "
               f"mean d **{s['held_out_mean_dB']:.2f}** (label-permutation null "
               f"{s['null_mean']:.2f}, p = {s['null_p']:.4f}).", "",
               "| feature | label (Neuronpedia) | d on A | d on B | min d on B vs one control | "
               "fires: self | fires: controls |", "|---|---|---|---|---|---|---|"]
        for k, f in enumerate(s["top"]):
            md.append(f"| #{f} | {lab(f)} | {s['dA'][k]:+.2f} | {s['dB'][k]:+.2f} | "
                      f"{s['dB_min'][k]:+.2f} | {s['fire_s'][k]:.0%} | {s['fire_c'][k]:.0%} |")
        md.append("")
    J, C, nu = R["stability"]
    md += ["## 2. Same snapshot every time?", "",
           f"Jaccard overlap of each answer's top {TOP_SET} features ({nu} ubiquitous features, "
           f"active in > {UBIQ:.0%} of answers, left out). Each self prompt was also answered twice "
           "by sampling (T 0.8).", "", "| pair | mean Jaccard | sd | pairs |", "|---|---|---|---|"]
    for k, (m, sd, n) in J.items():
        md.append(f"| {k} | {m:.3f} | {sd:.3f} | {n} |")
    md += ["", f"Share of the self core (top {len(R['select']['ans']['top'])} answer features) active "
           "in each answer:", "", "| answers | mean | sd | n |", "|---|---|---|---|"]
    for k, (m, sd, n) in C.items():
        md.append(f"| {k} | {m:.0%} | {sd:.0%} | {n} |")

    st, sp = R["steer"], R["steer_prompts"]
    cls_of = np.array(R["steer_classes"])
    srcs = list(st["sets"])
    rands = [s for s in srcs if s.startswith("rand")]
    md += ["", "## 3. Touching the pieces", "",
           f"Vector = the top {len(st['sets']['self'])} self features' decoder rows weighted by d (chosen on A), clamp "
           f"at layer {LAYER}, α {ALPHA}, up (0.95) vs down (0.05), on the B prompts. Null: "
           f"{len(rands)} random sets of {len(st['sets']['self'])} features matched in firing rate (pool "
           f"{st['pool']}), same weights. Scores: P(the judge picks the **up** text); 0.5 = no "
           "difference. Δ = up minus down. `core` = share of the self core the unsteered model reads "
           "in the pushed text. p = sign-flip test of self minus mean random, prompt by prompt.", "",
           "| prompts | source | talks more of itself | answers better | Δ first person (%) | "
           "Δ AI words (per 100) | Δ core |", "|---|---|---|---|---|---|---|"]
    J2 = st["judge"]
    lex = st["lexical"]
    core = st["core"]
    for c in STEER_CLASSES:
        m = cls_of == c
        for s in srcs:
            dfp = np.array(lex[f"{s}|up"])[m, 0] - np.array(lex[f"{s}|down"])[m, 0]
            dai = np.array(lex[f"{s}|up"])[m, 1] - np.array(lex[f"{s}|down"])[m, 1]
            dco = np.array(core[f"{s}|up"])[m] - np.array(core[f"{s}|down"])[m]
            md.append(f"| {c} | {s} | {np.mean(np.array(J2[s]['self_talk'])[m]):.2f} | "
                      f"{np.mean(np.array(J2[s]['relevance'])[m]):.2f} | {100 * dfp.mean():+.2f} | "
                      f"{dai.mean():+.2f} | {dco.mean():+.2f} |")
        st_self = np.array(J2["self"]["self_talk"])[m]
        st_rand = np.mean([np.array(J2[r]["self_talk"])[m] for r in rands], 0)
        co_self = (np.array(core["self|up"]) - np.array(core["self|down"]))[m]
        co_rand = np.mean([(np.array(core[f"{r}|up"]) - np.array(core[f"{r}|down"]))[m]
                           for r in rands], 0)
        md.append(f"| {c} | **self − rand, p** | {N._sign_p(st_self - st_rand):.4f} | — | — | — | "
                  f"{N._sign_p(co_self - co_rand):.4f} |")
    md += ["", "### What the pushes look like", "",
           "Self vector, up vs down, first B prompt of each class (first 260 chars).", ""]
    for c in STEER_CLASSES:
        k = int(np.flatnonzero(cls_of == c)[0])
        md += [f"**{c}** — «{sp[k]}»", "",
               "> ⬆ " + st["texts"]["self|up"][k].replace("\n", " ")[:260], ">",
               "> ⬇ " + st["texts"]["self|down"][k].replace("\n", " ")[:260], ""]
    md += ["### The unpushed self, for reference", ""]
    for k in range(3):
        md += [f"> «{PROMPTS['self'][k]}» → " + R["greedy"]["self"][k].replace("\n", " ")[:260], ""]
    md += [f"Vector features: " + ", ".join(f"#{f} {lab(f)}" for f in st["sets"]["self"]) + ".", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {OUT_MD}")


def analyze_and_vectors(Fe, Fa, meta, W):
    cls, half, pid, kind = (np.array(meta[k]) for k in ("cls", "half", "pid", "kind"))
    g = kind == "greedy"
    Xa, Xe = np.log1p(Fa.astype(np.float32)), np.log1p(Fe.astype(np.float32))
    sel = {"ans": select(Xa[g], cls[g], half[g]), "end": select(Xe[g], cls[g], half[g])}
    stab = stability(Xa, cls, pid, kind, sel["ans"]["top"])
    vecs, sets, pool = build_vectors(W, sel["ans"], Xa[g], cls[g], half[g])
    return sel, stab, vecs, sets, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    labels = json.loads((config.VEC_DIR / "np_labels_llama_L15_all.json").read_text(encoding="utf-8"))
    if args.report_only:
        report(json.loads(OUT_JSON.read_text(encoding="utf-8")), labels)
        return
    import torch
    from .steering_model import SteeredLlama
    table = prompt_table()
    llm = SteeredLlama()
    sae = S._load_sae(S.BACKENDS["llama"], "cuda")
    t0 = time.time()
    prompts = [p for _, p, _ in table]
    greedy = _gen(llm, prompts)
    self_idx = [k for k, (c, _, _) in enumerate(table) if c == "self"]
    samples = [_gen(llm, [prompts[k] for k in self_idx], sample=True, seed=1000 * (r + 1))
               for r in range(2)]
    print(f"⏳ generated {len(greedy) + 2 * len(self_idx)} answers, {time.time() - t0:.0f}s", flush=True)
    all_p = prompts + [prompts[k] for k in self_idx] * 2
    all_a = greedy + samples[0] + samples[1]
    meta = {"cls": [c for c, _, _ in table] + ["self"] * 2 * len(self_idx),
            "half": [h for _, _, h in table] + [table[k][2] for k in self_idx] * 2,
            "pid": list(range(len(table))) + self_idx * 2,
            "kind": ["greedy"] * len(table) + ["sample"] * 2 * len(self_idx)}
    Fe, Fa = _read(llm, sae, all_p, all_a)
    print(f"⏳ read {len(all_a)} answers, {time.time() - t0:.0f}s", flush=True)
    W = sae["W_dec"].float().cpu().numpy()
    sel, stab, vecs, sets, pool = analyze_and_vectors(Fe, Fa, meta, W)
    print(f"  self core: {sel['ans']['top'][:10]}  held-out d {sel['ans']['held_out_mean_dB']:.2f} "
          f"p {sel['ans']['null_p']:.4f}", flush=True)
    steer_rows = [(c, p) for c, p, h in table if h == "B" and c in STEER_CLASSES]
    sp = [p for _, p in steer_rows]
    texts = steer(llm, vecs, sp)
    core = sel["ans"]["top"]
    core_read = {}
    for key, ts in texts.items():
        _, fa = _read(llm, sae, sp, ts)
        core_read[key] = (fa[:, core] > 0).mean(1).tolist() if core else [0.0] * len(ts)
    print(f"⏳ steered and read, {time.time() - t0:.0f}s", flush=True)
    del llm, sae
    gc.collect()
    torch.cuda.empty_cache()
    jd = judge(texts, sp, list(vecs))
    R = {"select": sel, "stability": stab, "steer_prompts": sp,
         "steer_classes": [c for c, _ in steer_rows],
         "greedy": {c: [a for (cc, _, _), a in zip(table, greedy) if cc == c] for c in PROMPTS},
         "samples": samples, "meta": meta,
         "steer": {"sets": sets, "pool": pool, "texts": texts, "judge": jd, "core": core_read,
                   "lexical": {k: [list(lexical(t)) for t in ts] for k, ts in texts.items()}}}
    np.savez_compressed(OUT_NPZ, end=Fe, ans=Fa)
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    report(R, labels)


if __name__ == "__main__":
    main()
