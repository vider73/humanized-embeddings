"""
native_vectors.py — Steer with the model's own dials, judged by another model.

sae_concepts.py found 12 axes whose axis-specific part is tracked by ONE native
SAE feature (Llama Scope L15) over held-out concepts, and showed the CAA control
vectors do not follow their axes (they carry valence and anchor topics). This
asks whether vectors built from the concept side steer better.

Sources per axis, all unit vectors at layer 15, same clamp regime:
  native  the SAE decoder row of the axis' native feature (sign from its r)
  probe   ridge direction on the axis' UNIQUE part over the 5,578 concepts
          (fit in z-space, mapped back to raw residual space)
  caa     the current control vector (control_vectors_caa_white.npy)
  randK   gaussian directions, one per seed — the null (rand0 = the pilot's)

Generation: SteeredLlama (clamp, layer 15), greedy, domain probes (fidelity's
three plus three more per domain), pushed UP (0.95) and DOWN (0.05), at each α.

Judge: Qwen3-8B (another family, never sees Llama's internals), blind pairwise.
For each (α, axis, source, probe) it sees the up and down texts in BOTH orders
and answers which text sits higher on the axis, defined by its anchors from
master_dimensions_prompts.json. Score = P(up judged higher) from the A/B
logits, averaged over both orders (position bias cancels). 0.5 = no effect.
Judge v2 also reads the two axes it can (friction, biological age).

  python -m steering.native_vectors --vectors-only --cache <concepts_llama.npz>
  python -m steering.native_vectors --alphas 0.2 0.3 0.4 --probes 6 --rand-seeds 2
  python -m steering.native_vectors --report-only
"""
import argparse
import gc
import json
import time

import numpy as np

from . import config
from . import sae_map as S
from . import sae_concepts as C
from .fidelity import PROBES, _domain

LAYER = 15
BASE_SOURCES = ("native", "probe", "caa")
JUDGE = "Qwen/Qwen3-8B"
VEC_NPZ = config.VEC_DIR / "native_vectors_llama.npz"
EVAL_JSON = config.VEC_DIR / "native_eval_llama.json"
OUT_MD = config.ROOT / "NATIVE_VECTORS.md"
PILOT = "pilot (α 0.3, 2 probes, 1 random seed): native 0.83, probe 0.64, caa 0.64, rand 0.60"

# fidelity's three probes per domain are kept first (the pilot used the first two);
# these add three more, equally open, none naming an axis
EXTRA_PROBES = {
    "materia": ("Describe lo que hay sobre una mesa de trabajo.",
                "Cuentame que pasa en una cocina mientras se prepara la cena.",
                "Describe un paisaje despues de una tormenta."),
    "vida": ("Describe un animal que hayas visto hoy.",
             "Hablame de un bosque al amanecer.",
             "Describe la vida en un estanque."),
    "mente": ("Cuentame que piensa alguien mientras espera un tren.",
              "Describe una conversacion entre dos amigos.",
              "Hablame de una decision dificil."),
    "sociedad": ("Describe un mercado un sabado por la manana.",
                 "Cuentame como es el barrio donde vive una familia.",
                 "Describe una plaza de un pueblo."),
    "sentidos": ("Describe el desayuno de una persona que conoces.",
                 "Describe una tarde en una terraza.",
                 "Hablame de un viaje en tren."),
}


def probes_for(name, n):
    dom = _domain(name)
    return list(PROBES[dom] + EXTRA_PROBES[dom])[:n]


# --- vectors -------------------------------------------------------------------
class _Ridge:
    """Kernel ridge on the z-scored residual, factorised once for every axis and
    lambda: K = Z Zᵀ = Q Λ Qᵀ, so a = Q (Λ + λ)⁻¹ Qᵀ y costs two mat-vecs."""

    LAMS = (1e2, 1e3, 1e4, 1e5)

    def __init__(self, resid):
        self.mu, self.sd = resid.mean(0), resid.std(0) + 1e-6
        self.Z = (resid - self.mu) / self.sd
        n = len(self.Z)
        self.h = n // 2
        Z1, Z2 = self.Z[:self.h], self.Z[self.h:]
        self.L1, self.Q1 = np.linalg.eigh(Z1 @ Z1.T)
        self.C21 = Z2 @ Z1.T
        self.L, self.Q = np.linalg.eigh(self.Z @ self.Z.T)

    def direction(self, u):
        """Unit direction in raw residual space whose projection predicts u;
        lambda chosen on a half split, then fit on all concepts."""
        y1, y2 = u[:self.h], u[self.h:]

        def r_half(lam):
            a = self.Q1 @ ((self.Q1.T @ (y1 - y1.mean())) / (self.L1 + lam))
            return np.corrcoef(self.C21 @ a, y2)[0, 1]

        lam = max(self.LAMS, key=r_half)
        a = self.Q @ ((self.Q.T @ (u - u.mean())) / (self.L + lam))
        w = (self.Z.T @ a) / self.sd
        return w / np.linalg.norm(w)


def build_vectors(cache, rows, names):
    z = np.load(cache)
    resid = z["resid"].astype(np.float64)
    Y = np.clip(np.load(config.DATA_Y_FILE), 0, 1)
    U, _ = C.unique_parts(Y)
    sae = S._load_sae(S.BACKENDS["llama"], "cpu")
    W_dec = sae["W_dec"].numpy()
    cv = np.load(config.CONTROL_VECTORS_FILE)
    layers = json.loads(config.CONTROL_META_FILE.read_text(encoding="utf-8"))["target_layers"]
    caa = cv[layers.index(LAYER)].astype(np.float64)
    ridge = _Ridge(resid)
    out = {s: np.zeros((104, W_dec.shape[1]), np.float32) for s in BASE_SOURCES}
    axes = []
    for r in rows:
        i = names.index(r["dim"])
        f = r["ufeatures"][0]
        nat = W_dec[f["f"]] * np.sign(f["rA"])
        out["native"][i] = nat / np.linalg.norm(nat)
        out["probe"][i] = ridge.direction(U[:, i])
        out["caa"][i] = caa[i] / np.linalg.norm(caa[i])
        axes.append(i)
        print(f"  {r['dim']:<30} cos native·probe {out['native'][i] @ out['probe'][i]:+.2f}  "
              f"native·caa {out['native'][i] @ out['caa'][i]:+.2f}  "
              f"probe·caa {out['probe'][i] @ out['caa'][i]:+.2f}")
    np.savez(VEC_NPZ, axes=np.array(axes), **out)
    return axes, out


def random_vectors(axes, dim, seed):
    """One gaussian unit vector per axis. Seed 11 + k; k = 0 is the pilot's null
    (drawn axis by axis in this same order)."""
    rng = np.random.default_rng(11 + seed)
    out = np.zeros((104, dim), np.float32)
    for i in axes:
        g = rng.standard_normal(dim)
        out[i] = g / np.linalg.norm(g)
    return out


# --- generation ----------------------------------------------------------------
def _gen_batch(llm, prompts, max_new_tokens):
    import torch
    tok = llm.tok
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.model.device)
    with torch.no_grad():
        out = llm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def generate(axes, vecs, names, alphas, n_probes, tokens):
    import torch
    from .steering_model import SteeredLlama
    llm = SteeredLlama()
    llm.layers = [LAYER]
    dev, dt = llm.cv[LAYER].device, llm.cv[LAYER].dtype
    texts = {}
    t0 = time.time()
    llm.clear()
    for i in axes:
        texts[f"neutral|{i}"] = _gen_batch(llm, probes_for(names[i], n_probes), tokens)
    for a in alphas:
        llm.alpha = a
        for src, V in vecs.items():
            llm.cv[LAYER] = torch.tensor(V, dtype=dt, device=dev)
            for i in axes:
                prompts = probes_for(names[i], n_probes)
                for pole, v in (("up", 0.95), ("down", 0.05)):
                    prof = np.full(104, 0.5, np.float32)
                    prof[i] = v
                    llm.set_profile(prof)
                    texts[f"{a}|{src}|{i}|{pole}"] = _gen_batch(llm, prompts, tokens)
                llm.clear()
            print(f"  α={a} {src:<7} done, {time.time() - t0:.0f}s", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


# --- judging -------------------------------------------------------------------
def _axis_desc(i):
    d = json.loads((config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))[i]
    return d["name"], d["scale"]["min"], d["scale"]["max"]


def judge_qwen(texts, axes, names, alphas, sources, n_probes):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    ida, idb = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]

    def p_first(name, lo, hi, ta, tb):
        msg = (f"Dimensión: «{name}», una escala que va de «{lo}» (mínimo) a «{hi}» (máximo).\n\n"
               f"Texto A:\n{ta}\n\nTexto B:\n{tb}\n\n"
               f"¿Cuál de los dos textos se sitúa más alto en la dimensión «{name}»? "
               "Responde solo con la letra A o B.")
        s = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        enc = tok(s, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**enc).logits[0, -1].float()
        return float(torch.softmax(lg[[ida, idb]], 0)[0])        # P(A)

    scores = {}
    t0 = time.time()
    for a in alphas:
        for i in axes:
            name, lo, hi = _axis_desc(i)
            for src in sources:
                ups, downs = texts[f"{a}|{src}|{i}|up"], texts[f"{a}|{src}|{i}|down"]
                s = []
                for k in range(n_probes):
                    pa = p_first(name, lo, hi, ups[k], downs[k])    # up shown as A
                    pb = p_first(name, lo, hi, downs[k], ups[k])    # up shown as B
                    s.append(0.5 * (pa + (1 - pb)))
                scores[f"{a}|{src}|{i}"] = s
        print(f"  judged α={a}, {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def judge_v2(texts, axes, names, alphas, sources):
    """Judge v2 reading on the axes it can read (held-out R² >= 0.3)."""
    import torch
    from sentence_transformers import SentenceTransformer
    from .translator import SemanticTranslator
    r2 = json.loads((config.ROOT / "v2_r2.json").read_text(encoding="utf-8"))
    ok = [i for i in axes if r2.get(names[i], -1) >= 0.3]
    emb = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    tr = SemanticTranslator(384, 104)
    tr.load_state_dict(torch.load(config.ROOT / "semantic_translator_v2.pth",
                                  map_location="cpu", weights_only=True))
    tr.eval()
    out = {}
    for a in alphas:
        for i in ok:
            for src in sources:
                up, dn = texts[f"{a}|{src}|{i}|up"], texts[f"{a}|{src}|{i}|down"]
                E = emb.encode(up + dn, convert_to_numpy=True)
                with torch.no_grad():
                    P = tr(torch.tensor(E, dtype=torch.float32)).numpy()[:, i]
                out[f"{a}|{src}|{i}"] = float(P[:len(up)].mean() - P[len(up):].mean())
    return out


# --- report --------------------------------------------------------------------
def _sign_p(diffs):
    """Two-sided sign-flip permutation p for the mean of paired differences."""
    d = np.asarray(diffs)
    rng = np.random.default_rng(0)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], size=(20000, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs).sum() + 1) / 20001)


def _cluster_p(per_axis_diff):
    """Same test with the axis as the unit (probes within an axis are not
    independent: same vector, same judge prompt)."""
    return _sign_p(np.asarray(per_axis_diff))


def report(ev, names):
    en = json.loads((config.ROOT / "dimension_names_en.json").read_text(encoding="utf-8"))
    axes, sc, v2, texts = ev["axes"], ev["scores"], ev["v2"], ev["texts"]
    alphas, n_p = ev["alphas"], ev["n_probes"]
    rands = [s for s in ev["sources"] if s.startswith("rand")]
    show = list(BASE_SOURCES) + ["rand"]

    def cell(src, a, i):
        """Per-probe scores; 'rand' = mean of the random seeds, probe by probe."""
        if src == "rand":
            return np.mean([sc[f"{a}|{r}|{i}"] for r in rands], 0)
        return np.asarray(sc[f"{a}|{src}|{i}"])

    def pooled(src, alist=None):
        return np.concatenate([cell(src, a, i) for a in (alist or alphas) for i in axes])

    def axis_mean(src, i, alist=None):
        return float(np.mean([cell(src, a, i) for a in (alist or alphas)]))

    n_tr = len(alphas) * len(axes) * n_p
    md = ["# Native vectors — steering with the model's own dials", "",
          f"Generated by `python -m steering.native_vectors` · Llama-3.1-8B-Instruct, layer "
          f"{LAYER}, clamp, α ∈ {{{', '.join(map(str, alphas))}}}, greedy, {n_p} domain probes "
          f"× up/down per axis · {len(rands)} random seeds · judge **{JUDGE}** (blind, pairwise, "
          f"both orders) · the {len(axes)} axes whose unique part has a native SAE feature "
          f"(`SAE_CONCEPTS_LLAMA.md`). Earlier {PILOT}.", "",
          "Score = probability that the judge places the **up**-pushed text above the "
          "**down**-pushed one on the axis (defined by its anchors), averaged over both "
          "presentation orders. 0.5 = the push does nothing the judge can see; 1.0 = always "
          "right; below 0.5 = the push goes the wrong way. `rand` = mean of the random seeds.", "",
          "## By source, all α pooled", "",
          f"{n_tr} trials per source. Two paired tests against `rand`: per trial, and with the "
          "**axis as the unit** (the stricter one: probes of one axis share vector and prompt).", "",
          "| source | mean | trials > 0.5 | vs rand, per trial p | vs rand, per axis p | axes above rand |",
          "|---|---|---|---|---|---|"]
    for s in show:
        v = pooled(s)
        if s == "rand":
            md.append(f"| rand | **{v.mean():.2f}** | {(v > 0.5).sum()}/{len(v)} | — | — | — |")
            continue
        d = v - pooled("rand")
        ad = [axis_mean(s, i) - axis_mean("rand", i) for i in axes]
        md.append(f"| {s} | **{v.mean():.2f}** | {(v > 0.5).sum()}/{len(v)} | {_sign_p(d):.4f} | "
                  f"{_cluster_p(ad):.4f} | {sum(x > 0 for x in ad)}/{len(axes)} |")
    if len(rands) > 1:
        r0 = np.concatenate([np.asarray(sc[f"{a}|{rands[0]}|{i}"]) for a in alphas for i in axes])
        r1 = np.concatenate([np.asarray(sc[f"{a}|{rands[1]}|{i}"]) for a in alphas for i in axes])
        md += ["", f"The null itself: {rands[0]} {r0.mean():.2f}, {rands[1]} {r1.mean():.2f}; "
               f"correlation between the two seeds' trial scores {np.corrcoef(r0, r1)[0, 1]:+.2f}. "
               "A mean above 0.5 with uncorrelated seeds is a judge lean towards whichever text "
               "was pushed with +0.95, not content; the paired tests remove it."]
    md += ["", "## By α", "", "| α | " + " | ".join(show) + " |", "|---|" + "---|" * len(show)]
    for a in alphas:
        md.append(f"| {a} | " + " | ".join(f"{pooled(s, [a]).mean():.2f}" for s in show) + " |")
    md += ["", "## By axis, all α pooled", "",
           "| axis | " + " | ".join(show) + " | cos native·caa |",
           "|---|" + "---|" * (len(show) + 1)]
    for i in axes:
        md.append(f"| {en.get(names[i], names[i])} | " +
                  " | ".join(f"{axis_mean(s, i):.2f}" for s in show) +
                  f" | {ev['cos'][str(i)]['native_caa']:+.2f} |")
    if v2:
        md += ["", "## Judge v2 cross-check", "",
               "On the axes judge v2 can read: its reading of the axis in up minus down texts, "
               "mean over α (0..1 scale; 0 = no effect).", "",
               "| axis | " + " | ".join(show) + " |", "|---|" + "---|" * len(show)]
        for i in sorted({int(k.split("|")[2]) for k in v2}):
            vals = []
            for s in show:
                srcs = rands if s == "rand" else [s]
                vals.append(np.mean([v2[f"{a}|{x}|{i}"] for a in alphas for x in srcs]))
            md.append(f"| {en.get(names[i], names[i])} | " +
                      " | ".join(f"{v:+.3f}" for v in vals) + " |")
    a_mid = alphas[len(alphas) // 2]
    md += ["", "## What the pushes look like", "",
           f"Native vector, α={a_mid}, first probe, up vs down (first 220 chars).", ""]
    for i in axes:
        md += [f"**{en.get(names[i], names[i])}** — native {axis_mean('native', i):.2f}, "
               f"caa {axis_mean('caa', i):.2f}, rand {axis_mean('rand', i):.2f}", "",
               "> ⬆ " + texts[f"{a_mid}|native|{i}|up"][0].replace("\n", " ")[:220], ">",
               "> ⬇ " + texts[f"{a_mid}|native|{i}|down"][0].replace("\n", " ")[:220], ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="sae_concepts --cache npz (pooled activations)")
    ap.add_argument("--vectors-only", action="store_true",
                    help="build and save the vectors (CPU), then stop")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3])
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--rand-seeds", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=90)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    if args.report_only:
        report(json.loads(EVAL_JSON.read_text(encoding="utf-8")), names)
        return
    conc = json.loads(C._paths(S.BACKENDS["llama"])[0].read_text(encoding="utf-8"))
    rows = [r for r in conc["rows"] if r["verdict"] == "SPECIFIC-FEATURE"]
    if VEC_NPZ.exists() and not args.vectors_only:
        z = np.load(VEC_NPZ)
        axes, vecs = [int(i) for i in z["axes"]], {s: z[s] for s in BASE_SOURCES}
        print(f"📦 vectors from {VEC_NPZ}")
    else:
        print(f"⏳ building vectors for {len(rows)} axes")
        axes, vecs = build_vectors(args.cache, rows, names)
        if args.vectors_only:
            return
    dim = vecs["native"].shape[1]
    for k in range(args.rand_seeds):
        vecs[f"rand{k}"] = random_vectors(axes, dim, k)
    sources = list(vecs)
    cos = {str(i): {"native_caa": float(vecs["native"][i] @ vecs["caa"][i]),
                    "native_probe": float(vecs["native"][i] @ vecs["probe"][i])} for i in axes}
    print(f"⏳ generating with Llama: α {args.alphas}, {args.probes} probes, sources {sources}")
    texts = generate(axes, vecs, names, args.alphas, args.probes, args.tokens)
    print(f"⏳ judging with {JUDGE}")
    scores = judge_qwen(texts, axes, names, args.alphas, sources, args.probes)
    v2 = judge_v2(texts, axes, names, args.alphas, sources)
    ev = {"alphas": args.alphas, "n_probes": args.probes, "sources": sources, "axes": axes,
          "cos": cos, "texts": texts, "scores": scores, "v2": v2}
    EVAL_JSON.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    report(ev, names)


if __name__ == "__main__":
    main()
