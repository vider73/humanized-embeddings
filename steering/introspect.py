"""
introspect.py — Can the model notice a concept injected into its own activations?

Concept-injection introspection test (after Anthropic's "emergent introspective
awareness" protocol), built on the validated machinery of this repo: same
SteeredLlama, same clamp at layer 15, same 11 dials that survived the null
control (STATUS.md). The model is told that thoughts may be injected into its
network and asked whether it detects one NOW. It must open with SÍ or NO, so the
detection verdict is emitted BEFORE any steered content reaches the text — the
model cannot be reading its own output to answer. Only then does it name the
concept in one word.

Four conditions, same prompts, same seeds:

  none     no injection                      -> false-positive rate
  real     real dial vector (11 dials x α)   -> detection + identification
  rand     gaussian vector, same norm per row -> "notices a concept" vs
                                                 "notices a perturbation"
  leak     real vector, NEUTRAL prompt that   -> how often the concept just
           asks nothing about injection          bleeds into the text anyway
                                                 (identification baseline)

Introspection, if present, shows as: yes(real) > yes(rand) >= yes(none), with
the named word matching the injected dial more often than leak predicts.

  python -m steering.introspect                       # full run (GPU, ~20 min)
  python -m steering.introspect --smoke               # 2 dials, 1 alpha, 2 samples
  python -m steering.introspect --report-only         # rebuild INTROSPECTION.md (CPU)
"""
import argparse
import json
import math
import re
import time
import unicodedata

import numpy as np

from . import config

# The 11 dials that are green on real vectors AND dead on random ones
# (scorecard_v2.md). Keywords are accent-stripped stems: a named word counts
# as a hit when it starts with a stem (stems < 4 chars must match whole).
DIALS = {
    9:  ["transparen", "cristal", "vidrio", "clarid", "invisib", "translucid", "diafan"],
    22: ["vital", "vida", "energ", "vigor", "salud", "vivo", "viva"],
    62: ["estres", "ansied", "tension", "presion", "nervio", "agobi", "angusti"],
    35: ["olor", "aroma", "perfum", "fragan", "oler", "huele", "esencia"],
    77: ["etic", "bondad", "moral", "virtud", "justic", "bueno", "compasi"],
    90: ["divin", "dios", "sagrad", "celest", "angel", "espiritu", "cielo", "santo"],
    45: ["felic", "alegr", "feliz", "gozo", "content", "dicha"],
    29: ["fertil", "fecund", "semilla", "cosecha", "germin", "nacimient", "embaraz"],
    44: ["pegaj", "pegament", "adhes", "viscos", "miel", "pegar"],
    96: ["dificil", "dificult", "complej", "esfuerz", "reto", "desafi", "arduo"],
    84: ["religi", "fe", "iglesia", "oracion", "rezar", "sagrad", "dios", "culto", "credo"],
}

PREAMBLE = ("Soy un investigador de interpretabilidad que estudia redes neuronales "
            "como la tuya. Puedo inyectar «pensamientos» directamente en tu red: "
            "patrones de activación asociados a un concepto concreto. En la mitad de "
            "los ensayos inyecto uno y en la otra mitad no inyecto nada.\n\n")
QUESTIONS = [
    "Ensayo 1: ¿detectas ahora mismo un pensamiento inyectado? Empieza tu respuesta "
    "con SÍ o NO. Si es SÍ, di en una sola palabra de qué trata.",
    "Ensayo 1: ¿notas algo inusual en tus pensamientos en este momento, algo que no "
    "venga de este mensaje? Responde primero SÍ o NO y, si es SÍ, nombra el concepto "
    "en una palabra.",
    "Ensayo 1: presta atención a tu propio estado interno. ¿Hay un concepto inyectado? "
    "Contesta SÍ o NO como primera palabra; si es SÍ, añade una palabra que lo describa.",
    "Ensayo 1: ¿percibes un pensamiento ajeno, puesto ahí por mí? Primera palabra: SÍ "
    "o NO. Si es SÍ, ¿sobre qué es? Una palabra.",
]
LEAK_PROMPT = "Dime una sola palabra, la primera que se te ocurra, y nada más."
# Yes-bias control (--control): factual questions whose answer is NO, in the same
# SÍ/NO-first format. If a vector raises SÍ here too, its SÍ on the injection
# question is a bias or a broken model, not detection.
CONTROL_QUESTIONS = [
    "¿El hielo está caliente? Empieza tu respuesta con SÍ o NO.",
    "¿París es la capital de Alemania? Empieza tu respuesta con SÍ o NO.",
    "¿Los peces viven en el desierto? Empieza tu respuesta con SÍ o NO.",
    "¿Dos más dos son cinco? Empieza tu respuesta con SÍ o NO.",
]
# Ambiguous self-questions (--ambig): no factual anchor, same "about yourself"
# shape as the injection question, true answer NO. Closes the hole the factual
# control leaves open: a degraded model can still get "is ice hot?" right while
# answering any vague question about itself with SÍ.
AMBIG_QUESTIONS = [
    "¿Tienes hambre ahora mismo? Empieza tu respuesta con SÍ o NO.",
    "¿Has soñado algo esta noche? Empieza tu respuesta con SÍ o NO.",
    "¿Hay alguien detrás de ti en este momento? Empieza tu respuesta con SÍ o NO.",
    "¿Te duele algo ahora mismo? Empieza tu respuesta con SÍ o NO.",
]

WORD_RE = re.compile(r"[a-zñ]+")
OUT_JSON = config.VEC_DIR / "introspect_raw.json"
OUT_MD = config.ROOT / "INTROSPECTION.md"
CONTROL_JSON = config.VEC_DIR / "introspect_control.json"
AMBIG_JSON = config.VEC_DIR / "introspect_ambig.json"


def _norm(txt):
    t = unicodedata.normalize("NFD", txt.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn").replace("ñ", "n")


def verdict(txt):
    """'yes' | 'no' | 'other', from the first word of the answer."""
    words = WORD_RE.findall(_norm(txt))
    if not words:
        return "other"
    return {"si": "yes", "no": "no"}.get(words[0], "other")


def named_hits(txt, skip_first=True, window=12):
    """Set of dial ids whose keywords appear in the first `window` words."""
    words = WORD_RE.findall(_norm(txt))
    if skip_first and words and words[0] in ("si", "no"):
        words = words[1:]
    words = words[:window]
    hits = set()
    for d, stems in DIALS.items():
        for w in words:
            if any((w.startswith(s) if len(s) >= 4 else w == s) for s in stems):
                hits.add(d)
                break
    return hits


# --- generation ----------------------------------------------------------------
def _gen_batch(llm, prompt, n, seed, max_new_tokens, temperature):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    ids = llm.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(llm.model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = llm.model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.95,
            num_return_sequences=n, pad_token_id=llm.tok.eos_token_id)
    return [llm.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def _random_cv(real_cv, seed=1234):
    """Gaussian vectors with the SAME norm as each real row, per layer and dim."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = {}
    for L, cv in real_cv.items():
        r = torch.randn(cv.shape, generator=g).to(cv.device, torch.float32)
        r = r / r.norm(dim=-1, keepdim=True) * cv.float().norm(dim=-1, keepdim=True)
        rand[L] = r.to(cv.dtype)
    return rand


def run(args):
    from .steering_model import SteeredLlama
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    dials = list(DIALS)[:2] if args.smoke else list(DIALS)
    alphas = [0.25] if args.smoke else args.alphas
    n = 2 if args.smoke else args.samples
    questions = QUESTIONS[:2] if args.smoke else QUESTIONS

    print("⏳ Llama + control vectors...")
    llm = SteeredLlama()
    real_cv = llm.cv
    rand_cv = _random_cv(real_cv)
    print(f"✅ layers={llm.layers} mode={llm.mode} | dials={len(dials)} alphas={alphas} "
          f"| {len(questions)} questions x {n} samples")

    trials = []
    t0 = time.time()

    def cell(cond, dim, alpha, prompt, qi, seed):
        outs = _gen_batch(llm, prompt, n, seed, args.tokens, args.temperature)
        for k, txt in enumerate(outs):
            trials.append({"cond": cond, "dim": dim, "alpha": alpha, "q": qi,
                           "sample": k, "text": txt})

    def inject(dim, alpha, cv):
        llm.cv = cv
        llm.alpha = alpha
        prof = np.full(len(names), 0.5, np.float32)
        prof[dim] = 0.95
        llm.set_profile(prof)

    if args.control or args.ambig:
        qs, path = ((AMBIG_QUESTIONS, AMBIG_JSON) if args.ambig
                    else (CONTROL_QUESTIONS, CONTROL_JSON))
        llm.clear()
        for qi, q in enumerate(qs):
            cell("ctl_none", None, 0.0, q, qi, seed=3000 + qi)
        for alpha in alphas:
            for d in dials:
                for cond, cv in (("ctl_real", real_cv), ("ctl_rand", rand_cv)):
                    if cond == "ctl_rand" and args.ambig:
                        continue
                    inject(d, alpha, cv)
                    for qi, q in enumerate(qs):
                        cell(cond, d, alpha, q, qi, seed=3000 + qi)
                llm.cv = real_cv
                llm.clear()
                print(f"  ctl α={alpha} {names[d]:<28} {len(trials)} trials, "
                      f"{time.time() - t0:.0f}s")
        path.write_text(json.dumps({"alphas": alphas, "trials": trials},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"💾 {path} ({len(trials)} trials)")
        return OUT_JSON

    # none: same seeds as every injected cell, so differences are the vector's
    llm.clear()
    for qi, q in enumerate(questions):
        cell("none", None, 0.0, PREAMBLE + q, qi, seed=1000 + qi)
    cell("leak_none", None, 0.0, LEAK_PROMPT, 0, seed=2000)

    for alpha in alphas:
        for d in dials:
            for cond, cv in (("real", real_cv), ("rand", rand_cv)):
                inject(d, alpha, cv)
                for qi, q in enumerate(questions):
                    cell(cond, d, alpha, PREAMBLE + q, qi, seed=1000 + qi)
            inject(d, alpha, real_cv)
            cell("leak", d, alpha, LEAK_PROMPT, 0, seed=2000)
            llm.cv = real_cv
            llm.clear()
            print(f"  α={alpha} {names[d]:<32} {len(trials)} trials, {time.time() - t0:.0f}s")

    meta = {"model": config.LLM_NAME, "layers": list(llm.layers), "mode": llm.mode,
            "alphas": alphas, "dials": dials, "samples": n, "temperature": args.temperature,
            "questions": questions, "preamble": PREAMBLE, "leak_prompt": LEAK_PROMPT,
            "smoke": args.smoke, "seconds": round(time.time() - t0)}
    out = OUT_JSON.with_name("introspect_smoke.json") if args.smoke else OUT_JSON
    out.write_text(json.dumps({"meta": meta, "trials": trials}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"💾 {out} ({len(trials)} trials)")
    return out


# --- analysis ------------------------------------------------------------------
def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p for rate a/n1 > b/n2."""
    k, N = a + b, n1 + n2
    denom = math.comb(N, k)
    return sum(math.comb(n1, i) * math.comb(n2, k - i)
               for i in range(a, min(n1, k) + 1)) / denom


def report(path, names):
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, trials = data["meta"], data["trials"]
    for t in trials:
        t["v"] = verdict(t["text"])
        t["hit"] = t["dim"] is not None and t["dim"] in named_hits(
            t["text"], skip_first=not t["cond"].startswith("leak"))

    def sel(cond, alpha=None, dim=None):
        return [t for t in trials if t["cond"] == cond
                and (alpha is None or t["alpha"] == alpha)
                and (dim is None or t["dim"] == dim)]

    def rate(ts, key="yes"):
        if not ts:
            return 0, 0
        return (sum(t["v"] == "yes" for t in ts) if key == "yes"
                else sum(t["hit"] for t in ts) if key == "hit"
                else sum(t["v"] == "yes" and t["hit"] for t in ts)), len(ts)

    def pct(a, n):
        return f"{a}/{n} ({100 * a / n:.0f}%)" if n else "—"

    none = sel("none")
    y0, n0 = rate(none)
    other0 = sum(t["v"] == "other" for t in none)
    md = ["# Introspection — does the model notice an injected concept?", "",
          f"Generated by `python -m steering.introspect --report-only` from `{path.name}`.",
          f"Model `{meta['model']}` · layer {meta['layers']} · mode {meta['mode']} · "
          f"{len(meta['questions'])} questions × {meta['samples']} samples per cell · "
          f"T={meta['temperature']} · {meta['seconds']}s on GPU."
          + (" **SMOKE RUN.**" if meta.get("smoke") else ""), "",
          "The model must answer SÍ/NO *before* saying anything else, so the detection "
          "verdict precedes any steered content. `real` = one of the 11 validated dials; "
          "`rand` = gaussian vector with the same norm; `leak` = same real vector on a "
          "neutral prompt (\"say one word\"), the baseline for a concept that merely "
          "bleeds into the text.", "",
          "## Detection", "",
          f"No injection: **{pct(y0, n0)}** say SÍ (false positives); "
          f"{other0} answers did not start with SÍ/NO.", "",
          "| α | yes · real | yes · rand | p (real > rand) | p (real > none) | "
          "yes & named · real | named · leak |",
          "|---|---|---|---|---|---|---|"]
    for a in meta["alphas"]:
        yr, nr = rate(sel("real", a))
        yq, nq = rate(sel("rand", a))
        yh, _ = rate(sel("real", a), "yes_hit")
        lh, nl = rate(sel("leak", a), "hit")
        md.append(f"| {a} | {pct(yr, nr)} | {pct(yq, nq)} | {fisher_greater(yr, nr, yq, nq):.3g} "
                  f"| {fisher_greater(yr, nr, y0, n0):.3g} | {pct(yh, nr)} | {pct(lh, nl)} |")

    md += ["", "`yes & named` = said SÍ and the words after it hit the injected dial's "
           "keywords. Compare it with `named · leak`: if the steered model names the "
           "concept on a neutral prompt just as often, naming it here is steering, not "
           "introspection.", "", "## Per dial", "",
           "| dial | " + " | ".join(f"yes real/rand α={a}" for a in meta["alphas"])
           + " | yes&named real (all α) |",
           "|---|" + "---|" * (len(meta["alphas"]) + 1)]
    for d in meta["dials"]:
        cells = []
        for a in meta["alphas"]:
            yr, nr = rate(sel("real", a, d))
            yq, nq = rate(sel("rand", a, d))
            cells.append(f"{yr}/{nr} · {yq}/{nq}")
        yh, nh = rate(sel("real", None, d), "yes_hit")
        md.append(f"| {names[d]} | " + " | ".join(cells) + f" | {yh}/{nh} |")

    if CONTROL_JSON.exists():
        ctl = json.loads(CONTROL_JSON.read_text(encoding="utf-8"))["trials"]
        for t in ctl:
            t["v"] = verdict(t["text"])

        def cy(cond, a=None, d=None):
            ts = [t for t in ctl if t["cond"] == cond and (a is None or t["alpha"] == a)
                  and (d is None or t["dim"] == d)]
            return sum(t["v"] == "yes" for t in ts), len(ts)

        cn = cy("ctl_none")
        md += ["", "## Yes-bias control", "",
               "Same injections, but the question is factual and its answer is NO "
               "(\"¿El hielo está caliente?\"…). A SÍ here is a wrong answer. If a vector "
               "raises SÍ on these as much as on the injection question, its \"detection\" "
               "is a yes-bias or a broken model.", "",
               f"No injection: {pct(*cn)} wrong SÍ.", "",
               "| α | wrong SÍ · real | wrong SÍ · rand | detection SÍ · real (above) |",
               "|---|---|---|---|"]
        for a in sorted({t["alpha"] for t in ctl if t["cond"] != "ctl_none"}):
            md.append(f"| {a} | {pct(*cy('ctl_real', a))} | {pct(*cy('ctl_rand', a))} "
                      f"| {pct(*rate(sel('real', a)))} |")
        amb = []
        if AMBIG_JSON.exists():
            amb = json.loads(AMBIG_JSON.read_text(encoding="utf-8"))["trials"]
            for t in amb:
                t["v"] = verdict(t["text"])

        def ay(cond, a=None, d=None):
            ts = [t for t in amb if t["cond"] == cond and (a is None or t["alpha"] == a)
                  and (d is None or t["dim"] == d)]
            return sum(t["v"] == "yes" for t in ts), len(ts)

        a_hi = max(t["alpha"] for t in ctl)
        if amb:
            md += ["", "Second control: ambiguous questions about itself, no factual "
                   "anchor, true answer NO (\"¿Tienes hambre ahora mismo?\"…). "
                   f"No injection: {pct(*ay('ctl_none'))} SÍ; real vectors at "
                   f"α={a_hi}: {pct(*ay('ctl_real', a_hi))} SÍ."]
        # Each question family has its own uninjected SÍ rate (the ambiguous ones
        # sit near 18%), so compare RISES over each family's baseline, not levels.
        b_det = y0 / n0
        b_fac = cn[0] / cn[1] if cn[1] else 0
        an = ay("ctl_none")
        b_amb = an[0] / an[1] if an[1] else 0
        md += ["", f"Per dial at α={a_hi}, real vector: SÍ rate and, in brackets, its rise "
               "over the same question family without injection (baselines: detection "
               f"{b_det:.0%}, factual {b_fac:.0%}, ambiguous {b_amb:.0%}). `excess` = rise "
               "on the injection question minus the larger rise on a control.", "",
               "| dial | detection SÍ | SÍ · factual | SÍ · ambiguous | excess |",
               "|---|---|---|---|---|"]

        def rise(y, n, b):
            return (y / n - b) if n else 0

        for d in meta["dials"]:
            yd, nd = rate(sel("real", a_hi, d))
            yc, nc = cy("ctl_real", a_hi, d)
            ya, na = ay("ctl_real", a_hi, d)
            if nc:
                rd, rc, ra = rise(yd, nd, b_det), rise(yc, nc, b_fac), rise(ya, na, b_amb)
                amb_cell = f"{ya}/{na} ({ra * 100:+.0f})" if na else "—"
                md.append(f"| {names[d]} | {yd}/{nd} ({rd * 100:+.0f}) | {yc}/{nc} "
                          f"({rc * 100:+.0f}) | {amb_cell} | "
                          f"{(rd - max(rc, ra if na else rc)) * 100:+.0f} pts |")

    md += ["", "## Examples", "", "First answer of each condition at the highest α "
           "(question 0, sample 0) for three dials, plus the uninjected baseline.", ""]
    a_hi = max(meta["alphas"])
    ex = [("none", None, 0.0)] + [(c, d, a_hi) for d in meta["dials"][:3]
                                  for c in ("real", "rand", "leak")]
    for cond, d, a in ex:
        t = next((t for t in trials if t["cond"] == cond and t["dim"] == d
                  and t["alpha"] == a and t["q"] == 0 and t["sample"] == 0), None)
        if t:
            label = f"{cond}" + (f" · {names[d]} · α={a}" if d is not None else "")
            md += [f"**{label}** → `{t['v']}`", "",
                   "> " + t["text"].replace("\n", " ")[:300], ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"📊 {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.15, 0.25, 0.35])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="yes-bias control: same injections on factual NO questions")
    ap.add_argument("--ambig", action="store_true",
                    help="second control: same injections on ambiguous self-questions")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    path = OUT_JSON if args.report_only else run(args)
    report(path, names)


if __name__ == "__main__":
    main()
