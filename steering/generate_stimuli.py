"""
generate_stimuli.py — Frases ricas por polo para CAA semantico (no lexico).

El problema de derivar CAA de la etiqueta corta ("Dios Creador") es que el
vector capta los TOKENS del polo, no su significado: al steerear rompe las
palabras antes de evocar el concepto. La cura es contrastar pools de frases
VARIADAS que encarnen cada polo. Aqui las genera el propio Llama, una vez,
y las cachea en un JSON EDITABLE (caa_stimuli.json).

  python -m steering.generate_stimuli            # genera todas las dimensiones
  python -m steering.generate_stimuli --only d090_divinidad d004_temperatura

Salida: caa_stimuli.json = { "d090_divinidad": {"pos":[...], "neg":[...]}, ... }
Puedes editar ese fichero a mano (anadir/quitar frases) antes de derivar.
"""
import json
import re
import argparse
import torch

from . import config
from .derive_vectors import _load_llm, _load_concepts


def _gen_sentences(tok, model, idea, name, k):
    """Pide k frases variadas que encarnen 'idea'. Devuelve lista de strings."""
    prompt = (
        f"Escribe exactamente {k} frases breves, variadas y naturales en espanol "
        f"que evoquen con fuerza esta idea:\n\n«{idea}»  (dimension: {name}).\n\n"
        "Una frase por linea. Sin numerar, sin vinetas, sin encabezados ni "
        "comillas. Varia el vocabulario, las escenas y el tono en cada frase."
    )
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt"
                                  ).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=k * 22, do_sample=True,
                             temperature=0.9, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    lines = []
    for ln in text.splitlines():
        ln = re.sub(r'^\s*[\d]+[\.\)\-]\s*', '', ln)     # quita "1. " "2) "
        ln = re.sub(r'^\s*[-*•]\s*', '', ln).strip().strip('"').strip()
        if len(ln) >= 8 and ln not in lines:
            lines.append(ln)
    return lines[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="solo estas dimensiones (por id)")
    ap.add_argument("--redo", action="store_true",
                    help="regenera aunque la dim ya tenga frases (por defecto se salta)")
    args = ap.parse_args()

    _, dim_names, _ = _load_concepts()
    master = json.loads(
        (config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in master}

    tok, model = _load_llm()
    print(f"modelo en VRAM: {model.get_memory_footprint()/1e9:.1f} GB "
          f"(4-bit deberia rondar ~6; si ves ~16, el quantizado NO esta aplicando)")

    # reanudar si ya habia algo
    store = {}
    if config.STIMULI_FILE.exists():
        store = json.loads(config.STIMULI_FILE.read_text(encoding="utf-8"))

    targets = args.only or dim_names
    for i, dim in enumerate(targets, 1):
        # reanudable: no repite trabajo ya hecho (salvo --redo)
        prev = store.get(dim, {})
        if not args.redo and prev.get("pos") and prev.get("neg"):
            print(f"[{i}/{len(targets)}] ya existe: {dim}")
            continue
        d = by_id.get(dim, {})
        sc = d.get("scale", {})
        name = d.get("name", dim)
        mn, mx = sc.get("min", ""), sc.get("max", "")
        if not mn or not mx:
            print(f"  AVISO sin polos: {dim}")
            continue
        print(f"[{i}/{len(targets)}] {dim}  ({mn}  <->  {mx})")
        pos = _gen_sentences(tok, model, mx, name, config.K_STIM)
        neg = _gen_sentences(tok, model, mn, name, config.K_STIM)
        store[dim] = {"pos": pos, "neg": neg}
        # guardado incremental (por si se corta)
        config.STIMULI_FILE.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()   # evita que el allocator crezca toda la noche

    print(f"OK estimulos -> {config.STIMULI_FILE}  ({len(store)} dims)")


if __name__ == "__main__":
    main()
