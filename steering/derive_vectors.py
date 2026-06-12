"""
derive_vectors.py — De tus 104 dimensiones a vectores de control en el
espacio nativo de Llama (4096d).  ESTO es el puente real.

Idea (representation engineering, diferencia de medias):
  Para cada dimension d_i:
    - conceptos del POLO ALTO  = los K que mas puntuan en d_i
    - conceptos del POLO BAJO  = los K que menos puntuan en d_i
    - se pasan por Llama, se captura el residual stream en cada capa objetivo
    - control_vector(d_i, capa) = mean(activaciones_alto) - mean(activaciones_bajo)
  El vector resultante apunta, dentro del cerebro de Llama, hacia "mas d_i".

Salida: control_vectors.npy con shape (n_capas_objetivo, 104, 4096), cada
vector normalizado a norma unidad.  La escala se aplica luego con ALPHA.

Requiere GPU + el modelo HF.  Corre en tu maquina, no en el sandbox.
"""
import json
import numpy as np
import torch
from tqdm import tqdm

from . import config


def _load_concepts():
    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    concepts = meta["concepts"]
    dim_names = meta["dimension_names"]
    Y = np.load(config.DATA_Y_FILE)              # (N, 104)
    assert Y.shape[0] == len(concepts)
    assert Y.shape[1] == len(dim_names)
    return concepts, dim_names, Y


def _load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.LLM_NAME)
    # OJO: output_hidden_states NO va aqui (global). Puesto en el modelo,
    # generate() calcula y devuelve las 33 capas en CADA token para tirarlas
    # (memoria y trabajo gratis). Se pide por-llamada en _reps, que es quien
    # lo necesita.
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if config.LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, **kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def _reps(texts, tok, model):
    """
    Representacion por texto en cada capa objetivo.
    Devuelve dict {capa: tensor (len(texts), HIDDEN_DIM)} en CPU float32.

    hidden_states del forward tiene N_LAYERS+1 entradas:
      hidden_states[0]   = embeddings
      hidden_states[L+1] = salida (residual) de la capa L
    """
    acc = {L: [] for L in config.TARGET_LAYERS}
    for text in texts:
        enc = tok(text, return_tensors="pt").to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple (1, seq, H)
        for L in config.TARGET_LAYERS:
            h = hs[L + 1][0]                        # (seq, H)
            if config.POOLING == "last":
                v = h[-1]
            else:
                v = h.mean(dim=0)                   # mean pooling sobre tokens
            acc[L].append(v.float().cpu())
    return {L: torch.stack(vs) for L, vs in acc.items()}


def _expand(concepts):
    """Envuelve cada concepto en las plantillas neutras (reduce ruido)."""
    out = []
    for c in concepts:
        out.extend(t.format(c=c) for t in config.TEMPLATES)
    return out


def _load_poles(dim_names):
    """Devuelve [(min_desc, max_desc)] por dimension, desde master_dimensions."""
    master = json.loads(
        (config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))
    by_id = {d["id"]: d.get("scale", {}) for d in master}
    poles = [(by_id.get(n, {}).get("min", ""), by_id.get(n, {}).get("max", ""))
             for n in dim_names]
    missing = [dim_names[i] for i, (a, b) in enumerate(poles) if not a or not b]
    if config.DERIVE_MODE == "caa" and missing:
        raise ValueError(f"Faltan polos para CAA en: {missing[:5]}...")
    return poles


_STIM_CACHE = None


def _stimuli():
    """Carga (una vez) las frases ricas por polo, si existen."""
    global _STIM_CACHE
    if _STIM_CACHE is None:
        _STIM_CACHE = (json.loads(config.STIMULI_FILE.read_text(encoding="utf-8"))
                       if config.STIMULI_FILE.exists() else {})
    return _STIM_CACHE


def _contrast_texts(di, dim_names, Y, concepts, poles):
    """(pos, neg) segun el modo de derivacion. pos = polo MAX, neg = polo MIN."""
    if config.DERIVE_MODE == "caa":
        # 1ª opcion: frases ricas generadas (direccion semantica, no lexica)
        stim = _stimuli().get(dim_names[di])
        if stim and stim.get("pos") and stim.get("neg"):
            return stim["pos"], stim["neg"]
        # fallback: moldes cortos sobre la etiqueta del polo
        mn, mx = poles[di]
        neg = [t.format(p=mn) for t in config.CAA_TEMPLATES]
        pos = [t.format(p=mx) for t in config.CAA_TEMPLATES]
    else:
        order = np.argsort(Y[:, di])
        neg = _expand([concepts[i] for i in order[: config.K_CONTRAST]])
        pos = _expand([concepts[i] for i in order[-config.K_CONTRAST:]])
    return pos, neg


def _ref_sigma(concepts, tok, model):
    """
    Estima la std por coordenada (4096) de las activaciones, por capa, sobre
    una muestra de referencia. Esta std mide cuanto 'grita' cada coordenada:
    las outlier tienen std enorme. Dividir por ella las pone a todas en pie
    de igualdad y evita que dominen la diferencia de medias (arregla el
    colapso de rango). Devuelve {L: sigma (H,)} en CPU float32.
    """
    rng = np.random.default_rng(0)
    n = min(config.N_REF, len(concepts))
    sample = [concepts[i] for i in rng.choice(len(concepts), n, replace=False)]
    reps = _reps(_expand(sample), tok, model)        # {L: (n*plantillas, H)}
    return {L: reps[L].std(dim=0) for L in reps}      # (H,) por capa


def derive():
    concepts, dim_names, Y = _load_concepts()
    tok, model = _load_llm()

    n_dims = len(dim_names)
    H = model.config.hidden_size               # se lee del modelo, no se hardcodea
    n_model_layers = model.config.num_hidden_layers
    layers = list(config.TARGET_LAYERS)
    bad = [L for L in layers if not (0 <= L < n_model_layers)]
    if bad:
        raise ValueError(f"TARGET_LAYERS fuera de rango {bad} (modelo tiene {n_model_layers} capas)")

    poles = _load_poles(dim_names)

    sigma = None
    if config.WHITEN:
        print(f"⏳ estimando std de referencia ({config.N_REF} conceptos)...")
        sigma = _ref_sigma(concepts, tok, model)     # {L: (H,)}

    print(f"⚙️  modo de derivacion: {config.DERIVE_MODE.upper()}")
    # (n_capas, n_dims, H)
    out = np.zeros((len(layers), n_dims, H), dtype=np.float32)

    for di in tqdm(range(n_dims), desc="dimensiones"):
        pos_txt, neg_txt = _contrast_texts(di, dim_names, Y, concepts, poles)
        rp = _reps(pos_txt, tok, model)
        rn = _reps(neg_txt, tok, model)

        for li, L in enumerate(layers):
            cv = rp[L].mean(0) - rn[L].mean(0)     # (H,) contraste pos - neg
            if sigma is not None:
                cv = cv / (sigma[L] + 1e-6)        # z-score por coordenada (blanqueo)
            n = torch.linalg.norm(cv)
            if n > 1e-8:
                cv = cv / n
            out[li, di] = cv.numpy()

    np.save(config.CONTROL_VECTORS_FILE, out)
    config.CONTROL_META_FILE.write_text(json.dumps({
        "shape": list(out.shape),
        "target_layers": layers,
        "dim_names": dim_names,
        "derive_mode": config.DERIVE_MODE,
        "k_contrast": config.K_CONTRAST,
        "pooling": config.POOLING,
        "normalized": "unit",
        "whiten": bool(config.WHITEN),
        "n_ref": config.N_REF if config.WHITEN else 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tag = f"{config.DERIVE_MODE.upper()}" + ("+blanqueo" if config.WHITEN else "")
    print(f"OK [{tag}] control_vectors {out.shape} -> {config.CONTROL_VECTORS_FILE}")


if __name__ == "__main__":
    derive()
