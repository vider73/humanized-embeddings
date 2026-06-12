"""
derive_vectors.py — From your 104 dimensions to control vectors in
Llama's native space (4096d).  THIS is the real bridge.

Idea (representation engineering, difference of means):
  For each dimension d_i:
    - HIGH-POLE concepts = the K that score highest on d_i
    - LOW-POLE concepts  = the K that score lowest on d_i
    - they are run through Llama, capturing the residual stream at each target layer
    - control_vector(d_i, layer) = mean(high_activations) - mean(low_activations)
  The resulting vector points, inside Llama's brain, toward "more d_i".

Output: control_vectors.npy with shape (n_target_layers, 104, 4096), each
vector normalized to unit norm.  Scale is applied later with ALPHA.

Requires GPU + the HF model.  Runs on your machine, not in the sandbox.
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
    # CAREFUL: output_hidden_states does NOT go here (globally). Set on the model,
    # generate() computes and returns all 33 layers on EVERY token only to throw
    # them away (free memory and work wasted). It is requested per-call in _reps,
    # which is the one that needs it.
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
    Per-text representation at each target layer.
    Returns dict {layer: tensor (len(texts), HIDDEN_DIM)} on CPU float32.

    hidden_states from the forward pass has N_LAYERS+1 entries:
      hidden_states[0]   = embeddings
      hidden_states[L+1] = output (residual) of layer L
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
                v = h.mean(dim=0)                   # mean pooling over tokens
            acc[L].append(v.float().cpu())
    return {L: torch.stack(vs) for L, vs in acc.items()}


def _expand(concepts):
    """Wraps each concept in the neutral templates (reduces noise)."""
    out = []
    for c in concepts:
        out.extend(t.format(c=c) for t in config.TEMPLATES)
    return out


def _load_poles(dim_names):
    """Returns [(min_desc, max_desc)] per dimension, from master_dimensions."""
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
    """Loads (once) the rich sentences per pole, if they exist."""
    global _STIM_CACHE
    if _STIM_CACHE is None:
        _STIM_CACHE = (json.loads(config.STIMULI_FILE.read_text(encoding="utf-8"))
                       if config.STIMULI_FILE.exists() else {})
    return _STIM_CACHE


def _contrast_texts(di, dim_names, Y, concepts, poles):
    """(pos, neg) according to the derivation mode. pos = MAX pole, neg = MIN pole."""
    if config.DERIVE_MODE == "caa":
        # 1st choice: generated rich sentences (semantic direction, not lexical)
        stim = _stimuli().get(dim_names[di])
        if stim and stim.get("pos") and stim.get("neg"):
            return stim["pos"], stim["neg"]
        # fallback: short molds over the pole's label
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
    Estimates the per-coordinate std (4096) of the activations, per layer, over
    a reference sample. This std measures how loudly each coordinate 'shouts':
    the outliers have a huge std. Dividing by it puts them all on equal
    footing and keeps them from dominating the difference of means (fixes the
    rank collapse). Returns {L: sigma (H,)} on CPU float32.
    """
    rng = np.random.default_rng(0)
    n = min(config.N_REF, len(concepts))
    sample = [concepts[i] for i in rng.choice(len(concepts), n, replace=False)]
    reps = _reps(_expand(sample), tok, model)        # {L: (n*templates, H)}
    return {L: reps[L].std(dim=0) for L in reps}      # (H,) per layer


def derive():
    concepts, dim_names, Y = _load_concepts()
    tok, model = _load_llm()

    n_dims = len(dim_names)
    H = model.config.hidden_size               # read from the model, not hardcoded
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
    # (n_layers, n_dims, H)
    out = np.zeros((len(layers), n_dims, H), dtype=np.float32)

    for di in tqdm(range(n_dims), desc="dimensiones"):
        pos_txt, neg_txt = _contrast_texts(di, dim_names, Y, concepts, poles)
        rp = _reps(pos_txt, tok, model)
        rn = _reps(neg_txt, tok, model)

        for li, L in enumerate(layers):
            cv = rp[L].mean(0) - rn[L].mean(0)     # (H,) pos - neg contrast
            if sigma is not None:
                cv = cv / (sigma[L] + 1e-6)        # per-coordinate z-score (whitening)
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
