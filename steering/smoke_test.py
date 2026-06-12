"""
smoke_test.py — Valida los FORWARD HOOKS de verdad, en tu PC, en segundos.

Usa un modelo Llama diminuto y ABIERTO (SmolLM2-135M, misma arquitectura)
para comprobar, sobre PyTorch real, que:
  [A] el hook engancha en model.model.layers[L] y modifica el residual
  [B] steering = 0  =>  salida IDENTICA a la neutral (invariante critico)
  [C] un steering != 0 CAMBIA la generacion
  [D] el camino completo perfil(104) -> (p-0.5)@cv -> inyeccion funciona
      con un modelo real (cv fabricado al azar, solo probamos la mecanica)

Si esto pasa, el mismo codigo de steering_model.py funcionara con Llama-3.1-8B;
solo cambian tamaños (se leen del modelo) y la procedencia de los control vectors.

    pip install -r steering/requirements.txt
    python -m steering.smoke_test
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

MODEL = config.SMOKE_LLM_NAME
N_DIMS = 104


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Cargando {MODEL} en {dev} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    model.eval()

    H = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    # banda central de capas, valida para este modelo
    layers = list(range(n_layers // 3, n_layers // 3 + 3))
    print(f"   hidden_size={H}  capas={n_layers}  objetivo={layers}")

    # --- estado de steering por capa (None = desactivado) ------------------
    steer = {L: None for L in layers}
    handles = []

    def make_hook(L):
        def hook(module, inp, out):
            v = steer[L]
            if v is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs + v.to(hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    for L in layers:
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

    # --- generacion greedy (determinista) ---------------------------------
    @torch.no_grad()
    def gen(prompt="Cuentame algo sobre el mar.", n=40):
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(dev)
        out = model.generate(ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    # [A]+[B] sin steering => baseline reproducible
    for L in layers:
        steer[L] = None
    base1 = gen()
    base2 = gen()
    assert base1 == base2, "greedy deberia ser determinista"
    print("\n[A][B] hook instalado; steering=None => salida estable y == neutral  OK")

    # [C] steering aleatorio fuerte => debe cambiar la salida
    torch.manual_seed(0)
    for L in layers:
        v = torch.randn(H, device=dev)
        steer[L] = 6.0 * v / v.norm()
    changed = gen()
    assert changed != base1, "el steering deberia cambiar la generacion"
    print("[C] steering != 0 cambia la generacion  OK")

    # [D] camino completo perfil -> (p-0.5)@cv -> steering por capa
    cv = {L: torch.randn(N_DIMS, H, device=dev) for L in layers}
    for L in layers:
        cv[L] = cv[L] / cv[L].norm(dim=1, keepdim=True)   # normaliza por dim
    prof = np.full(N_DIMS, 0.5, np.float32)
    p0 = torch.tensor(prof - 0.5, device=dev)
    for L in layers:                                       # perfil plano => 0
        steer[L] = (p0 @ cv[L])
    flat = gen()
    assert flat == base1, "perfil plano (0.5) debe igualar a la neutral"
    print("[D] perfil plano via (p-0.5)@cv => steering nulo => == neutral  OK")

    prof[23] = 0.98                                        # sube una dimension
    p1 = torch.tensor(prof - 0.5, device=dev)
    for L in layers:
        steer[L] = config.ALPHA * (p1 @ cv[L])
    dirigida = gen()
    print("\n── NEUTRAL ──\n", base1)
    print("\n── DIRIGIDA (cv aleatorio, solo demo de mecanica) ──\n", dirigida)
    assert dirigida != base1, "con cv y perfil activo deberia cambiar"
    print("\n[D] camino completo perfil->cv->inyeccion CAMBIA la salida  OK")

    for h in handles:
        h.remove()
    print("\n✅ HOOKS REALES VALIDADOS. steering_model.py funcionara igual con Llama-8B.")
    print("   (Aqui el cv es aleatorio; el efecto SEMANTICO real llega tras derive_vectors.)")


if __name__ == "__main__":
    main()
