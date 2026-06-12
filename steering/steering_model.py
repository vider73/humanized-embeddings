"""
steering_model.py — El motor. Llama-3.1-8B con inyeccion en el residual stream.

Tu perfil de 104 dims se convierte, por capa, en UN vector de 4096d:

    steering(capa) = sum_i (perfil_i - 0.5) * control_vector(i, capa)

y se suma a las activaciones de esa capa durante el forward (forward hook):

    h <- h + alpha * steering(capa)

El (perfil_i - 0.5) centra: por encima de 0.5 empuja hacia "mas d_i",
por debajo hacia "menos d_i". Con el perfil a 0.5 plano, steering = 0 (neutral).
"""
import json
import numpy as np
import torch

from . import config


class SteeredLlama:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(config.LLM_NAME)
        kwargs = dict(torch_dtype=torch.float16, device_map="auto")
        if config.LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, **kwargs)
        self.model.eval()

        cv = np.load(config.CONTROL_VECTORS_FILE)          # (n_capas, 104, H)
        self.meta = json.loads(config.CONTROL_META_FILE.read_text(encoding="utf-8"))
        derived = list(self.meta["target_layers"])         # capas YA derivadas
        # Cargamos TODAS las capas derivadas (~7MB) y registramos hook en todas:
        # asi el voicing puede tocar cualquier capa sin recargar. self.layers
        # es solo el conjunto POR DEFECTO donde inyecta set_profile.
        want = getattr(config, "INJECT_LAYERS", None)
        self.layers = [L for L in (want or derived) if L in derived] or derived
        dev = self.model.device
        self.cv = {L: torch.tensor(cv[derived.index(L)], dtype=torch.float16, device=dev)
                   for L in derived}                       # cada uno (104, H)

        self.alpha = config.ALPHA
        self.mode = getattr(config, "STEER_MODE", "add")   # "add" | "clamp"
        self.kp = getattr(config, "KP", 1.0)
        self._steer = {L: None for L in derived}           # nota activa por capa
        self._handles = []
        self._register_hooks()

    # --- hooks --------------------------------------------------------------
    def _register_hooks(self):
        def make_hook(L):
            def hook(module, inp, out):
                vec = self._steer[L]
                if vec is None:
                    return out
                # transformers nuevo: out es un tensor; antiguo: tupla con out[0]
                is_tuple = isinstance(out, tuple)
                hs = out[0] if is_tuple else out         # (batch, seq, H)
                if isinstance(vec, dict):
                    # MULTI-CLAMP: fija la proyeccion sobre CADA dim activa a
                    # su setpoint. delta = Ginv @ (targets - proj) reparte el
                    # empuje exacto aunque las direcciones se solapen.
                    # La nota puede traer su PROPIO alpha (voicing por capa).
                    V, Ginv, w = vec["V"], vec["Ginv"], vec["w"]
                    a = vec["alpha"] if vec.get("alpha") is not None else self.alpha
                    h32 = hs.float()
                    norm = h32.norm(dim=-1, keepdim=True)           # (b,s,1)
                    proj = h32 @ V.T                                # (b,s,k)
                    targets = a * norm * w                          # (b,s,k)
                    delta = (targets - proj) @ Ginv                 # (b,s,k)
                    hs = hs + (self.kp * (delta @ V)).to(hs.dtype)
                else:
                    v = vec.to(hs.dtype)
                    norm = hs.float().norm(dim=-1, keepdim=True).to(hs.dtype)
                    hs = hs + self.alpha * norm * v     # empuje clasico fijo
                return ((hs,) + tuple(out[1:])) if is_tuple else hs
            return hook

        for L in self.cv:                                  # todas las derivadas
            h = self.model.model.layers[L].register_forward_hook(make_hook(L))
            self._handles.append(h)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # --- control ------------------------------------------------------------
    def set_profile(self, profile: np.ndarray):
        """profile: (104,) en ~0..1. Solo activa las capas de self.layers.

        modo "add"  : direccion UNITARIA mezclada (como siempre).
        modo "clamp": MULTI-CLAMP — un controlador P por dim activa, cada una
          con su setpoint (alpha*|h| escalado por su slider). Resuelto exacto
          con la inversa de Gram: las dims que comparten componente no se
          pisan ni se diluyen. Un acorde = varias cuerdas, no una mezclada.
        """
        for L in self._steer:
            self._steer[L] = None
        w = (np.asarray(profile, np.float32) - 0.5) / 0.5     # sliders en [-1,1]
        active = np.where(np.abs(w) > 1e-3)[0]
        if len(active) == 0:
            return                       # perfil plano => DIRIGIDA == NEUTRAL
        for L in self.layers:
            cv = self.cv[L]
            if self.mode == "clamp":
                V = cv[active].float()                        # (k, H) filas unit
                G = V @ V.T                                   # (k, k) solapes
                eye = torch.eye(len(active), device=V.device)
                Ginv = torch.linalg.inv(G + 1e-4 * eye)
                self._steer[L] = {
                    "V": V, "Ginv": Ginv,
                    "w": torch.tensor(w[active], device=V.device),
                }
            else:
                p = torch.tensor((profile - 0.5).astype(np.float16),
                                 device=cv.device)
                v = p.to(cv.dtype) @ cv                       # (H,) mezcla
                n = torch.linalg.norm(v.float())
                self._steer[L] = (v / n.to(v.dtype)) if n > 1e-4 else None

    def set_voices(self, voices: dict):
        """VOICING: cada capa toca su propia nota con su propia presion.

        voices = {capa: perfil} o {capa: (perfil, alpha)}. Perfil (104,) en
        0..1 con 0.5 = silencio; alpha None usa self.alpha. Solo multi-clamp.
        Ejemplo: d090 fuerte en la 14, d023 suave en la 16:
            voices = {14: (p90, 0.35), 16: (p23, 0.20)}
        """
        for L in self._steer:
            self._steer[L] = None
        for L, spec in voices.items():
            if L not in self.cv:
                raise ValueError(f"capa {L} no derivada (tengo {list(self.cv)})")
            profile, a = spec if isinstance(spec, tuple) else (spec, None)
            w = (np.asarray(profile, np.float32) - 0.5) / 0.5
            active = np.where(np.abs(w) > 1e-3)[0]
            if len(active) == 0:
                continue
            V = self.cv[L][active].float()
            G = V @ V.T
            eye = torch.eye(len(active), device=V.device)
            self._steer[L] = {
                "V": V, "Ginv": torch.linalg.inv(G + 1e-4 * eye),
                "w": torch.tensor(w[active], device=V.device),
                "alpha": a,
            }

    def clear(self):
        for L in self._steer:
            self._steer[L] = None

    # --- generacion ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, user_msg: str, max_new_tokens: int = 200,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        messages = [{"role": "user", "content": user_msg}]
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        attn = torch.ones_like(ids)
        out = self.model.generate(
            ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5), top_p=top_p,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def generate_pair(self, user_msg: str, profile: np.ndarray, **kw):
        """Devuelve (neutral, dirigido) para comparar el efecto del steering."""
        self.clear()
        neutral = self.generate(user_msg, **kw)
        self.set_profile(profile)
        steered = self.generate(user_msg, **kw)
        self.clear()
        return neutral, steered
