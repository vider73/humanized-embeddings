"""
steering_model.py — The engine. Llama-3.1-8B with injection into the residual stream.

Your 104-dim profile is converted, per layer, into ONE 4096d vector:

    steering(layer) = sum_i (profile_i - 0.5) * control_vector(i, layer)

and added to that layer's activations during the forward pass (forward hook):

    h <- h + alpha * steering(layer)

The (profile_i - 0.5) centers it: above 0.5 it pushes toward "more d_i",
below toward "less d_i". With a flat 0.5 profile, steering = 0 (neutral).
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

        cv = np.load(config.CONTROL_VECTORS_FILE)          # (n_layers, 104, H)
        self.meta = json.loads(config.CONTROL_META_FILE.read_text(encoding="utf-8"))
        derived = list(self.meta["target_layers"])         # layers ALREADY derived
        # We load ALL derived layers (~7MB) and register a hook on every one:
        # that way voicing can play any layer without reloading. self.layers
        # is only the DEFAULT set where set_profile injects.
        want = getattr(config, "INJECT_LAYERS", None)
        self.layers = [L for L in (want or derived) if L in derived] or derived
        dev = self.model.device
        self.cv = {L: torch.tensor(cv[derived.index(L)], dtype=torch.float16, device=dev)
                   for L in derived}                       # each one (104, H)

        self.alpha = config.ALPHA
        self.mode = getattr(config, "STEER_MODE", "add")   # "add" | "clamp"
        self.kp = getattr(config, "KP", 1.0)
        self._steer = {L: None for L in derived}           # active note per layer
        self._handles = []
        self._register_hooks()

    # --- hooks --------------------------------------------------------------
    def _register_hooks(self):
        def make_hook(L):
            def hook(module, inp, out):
                vec = self._steer[L]
                if vec is None:
                    return out
                # new transformers: out is a tensor; old: tuple with out[0]
                is_tuple = isinstance(out, tuple)
                hs = out[0] if is_tuple else out         # (batch, seq, H)
                if isinstance(vec, dict):
                    # MULTI-CLAMP: pins the projection on EACH active dim to
                    # its setpoint. delta = Ginv @ (targets - proj) distributes
                    # the exact push even when the directions overlap.
                    # The note may carry its OWN alpha (per-layer voicing).
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
                    hs = hs + self.alpha * norm * v     # classic fixed push
                return ((hs,) + tuple(out[1:])) if is_tuple else hs
            return hook

        for L in self.cv:                                  # all derived layers
            h = self.model.model.layers[L].register_forward_hook(make_hook(L))
            self._handles.append(h)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # --- control ------------------------------------------------------------
    def set_profile(self, profile: np.ndarray):
        """profile: (104,) in ~0..1. Only activates the layers in self.layers.

        "add" mode  : blended UNIT direction (as always).
        "clamp" mode: MULTI-CLAMP — one P controller per active dim, each
          with its setpoint (alpha*|h| scaled by its slider). Solved exactly
          with the Gram inverse: dims that share a component neither stomp
          on each other nor get diluted. A chord = several strings, not one blended.
        """
        for L in self._steer:
            self._steer[L] = None
        w = (np.asarray(profile, np.float32) - 0.5) / 0.5     # sliders in [-1,1]
        active = np.where(np.abs(w) > 1e-3)[0]
        if len(active) == 0:
            return                       # flat profile => STEERED == NEUTRAL
        for L in self.layers:
            cv = self.cv[L]
            if self.mode == "clamp":
                V = cv[active].float()                        # (k, H) unit rows
                G = V @ V.T                                   # (k, k) overlaps
                eye = torch.eye(len(active), device=V.device)
                Ginv = torch.linalg.inv(G + 1e-4 * eye)
                self._steer[L] = {
                    "V": V, "Ginv": Ginv,
                    "w": torch.tensor(w[active], device=V.device),
                }
            else:
                p = torch.tensor((profile - 0.5).astype(np.float16),
                                 device=cv.device)
                v = p.to(cv.dtype) @ cv                       # (H,) blend
                n = torch.linalg.norm(v.float())
                self._steer[L] = (v / n.to(v.dtype)) if n > 1e-4 else None

    def set_voices(self, voices: dict):
        """VOICING: each layer plays its own note with its own pressure.

        voices = {layer: profile} or {layer: (profile, alpha)}. Profile (104,) in
        0..1 with 0.5 = silence; alpha None uses self.alpha. Multi-clamp only.
        Example: d090 strong on layer 14, d023 soft on layer 16:
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

    # --- generation ----------------------------------------------------------
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
        """Returns (neutral, steered) to compare the effect of the steering."""
        self.clear()
        neutral = self.generate(user_msg, **kw)
        self.set_profile(profile)
        steered = self.generate(user_msg, **kw)
        self.clear()
        return neutral, steered
