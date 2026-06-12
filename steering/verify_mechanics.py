"""
verify_mechanics.py — Tests the steering ALGEBRA without needing the LLM.

Replicates in numpy the logic of derive_vectors + steering_model with a
small synthetic "brain" (toy H and layers) and verifies:
  1. shapes of the control vectors matrix
  2. shapes of the per-layer steering and of the injection into the residual
  3. invariants: flat profile at 0.5 => null steering (exact neutral)
  4. coherence: maximizing dim i produces steering aligned with cv_i

This does NOT validate the PyTorch forward hooks (that runs on your machine),
but it guarantees that the math and the dimensions are correct.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- toy brain ---------------------------------------------------------------
H = 64                      # toy hidden dim (real: 4096)
N_DIMS = 104                # real dims
LAYERS = [3, 4, 5]          # toy target layers
K = 20                      # concepts per pole
N_CONCEPTS = 400

# synthetic Y (N, 104), values 0..1
Y = rng.random((N_CONCEPTS, N_DIMS)).astype(np.float32)

# Synthetic activations per concept and layer: we simulate that each dimension
# has a latent direction in the H space, and a concept's activation
# is the sum weighted by its Y values + noise.
true_dirs = {L: rng.standard_normal((N_DIMS, H)).astype(np.float32) for L in LAYERS}

def fake_activation(concept_idx, L):
    base = Y[concept_idx] @ true_dirs[L]            # (H,)
    return base + 0.05 * rng.standard_normal(H)

# --- 1. derivation (diff of means, normalized) ------------------------------
cv = np.zeros((len(LAYERS), N_DIMS, H), dtype=np.float32)
for di in range(N_DIMS):
    order = np.argsort(Y[:, di])
    low_idx, high_idx = order[:K], order[-K:]
    for li, L in enumerate(LAYERS):
        hi = np.stack([fake_activation(i, L) for i in high_idx]).mean(0)
        lo = np.stack([fake_activation(i, L) for i in low_idx]).mean(0)
        v = hi - lo
        n = np.linalg.norm(v)
        cv[li, di] = v / n if n > 1e-8 else v

assert cv.shape == (len(LAYERS), N_DIMS, H), cv.shape
print(f"[1] control vectors shape OK: {cv.shape}")
norms = np.linalg.norm(cv, axis=2)
assert np.allclose(norms, 1.0, atol=1e-4), norms.max()
print(f"[2] todos los control vectors a norma unidad OK (max dev {abs(norms-1).max():.2e})")

# --- 2. profile -> per-layer steering combination ---------------------------
def steering_per_layer(profile):
    p = (profile - 0.5).astype(np.float32)
    return {L: p @ cv[li] for li, L in enumerate(LAYERS)}   # each one (H,)

# invariant: flat 0.5 profile => exactly zero steering
flat = np.full(N_DIMS, 0.5, dtype=np.float32)
s_flat = steering_per_layer(flat)
for L in LAYERS:
    assert s_flat[L].shape == (H,)
    assert np.allclose(s_flat[L], 0.0, atol=1e-6)
print("[3] perfil plano (0.5) => steering nulo => DIRIGIDA == NEUTRAL OK")

# --- 3. injection into the residual (shape preserved) -----------------------
ALPHA = 8.0
batch, seq = 2, 7
h_res = rng.standard_normal((batch, seq, H)).astype(np.float32)
prof = flat.copy(); prof[23] = 0.98                 # raise d023 (consciencia)
s = steering_per_layer(prof)
h_new = h_res + ALPHA * s[LAYERS[0]]                 # broadcast (H,) over (b,s,H)
assert h_new.shape == h_res.shape == (batch, seq, H)
print(f"[4] inyeccion residual preserva shape OK: {h_new.shape}")

# --- 4. coherence: raising dim i aligns the steering with cv_i --------------
i = 23
s_dim = steering_per_layer(prof)[LAYERS[0]]
cos = float(s_dim @ cv[0, i] / (np.linalg.norm(s_dim) * np.linalg.norm(cv[0, i])))
print(f"[5] coseno(steering, cv_dim23) = {cos:.3f}  (debe ser claramente >0)")
assert cos > 0.3

# energy of the change scales with alpha (magnitude sanity check)
mag = ALPHA * np.linalg.norm(s_dim) / np.linalg.norm(h_res.mean((0, 1)) + 1e-9)
print(f"[6] |steering*alpha| relativo a |h| ~ {mag:.2f}  (regula con ALPHA)")

print("\n✅ Algebra y shapes del steering: CORRECTOS.")
print("   (Los forward hooks de PyTorch se validan al correr en tu 4090.)")
