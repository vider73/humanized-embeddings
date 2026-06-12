"""
verify_mechanics.py — Prueba la ALGEBRA del steering sin necesidad del LLM.

Replica en numpy la logica de derive_vectors + steering_model con un
"cerebro" sintetico pequeño (H y capas de juguete) y verifica:
  1. shapes de la matriz de control vectors
  2. shapes del steering por capa y de la inyeccion en el residual
  3. invariantes: perfil plano a 0.5 => steering nulo (neutral exacto)
  4. coherencia: maximizar la dim i produce steering alineado con cv_i

Esto NO valida los forward hooks de PyTorch (eso corre en tu maquina),
pero garantiza que las matematicas y las dimensiones son correctas.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- cerebro de juguete ----------------------------------------------------
H = 64                      # hidden dim de juguete (en real: 4096)
N_DIMS = 104                # dims reales
LAYERS = [3, 4, 5]          # capas objetivo de juguete
K = 20                      # conceptos por polo
N_CONCEPTS = 400

# Y sintetico (N, 104), valores 0..1
Y = rng.random((N_CONCEPTS, N_DIMS)).astype(np.float32)

# Activaciones sinteticas por concepto y capa: simulamos que cada dimension
# tiene una direccion latente en el espacio H, y la activacion de un concepto
# es la suma ponderada por sus valores Y + ruido.
true_dirs = {L: rng.standard_normal((N_DIMS, H)).astype(np.float32) for L in LAYERS}

def fake_activation(concept_idx, L):
    base = Y[concept_idx] @ true_dirs[L]            # (H,)
    return base + 0.05 * rng.standard_normal(H)

# --- 1. derivacion (diff de medias, normalizada) ---------------------------
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

# --- 2. combinacion perfil -> steering por capa ----------------------------
def steering_per_layer(profile):
    p = (profile - 0.5).astype(np.float32)
    return {L: p @ cv[li] for li, L in enumerate(LAYERS)}   # cada uno (H,)

# invariante: perfil plano 0.5 => steering exactamente cero
flat = np.full(N_DIMS, 0.5, dtype=np.float32)
s_flat = steering_per_layer(flat)
for L in LAYERS:
    assert s_flat[L].shape == (H,)
    assert np.allclose(s_flat[L], 0.0, atol=1e-6)
print("[3] perfil plano (0.5) => steering nulo => DIRIGIDA == NEUTRAL OK")

# --- 3. inyeccion en el residual (shape preservada) ------------------------
ALPHA = 8.0
batch, seq = 2, 7
h_res = rng.standard_normal((batch, seq, H)).astype(np.float32)
prof = flat.copy(); prof[23] = 0.98                 # subir d023 (consciencia)
s = steering_per_layer(prof)
h_new = h_res + ALPHA * s[LAYERS[0]]                 # broadcast (H,) sobre (b,s,H)
assert h_new.shape == h_res.shape == (batch, seq, H)
print(f"[4] inyeccion residual preserva shape OK: {h_new.shape}")

# --- 4. coherencia: subir dim i alinea el steering con cv_i ----------------
i = 23
s_dim = steering_per_layer(prof)[LAYERS[0]]
cos = float(s_dim @ cv[0, i] / (np.linalg.norm(s_dim) * np.linalg.norm(cv[0, i])))
print(f"[5] coseno(steering, cv_dim23) = {cos:.3f}  (debe ser claramente >0)")
assert cos > 0.3

# energia del cambio escala con alpha (sanity de magnitud)
mag = ALPHA * np.linalg.norm(s_dim) / np.linalg.norm(h_res.mean((0, 1)) + 1e-9)
print(f"[6] |steering*alpha| relativo a |h| ~ {mag:.2f}  (regula con ALPHA)")

print("\n✅ Algebra y shapes del steering: CORRECTOS.")
print("   (Los forward hooks de PyTorch se validan al correr en tu 4090.)")
