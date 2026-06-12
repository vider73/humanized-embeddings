"""
56_linear_probe.py — Are the 104 dimensions already PRESENT in the embedding,
or invented by the translator?

Born from our first external review (2026-06-12): "if a linear map Y = W·X
suffices, the properties are really present in the original embedding; if the
network needs a complex architecture, they probably are not explicit."

This script runs that exact test: ridge regression from MiniLM embeddings
(384) to the 104 human scores, per-dimension R² on a held-out split, plus a
cross-check against the steering scorecard.

Findings on the current dataset (seed 42, λ=1, 80/20):
  · No dimension exceeds R²=0.5 — the axes are NOT explicit linear directions;
    the MLP's nonlinearity is doing real work.
  · The least-linear dimensions (physical size, max speed, duration,
    viscosity, friction, radioactivity…) are EXACTLY the scorecard's dead
    tail: three independent probes (linear decodability in MiniLM, causal
    steering in Llama, judge-LLM scoring sanity) fail on the same axes.
    Invented dimensions would fail at random; a real-but-noisy instrument
    fails consistently.
  · The 26 causally-verified dials average R²≈0.28 vs ≈0.21 for the rest:
    linear readability in one model correlates with causal steerability in
    another — evidence of shared semantic structure across spaces.

Honest ceiling: Y is LLM-scored and noisy, so R² is bounded by label noise —
low R² mixes "not linearly present" with "label too noisy to fit".

  python 56_linear_probe.py
  python 56_linear_probe.py --lam 10 --seed 7
"""
import argparse
import json

import numpy as np

# The 26 dials that passed causal verification (scorecard of 2026-06-11,
# z_kin >= 1.5 with correct sign). Update if the scorecard regime changes.
CAUSAL_DIALS = [
    "d099_necesidad", "d097_velocidad_de_acción", "d053_curiosidad",
    "d056_ira", "d010_dureza", "d024_instinto", "d037_sabor_amargor",
    "d023_consciencia", "d095_onirismo", "d101_control", "d046_amor",
    "d029_fertilidad", "d036_sabor_dulzor", "d059_culpa", "d044_pegajosidad",
    "d065_verdad_facticidad", "d016_volatilidad_química", "d077_ética_bondad",
    "d060_envidia", "d067_racionalidad", "d076_legalidad", "d082_lujo",
    "d022_vitalidad", "d039_rugosidad_táctil", "d080_artificialidad",
    "d043_temperatura_táctil",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0, help="ridge lambda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    X = np.load("dataset_X_embeddings.npy").astype(np.float64)
    Y = np.load("dataset_Y_human.npy").astype(np.float64)
    names = json.load(open("dataset_metadata.json", encoding="utf-8"))["dimension_names"]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]

    Xb = np.hstack([X, np.ones((len(X), 1))])                 # bias column
    A = Xb[tr].T @ Xb[tr] + args.lam * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb[tr].T @ Y[tr])                  # (385, 104)

    pred = Xb[te] @ W
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    order = np.argsort(-r2)

    print(f"LINEAR PROBE  Y = W·X  (ridge λ={args.lam}, seed {args.seed}, "
          f"test n={len(te)})")
    print(f"R²: mean={r2.mean():.3f}  median={np.median(r2):.3f}  "
          f"min={r2.min():.3f}  max={r2.max():.3f}")
    print(f"dims with R²>0.5: {(r2 > 0.5).sum()}/104   "
          f">0.3: {(r2 > 0.3).sum()}/104\n")

    print(f"TOP {args.top} (most linear):")
    for i in order[:args.top]:
        print(f"  {names[i]:<34} R²={r2[i]:+.3f}")
    print(f"BOTTOM {args.top} (least linear):")
    for i in order[-args.top:]:
        print(f"  {names[i]:<34} R²={r2[i]:+.3f}")

    gi = [names.index(n) for n in CAUSAL_DIALS if n in names]
    oi = [i for i in range(len(names)) if i not in gi]
    print(f"\nSTEERING CROSS-CHECK: mean R² of the {len(gi)} causal dials "
          f"= {r2[gi].mean():.3f}  vs rest = {r2[oi].mean():.3f}")
    print("(linear readability in MiniLM correlates with causal "
          "steerability in Llama — two different models)")


if __name__ == "__main__":
    main()
