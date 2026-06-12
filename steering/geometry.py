"""
geometry.py — Measures the health of a control vectors file.

The key number is the EFFECTIVE RANK: how many truly independent
directions there are among the 104 nominal ones. Rank ~1-2 = collapse (all the
same axis). Rank ~15-18 = healthy (close to the rank of the human table).

  python -m steering.geometry                       # uses config.CONTROL_VECTORS_FILE
  python -m steering.geometry vectors/control_vectors.npy
  python -m steering.geometry vectors/control_vectors_white.npy --layer 15
"""
import json
import argparse
from pathlib import Path
import numpy as np

from . import config


def effective_rank(M):
    """M: (n_dims, H) unit-norm. Returns participation ratio and entropy-rank."""
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    pr = (s ** 2).sum() ** 2 / (s ** 4).sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return float(pr), erank


def report(path, layer):
    p = Path(path)
    cv = np.load(p)                                  # (layers, dims, H)
    meta_p = p.with_name(p.name.replace("control_vectors", "control_meta")
                         .replace(".npy", ".json"))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    names = [n.split("_", 1)[-1] for n in meta["dim_names"]]
    layers = meta["target_layers"]
    li = layers.index(layer) if layer in layers else len(layers) // 2
    M = cv[li].astype(np.float64)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

    pr, erank = effective_rank(M)
    C = M @ M.T
    off = C[~np.eye(M.shape[0], dtype=bool)]

    print(f"=== {path}  (capa {layers[li]}) ===")
    print(f"rango efectivo  : PR={pr:.1f} | entropia={erank:.1f}   (de {M.shape[0]} nominales)")
    print(f"ortogonalidad   : |cos|medio={np.abs(off).mean():.3f}  "
          f"frac|cos|>0.5={(np.abs(off) > 0.5).mean()*100:.0f}%")

    iu = np.triu_indices(M.shape[0], 1)
    cos = C[iu]
    o = np.argsort(cos)
    print("pares mas alineados (redundancia):")
    for k in o[::-1][:4]:
        a, b = iu[0][k], iu[1][k]
        print(f"   {cos[k]:+.2f}  {names[a]} ~ {names[b]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(config.CONTROL_VECTORS_FILE))
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    report(args.path, args.layer)


if __name__ == "__main__":
    main()
