"""
blind_test.py — Mata el sesgo de confirmacion. Test CIEGO de steering.

Elige dims AL AZAR (o las que indiques), las inyecta aisladas sin decirte
cual es cual, te enseña los textos etiquetados A/B/C/... y la lista de
nombres de dimension implicados. Tu emparejas letra->dimension. Al pulsar
Enter, revela la respuesta y tu puntuacion.

Si aciertas por encima del azar, el efecto es real y no pareidolia.

  python -m steering.blind_test                       # 4 dims al azar, alpha 0.15
  python -m steering.blind_test --k 5 --alpha 0.15
  python -m steering.blind_test --dims 4 23 41 --seed 7
  python -m steering.blind_test --phrase "describe una habitacion"
"""
import argparse
import json
import random
import numpy as np

from . import config
from .steering_model import SteeredLlama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="nº de dims a sortear")
    ap.add_argument("--dims", type=int, nargs="+", help="dims concretas (en vez de azar)")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--val", type=float, default=0.95, help="valor al que se sube la dim")
    ap.add_argument("--phrase", default="hablame de un dia cualquiera")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dim_names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    n_dims = len(dim_names)

    dims = args.dims if args.dims else random.sample(range(n_dims), args.k)

    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    llm.alpha = args.alpha
    print(f"✅ Listo. capas={llm.layers} | alpha={args.alpha} | frase=«{args.phrase}»\n")

    # genera un texto por dim (aislada)
    items = []
    for di in dims:
        prof = np.full(n_dims, 0.5, np.float32)
        prof[di] = args.val
        llm.set_profile(prof)
        txt = llm.generate(args.phrase, max_new_tokens=args.tokens, temperature=0.0)
        items.append((di, txt))
    llm.clear()

    # baraja el orden de presentacion (A, B, C...)
    random.shuffle(items)
    letters = [chr(ord("A") + i) for i in range(len(items))]

    print("=" * 60)
    print("  TEST CIEGO — ¿que dimension moví en cada texto?")
    print("=" * 60)
    for L, (_, txt) in zip(letters, items):
        print(f"\n[{L}] {'-'*52}\n{txt}")

    # candidatos: nombres ORDENADOS alfabeticamente (el orden no delata nada)
    candidatos = sorted(dim_names[di] for di, _ in items)
    print(f"\n{'='*60}\n  DIMENSIONES IMPLICADAS (desordenadas respecto a A/B/C):")
    for name in candidatos:
        print(f"   · {name}")
    print("=" * 60)
    print("\nApunta tu apuesta letra→dimension. Cuando la tengas:")
    input(">> Pulsa Enter para revelar... ")

    print(f"\n{'='*60}\n  RESPUESTA:")
    for L, (di, _) in zip(letters, items):
        print(f"   [{L}] -> [{di:03d}] {dim_names[di]}")
    print("=" * 60)
    print(f"\nAzar puro = 1/{len(items)} de acertar cada una. "
          f"Si superas eso de forma clara, el steering es REAL.")


if __name__ == "__main__":
    main()
