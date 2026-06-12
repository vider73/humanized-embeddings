"""
demo.py — El juguete honesto. Escribes algo, ves dos respuestas:
  ⚪ neutral  (sin steering)
  🔴 dirigida (el LLM pensando inclinado hacia tu perfil de 104 dims)

Uso:
  python -m steering.demo                 # perfil = el de tu propio texto
  python -m steering.demo --dim 23 0.95   # fuerza la dim d023 (consciencia) a 0.95

Pasos previos (una vez):
  python -m steering.derive_vectors       # crea vectors/control_vectors.npy

Requiere GPU + el modelo HF. Corre en tu maquina.
"""
import argparse
import numpy as np

from . import config
from .translator import Humanizer
from .steering_model import SteeredLlama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None, help="override de fuerza")
    ap.add_argument("--dim", nargs=2, action="append", metavar=("IDX", "VAL"),
                    help="fija una dim manualmente, p.ej. --dim 23 0.95")
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()

    print("⏳ Cargando humanizador...")
    hz = Humanizer(device="cpu")
    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    if args.alpha is not None:
        llm.alpha = args.alpha
    print(f"✅ Listo. alpha={llm.alpha} | capas={llm.layers}\n")

    while True:
        try:
            q = input("📝 >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("salir", "exit", "q", ""):
            break

        prof = hz.profile(q)
        if args.dim:                              # overrides manuales
            for idx, val in args.dim:
                prof[int(idx)] = float(val)

        print("\n🧠 perfil top:")
        for i, name, v in hz.top_dims(prof):
            bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
            print(f"   [{i:03d}] {name:<32} {v:.3f} {bar}")

        neutral, steered = llm.generate_pair(q, prof, max_new_tokens=args.tokens)
        print(f"\n{'─'*60}\n⚪ NEUTRAL:\n{'─'*60}\n{neutral}")
        print(f"\n{'─'*60}\n🔴 DIRIGIDA (α={llm.alpha}):\n{'─'*60}\n{steered}\n")


if __name__ == "__main__":
    main()
