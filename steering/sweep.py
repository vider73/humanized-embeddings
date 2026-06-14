"""
sweep.py — Easy mode: one phrase, several alphas in one go.

You type the phrase at the prompt and see the NEUTRAL once and then the STEERED
with each alpha (0.1, 0.2, 0.4, 0.8 by default), one after another. Greedy
(deterministic) so that the only thing that changes is the steering.

  python -m steering.sweep                       # profile = your phrase's own
  python -m steering.sweep --dim 23 0.95         # forces d023_consciencia
  python -m steering.sweep --alphas 0.1 0.3 0.6  # your own alphas
  python -m steering.sweep --tokens 120
"""
import argparse
import numpy as np

from .translator import Humanizer
from .steering_model import SteeredLlama
from . import transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--dim", nargs=2, action="append", metavar=("IDX", "VAL"),
                    help="fija una dim, p.ej. --dim 23 0.95")
    ap.add_argument("--isolate", action="store_true",
                    help="parte de 0.5 plano: empuja SOLO las dims de --dim (limpio)")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--mode", choices=["add", "clamp"], default=None,
                    help="add=empuje fijo clasico | clamp=controlador P (setpoint)")
    ap.add_argument("--kp", type=float, default=None, help="ganancia P (modo clamp)")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="inyectar solo en estas capas (subconjunto de INJECT_LAYERS)")
    ap.add_argument("--voice", type=float, nargs="+", action="append",
                    metavar="V", default=None,
                    help="nota con capa propia: DIM VAL CAPA [ALPHA]. Repetible. "
                         "Ej: --voice 90 0.95 14 0.35 --voice 23 0.95 16 0.2")
    args = ap.parse_args()

    print("⏳ Cargando humanizador...")
    hz = Humanizer(device="cpu")
    print("⏳ Cargando Llama + control vectors...")
    llm = SteeredLlama()
    if args.mode:
        llm.mode = args.mode
    if args.kp is not None:
        llm.kp = args.kp
    if args.layers:
        want = [L for L in args.layers if L in llm.cv]
        if want:
            llm.layers = want
        else:
            print(f"⚠ capas {args.layers} no derivadas; sigo con {llm.layers}")
    print(f"✅ Listo. capas={llm.layers} | modo={llm.mode} | alphas={args.alphas}\n")

    while True:
        try:
            q = input("📝 frase >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("salir", "exit", "q", ""):
            break

        # --isolate: ignores the phrase's profile, starts from flat 0.5 (neutral)
        prof = np.full(hz.human_dim, 0.5, np.float32) if args.isolate else hz.profile(q)
        if args.dim:
            for idx, val in args.dim:
                prof[int(idx)] = float(val)

        print("\n🧠 perfil top:")
        for i, name, v in hz.top_dims(prof):
            bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
            print(f"   [{i:03d}] {name:<32} {v:.3f} {bar}")

        # NEUTRAL (once, greedy)
        llm.clear()
        neutral = llm.generate(q, max_new_tokens=args.tokens, temperature=0.0)
        print(f"\n{'='*60}\n⚪ NEUTRAL\n{'='*60}\n{neutral}")

        if args.voice:
            # VOICING: each note on its own layer with its own pressure. --alphas
            # applies as the global alpha for notes WITHOUT their own alpha.
            voices = {}
            for v in args.voice:
                if len(v) not in (3, 4):
                    print(f"⚠ --voice mal formada (DIM VAL CAPA [ALPHA]): {v}")
                    continue
                di, val, L = int(v[0]), float(v[1]), int(v[2])
                a = float(v[3]) if len(v) == 4 else None
                p, pa = voices.get(L, (np.full(hz.human_dim, 0.5, np.float32), None))
                p[di] = val
                voices[L] = (p, a if a is not None else pa)
            for a in args.alphas:
                llm.alpha = a
                llm.set_voices(voices)
                steered = llm.generate(q, max_new_tokens=args.tokens, temperature=0.0)
                desc = ", ".join(f"L{L}:α={al if al is not None else a}"
                                 for L, (_, al) in voices.items())
                print(f"\n{'─'*60}\n🎹 voicing [{desc}]\n{'─'*60}\n{steered}")
                transcript.save("sweep/voicing", q, neutral, steered,
                                notes=transcript.notes_from_voices(
                                    voices, hz.dim_names, fallback_alpha=a),
                                mode=llm.mode)
        else:
            # STEERED for each alpha (all notes on self.layers)
            llm.set_profile(prof)
            for a in args.alphas:
                llm.alpha = a
                steered = llm.generate(q, max_new_tokens=args.tokens, temperature=0.0)
                print(f"\n{'─'*60}\n🔴 α={a}\n{'─'*60}\n{steered}")
                transcript.save("sweep", q, neutral, steered,
                                notes=transcript.notes_from_profile(
                                    prof, hz.dim_names, llm.layers, a),
                                mode=llm.mode)
        llm.clear()
        print()


if __name__ == "__main__":
    main()
