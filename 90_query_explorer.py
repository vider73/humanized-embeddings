"""
90_query_explorer.py
════════════════════
The main toy 🎮

Lets you explore the table of 5M humanized embeddings with:
  • Search by term/phrase               → nearest neighbors
  • Arithmetic in humanized space       → king - man + woman = ?
  • Navigation by dimension             → top terms in dimension X
  • Interpolation between two terms     → semantic trajectory
  • Detailed profile of any term
  • Anomaly detector                    → terms with "impossible" profiles
  • Polygraph mode                      → compares expected vs actual profile of a text

Usage:
  python 90_query_explorer.py
  python 90_query_explorer.py --interactive    # interactive CLI mode
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATOR_PATH      = "semantic_translator.pth"
REVERSE_PATH         = "reverse_translator.pth"
METADATA_FILE        = "dataset_metadata.json"

TABLE_DIR        = Path("tabla")
TABLE_EMB_FILE   = TABLE_DIR / "tabla_embeddings.npy"
TABLE_WORDS_FILE = TABLE_DIR / "tabla_words.json"
TABLE_STATS_FILE = TABLE_DIR / "tabla_stats.npy"
FAISS_INDEX_FILE = TABLE_DIR / "faiss_index.bin"
FAISS_META_FILE  = TABLE_DIR / "faiss_meta.json"

TOP_K_DEFAULT    = 50
INTERP_STEPS     = 20


# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURES
# ──────────────────────────────────────────────────────────────────────────────
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Dropout(0.1), nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(0.1), nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
class HumanizedExplorer:
    def __init__(self):
        self._load_all()

    def _load_all(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        print("🧠 Cargando sistema…")

        # Metadata and dimensions
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim_names = meta["dimension_names"]
        self.human_dim = len(self.dim_names)

        # Sentence transformer
        self.embedder  = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.input_dim = self.embedder.encode("test").shape[0]

        # Translator + Reverse
        self.translator = SemanticTranslator(self.input_dim, self.human_dim)
        self.translator.load_state_dict(torch.load(TRANSLATOR_PATH, map_location="cpu"))
        self.translator.eval()

        self.synthesizer = HumanToEmbedding(self.human_dim, self.input_dim)
        self.synthesizer.load_state_dict(torch.load(REVERSE_PATH, map_location="cpu"))
        self.synthesizer.eval()

        # Word table
        print("📂 Cargando palabras…")
        with open(TABLE_WORDS_FILE, "r", encoding="utf-8") as f:
            self.words = json.load(f)
        self.N = len(self.words)

        # Embeddings table (memmap)
        print(f"📂 Abriendo tabla memmap ({self.N:,} × {self.human_dim})…")
        self.table = np.memmap(TABLE_EMB_FILE, dtype="float32", mode="r",
                               shape=(self.N, self.human_dim))

        # Statistics
        self.stats = np.load(TABLE_STATS_FILE) if TABLE_STATS_FILE.exists() else None

        # FAISS
        print("⚡ Cargando índice FAISS…")
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(FAISS_META_FILE, "r") as f:
            self.faiss_meta = json.load(f)

        print(f"✅ Listo — {self.N:,} términos | {self.human_dim} dimensiones")
        print(f"   Índice: {self.faiss_meta['index_type']} | "
              f"{self.faiss_meta['index_size_mb']} MB\n")

    # ── Encoding ──────────────────────────────────────────────────────────────
    def encode(self, text: str) -> np.ndarray:
        """Text -> humanized vector"""
        with torch.no_grad():
            raw = self.embedder.encode(text, convert_to_tensor=False)
            raw = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
            vec = self.translator(raw).cpu().numpy()[0]
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            raw  = self.embedder.encode(texts, convert_to_tensor=True)
            raw  = raw.to(next(self.translator.parameters()).device)
            vecs = self.translator(raw).cpu().numpy()
        return vecs

    # ── Search ────────────────────────────────────────────────────────────────
    def search_vec(self, vec: np.ndarray, k: int = TOP_K_DEFAULT,
                   exclude: list[str] | None = None) -> list[tuple[str, float, np.ndarray]]:
        """Searches for the k nearest neighbors of a humanized vector."""
        query = vec.reshape(1, -1).astype(np.float32)
        D, I  = self.index.search(query, k + (len(exclude) if exclude else 0) + 5)
        results = []
        ex_set  = set(s.lower() for s in (exclude or []))
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= self.N:
                continue
            word = self.words[idx]
            if word.lower() in ex_set:
                continue
            results.append((word, float(dist), self.table[idx]))
            if len(results) >= k:
                break
        return results

    def search_term(self, text: str, k: int = TOP_K_DEFAULT) -> list[tuple[str, float, np.ndarray]]:
        vec = self.encode(text)
        return self.search_vec(vec, k, exclude=[text])

    # ── Profile ───────────────────────────────────────────────────────────────
    def profile(self, text: str, top_n: int = 10) -> dict:
        """Returns the complete semantic profile of a term."""
        vec    = self.encode(text)
        ranked = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)
        top    = [(self.dim_names[i], float(v)) for i, v in ranked[:top_n]]
        bottom = [(self.dim_names[i], float(v)) for i, v in ranked[-top_n:]]
        return {"term": text, "vector": vec, "top": top, "bottom": bottom}

    # ── Arithmetic ────────────────────────────────────────────────────────────
    def arithmetic(self, pos: list[str], neg: list[str] = None,
                   k: int = TOP_K_DEFAULT) -> list[tuple[str, float, np.ndarray]]:
        """
        Vector arithmetic in humanized space.
        Example: arithmetic(["rey", "mujer"], ["hombre"])
        """
        result = np.zeros(self.human_dim, dtype=np.float32)
        for t in pos:
            result = result + self.encode(t)
        for t in (neg or []):
            result = result - self.encode(t)
        result = np.clip(result, 0.0, 1.0)
        exclude = pos + (neg or [])
        return self.search_vec(result, k, exclude=exclude)

    # ── Interpolation ─────────────────────────────────────────────────────────
    def interpolate(self, term_a: str, term_b: str,
                    steps: int = INTERP_STEPS) -> list[dict]:
        """
        Semantic trajectory between two terms.
        Returns `steps` intermediate points with their nearest neighbor.
        """
        vec_a = self.encode(term_a)
        vec_b = self.encode(term_b)
        path  = []
        for i in range(steps + 1):
            t     = i / steps
            vec   = (1 - t) * vec_a + t * vec_b
            vec   = np.clip(vec, 0.0, 1.0)
            nn    = self.search_vec(vec, k=1)[0]
            path.append({
                "t":       round(t, 2),
                "nearest": nn[0],
                "dist":    round(nn[1], 4),
                "vec":     vec,
            })
        return path

    def interpolate_dual(self, term_a: str, term_b: str,
                         steps: int = INTERP_STEPS) -> list[dict]:
        """
        DUAL trajectory: compares interpolation in humanized space
        vs interpolation in translated raw space (384-dims).
        """
        # Humanized vectors
        vec_a_h = self.encode(term_a)
        vec_b_h = self.encode(term_b)

        # Raw vectors (384-dims) from the sentence transformer
        with torch.no_grad():
            vec_a_r = self.embedder.encode(term_a, convert_to_tensor=False).astype(np.float32)
            vec_b_r = self.embedder.encode(term_b, convert_to_tensor=False).astype(np.float32)

        path = []
        for i in range(steps + 1):
            t = i / steps

            # Humanized trajectory
            vec_h = np.clip((1 - t) * vec_a_h + t * vec_b_h, 0.0, 1.0)
            nn_h  = self.search_vec(vec_h, k=1)[0]

            # Raw trajectory → translated into humanized space
            vec_r_interp = (1 - t) * vec_a_r + t * vec_b_r
            with torch.no_grad():
                ten = torch.tensor(vec_r_interp, dtype=torch.float32).unsqueeze(0)
                vec_r_h = self.translator(ten).cpu().numpy()[0]
            nn_r = self.search_vec(vec_r_h, k=1)[0]

            # Divergence between the two trajectories
            divergence = float(np.abs(vec_h - vec_r_h).mean())

            path.append({
                "t":          round(t, 2),
                "human":      nn_h[0],
                "raw":        nn_r[0],
                "same":       nn_h[0].lower() == nn_r[0].lower(),
                "divergence": round(divergence, 4),
            })
        return path

    # ── Top by dimension ──────────────────────────────────────────────────────
    def top_by_dimension(self, dim_idx: int, k: int = 50,
                          descending: bool = True) -> list[tuple[str, float]]:
        """The k terms with the highest/lowest value in a dimension."""
        # With 5M entries reading the whole column is viable (only 20MB per dim)
        col   = self.table[:, dim_idx].copy()
        order = np.argsort(col)
        if descending:
            order = order[::-1]
        return [(self.words[i], float(col[i])) for i in order[:k]]

    # ── Semantic polygraph ────────────────────────────────────────────────────
    def poligraph(self, texts: list[str]) -> dict:
        """
        Analyzes a list of texts and detects semantic inconsistencies.
        Useful for evaluating LLM outputs.
        """
        vecs    = self.encode_batch(texts)
        mean    = vecs.mean(axis=0)
        std     = vecs.std(axis=0)

        # Dimensions with the most variation (inconsistent)
        volatile_dims = sorted(enumerate(std), key=lambda x: x[1], reverse=True)[:10]
        # Most active dimensions on average
        active_dims   = sorted(enumerate(mean), key=lambda x: x[1], reverse=True)[:10]

        return {
            "n_texts":       len(texts),
            "mean_vec":      mean,
            "std_vec":       std,
            "volatile_dims": [(self.dim_names[i], float(v)) for i, v in volatile_dims],
            "active_dims":   [(self.dim_names[i], float(v)) for i, v in active_dims],
            "coherence":     float(1.0 - std.mean()),   # 1=fully coherent
        }

    # ── Detect anomalies ──────────────────────────────────────────────────────
    def find_anomalies(self, terms: list[str]) -> list[dict]:
        """
        Detects terms whose humanized profile is unusually extreme
        or contradictory with respect to their neighbors.
        """
        vecs     = self.encode_batch(terms)
        results  = []
        for term, vec in zip(terms, vecs):
            neighbors = self.search_vec(vec, k=5, exclude=[term])
            if not neighbors:
                continue
            nbr_vecs  = np.array([n[2] for n in neighbors])
            nbr_mean  = nbr_vecs.mean(axis=0)
            deviation = float(np.abs(vec - nbr_mean).mean())
            results.append({
                "term":      term,
                "deviation": round(deviation, 4),
                "neighbors": [n[0] for n in neighbors],
            })
        results.sort(key=lambda x: x["deviation"], reverse=True)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
BARS = "▁▂▃▄▅▆▇█"

def _bar(v: float, width: int = 20) -> str:
    filled = int(v * width)
    frac   = int((v * width - filled) * 8)
    bar    = "█" * filled + (BARS[frac] if frac else "") + "░" * (width - filled)
    return bar[:width]

def _color(v: float) -> str:
    """ANSI color green→red depending on value."""
    if   v > 0.8: return "\033[91m"   # red
    elif v > 0.6: return "\033[93m"   # yellow
    elif v > 0.4: return "\033[92m"   # green
    else:         return "\033[94m"   # blue
RESET = "\033[0m"

def print_results(results: list[tuple[str, float, np.ndarray]], title: str = ""):
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    for rank, (word, dist, vec) in enumerate(results, 1):
        color = _color(dist)
        print(f"  {rank:2d}. {color}[{dist:6.4f}]{RESET}  {word:<35}")

def print_profile(p: dict, top_n: int = 12):
    print(f"\n{'═'*60}")
    print(f"  🔍 PERFIL: «{p['term']}»")
    print(f"{'═'*60}")
    print(f"  TOP dimensiones activas:")
    for name, val in p["top"][:top_n]:
        bar = _bar(val)
        color = _color(val)
        print(f"    {color}{bar}{RESET}  {val:.3f}  {name}")
    print(f"\n  BOTTOM dimensiones:")
    for name, val in p["bottom"][:5]:
        bar = _bar(val)
        print(f"    {bar}  {val:.3f}  {name}")

def print_interpolation(path: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  🌊 TRAYECTORIA SEMÁNTICA")
    print(f"{'─'*60}")
    for step in path:
        arrow = "━" * int(step["t"] * 30)
        space = " " * (30 - len(arrow))
        print(f"  t={step['t']:.2f}  [{arrow}>{space}]  {step['nearest']}")


def print_dual_interpolation(path, term_a, term_b):
    G = chr(27)+'[92m'; Y = chr(27)+'[93m'; R = chr(27)+'[91m'; X = chr(27)+'[0m'
    print()
    print('  ' + chr(8212)*72)
    print('  TRAYECTORIA DUAL: ' + term_a + ' -> ' + term_b)
    print('  ' + chr(8212)*72)
    print('  t      HUMANIZADO                   RAW->TRADUCIDO               DIV')
    print('  ' + '-'*70)
    for s in path:
        col  = G if s['same'] else Y
        dcol = R if s['divergence'] > 0.1 else ''
        sym  = 'ok' if s['same'] else '!='
        h = s['human'][:27]; r = s['raw'][:27]
        print('  t='+str(s['t'])+'  '+col+h.ljust(28)+X+'  '+r.ljust(28)+'  '+dcol+str(s['divergence'])+X+' '+col+sym+X)
    matches = sum(1 for s in path if s['same'])
    avg_div = sum(s['divergence'] for s in path) / len(path)
    max_s   = max(path, key=lambda s: s['divergence'])
    print()
    print('  Coincidencias     : '+str(matches)+'/'+str(len(path)))
    print('  Divergencia media : '+str(round(avg_div,4)))
    print('  Max divergencia   : t='+str(max_s['t'])+' ('+str(max_s['divergence'])+')')
    if avg_div > 0.08:
        print('  '+R+'ESPACIOS DIVERGENTES - la red humanizadora reinterpreta la trayectoria'+X)
    else:
        print('  '+G+'ESPACIOS ALINEADOS - ambas trayectorias cuentan la misma historia'+X)


# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ──────────────────────────────────────────────────────────────────────────────
HELP_TEXT = """
╔══════════════════════════════════════════════════════╗
║  🧠 HUMANIZED EMBEDDING EXPLORER                     ║
╠══════════════════════════════════════════════════════╣
║  <término>                → vecinos más cercanos     ║
║  + rey mujer - hombre     → aritmética vectorial     ║
║  ~ sacerdote robot        → interpolación            ║
║  ~~ sacerdote robot       → interpolación DUAL        ║
║  ! religiosidad           → top por dimensión        ║
║  ? sacerdote              → perfil detallado         ║
║  pg texto1 | texto2 | ..  → polígrafo semántico      ║
║  dim N                    → top por índice N         ║
║  help                     → este menú                ║
║  exit / quit              → salir                    ║
╚══════════════════════════════════════════════════════╝
"""

def interactive_loop(explorer: HumanizedExplorer):
    print(HELP_TEXT)
    while True:
        try:
            raw = input("▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("👋 Hasta luego")
            break
        if raw.lower() == "help":
            print(HELP_TEXT)
            continue

        # ── Arithmetic: + rey mujer - hombre ─────────────────────────────
        if raw.startswith("+") or (raw.startswith("-") and len(raw) > 1 and raw[1] != " "):
            pos_terms, neg_terms = [], []
            current = "pos"
            for tok in raw.replace("+", " + ").replace("-", " - ").split():
                if tok == "+":   current = "pos"
                elif tok == "-": current = "neg"
                elif tok:
                    (pos_terms if current == "pos" else neg_terms).append(tok)
            if not pos_terms:
                print("  ⚠ Especifica al menos un término positivo")
                continue
            label = " + ".join(pos_terms)
            if neg_terms: label += " - " + " - ".join(neg_terms)
            print(f"  ⚙️ Calculando: {label}…")
            results = explorer.arithmetic(pos_terms, neg_terms)
            print_results(results, f"Aritmética: {label}")

        # ── Dual interpolation: ~~ sacerdote robot ────────────────────────
        elif raw.startswith("~~"):
            parts = raw[2:].strip().split()
            if len(parts) < 2:
                print("  ⚠ Uso: ~~ término_a término_b")
                continue
            a = parts[0]; b = parts[1]
            print(f"  ⚙️ Interpolación dual «{a}» → «{b}»…")
            path = explorer.interpolate_dual(a, b)
            print_dual_interpolation(path, a, b)

        # ── Simple interpolation: ~ sacerdote robot ───────────────────────
        elif raw.startswith("~~"):
            parts = raw[2:].strip().split()
            if len(parts) < 2:
                print('  Uso: ~~ termino_a termino_b')
                continue
            a, b = parts[0], parts[1]
            print('  Calculando interpolacion dual ' + a + ' -> ' + b)
            path = explorer.interpolate_dual(a, b)
            print_dual_interpolation(path, a, b)

        elif raw.startswith("~"):
            parts = raw[1:].strip().split()
            if len(parts) < 2:
                print("  ⚠ Uso: ~ término_a término_b")
                continue
            a, b = " ".join(parts[:len(parts)//2]), " ".join(parts[len(parts)//2:])
            if len(parts) == 2:
                a, b = parts[0], parts[1]
            print(f"  ⚙️ Interpolando «{a}» → «{b}»…")
            path = explorer.interpolate(a, b)
            print_interpolation(path)

        # ── Top by dimension name: ! religiosidad ─────────────────────────
        elif raw.startswith("!"):
            query = raw[1:].strip().lower()
            matches = [(i, n) for i, n in enumerate(explorer.dim_names)
                       if query in n.lower()]
            if not matches:
                print(f"  ⚠ Dimensión «{query}» no encontrada")
                print(f"  Dimensiones disponibles (muestra): {explorer.dim_names[:10]}")
                continue
            idx, name = matches[0]
            print(f"  📊 Top 50 en dimensión [{idx}] «{name}»:")
            tops = explorer.top_by_dimension(idx, k=50)
            for rank, (w, v) in enumerate(tops, 1):
                bar = _bar(v, 15)
                print(f"    {rank:2d}. {bar}  {v:.4f}  {w}")

        # ── Top by numeric index: dim 42 ─────────────────────────────────
        elif raw.lower().startswith("dim "):
            try:
                idx = int(raw.split()[1])
                name = explorer.dim_names[idx]
                print(f"  📊 Top 50 en dimensión [{idx}] «{name}»:")
                tops = explorer.top_by_dimension(idx, k=50)
                for rank, (w, v) in enumerate(tops, 1):
                    bar = _bar(v, 15)
                    print(f"    {rank:2d}. {bar}  {v:.4f}  {w}")
            except (ValueError, IndexError):
                print("  ⚠ Uso: dim <número>")

        # ── Profile: ? sacerdote ───────────────────────────────────────────
        elif raw.startswith("?"):
            term = raw[1:].strip()
            print(f"  ⚙️ Calculando perfil de «{term}»…")
            p = explorer.profile(term)
            print_profile(p)

        # ── Polygraph: pg texto1 | texto2 ─────────────────────────────────
        elif raw.lower().startswith("pg "):
            texts = [t.strip() for t in raw[3:].split("|") if t.strip()]
            if len(texts) < 2:
                print("  ⚠ Uso: pg texto1 | texto2 | texto3")
                continue
            print(f"  ⚙️ Analizando {len(texts)} textos…")
            result = explorer.poligraph(texts)
            print(f"\n  🔎 POLÍGRAFO SEMÁNTICO ({len(texts)} muestras)")
            print(f"  Coherencia global: {result['coherence']:.3f}  "
                  f"({'✅ coherente' if result['coherence'] > 0.7 else '⚠ inconsistente'})")
            print(f"\n  Dimensiones más volátiles (inconsistencia):")
            for name, std in result["volatile_dims"][:6]:
                print(f"    {_bar(std, 12)}  std={std:.4f}  {name}")
            print(f"\n  Dimensiones dominantes (media):")
            for name, mean in result["active_dims"][:6]:
                print(f"    {_bar(mean, 12)}  avg={mean:.4f}  {name}")

        # ── Regular search ─────────────────────────────────────────────────
        else:
            term = raw
            print(f"  ⚙️ Buscando vecinos de «{term}»…")
            results = explorer.search_term(term)
            print_results(results, f"Vecinos de «{term}»")


# ──────────────────────────────────────────────────────────────────────────────
# AUTOMATIC DEMO
# ──────────────────────────────────────────────────────────────────────────────
def run_demo(explorer: HumanizedExplorer):
    print("\n" + "═"*60)
    print("  🎮 DEMO AUTOMÁTICO")
    print("═"*60)

    # 1. Profile
    for term in ["sacerdote", "inteligencia artificial", "amor"]:
        p = explorer.profile(term)
        print_profile(p)

    # 2. Arithmetic
    print("\n[Aritmética] sacerdote + tecnología - religión:")
    results = explorer.arithmetic(["sacerdote", "tecnología"], ["religión"])
    print_results(results)

    print("\n[Aritmética] rey - hombre + mujer:")
    results = explorer.arithmetic(["rey", "mujer"], ["hombre"])
    print_results(results)

    # 3. Interpolation
    print("\n[Interpolación] ciencia → misticismo:")
    path = explorer.interpolate("ciencia", "misticismo", steps=8)
    print_interpolation(path)

    # 4. Top dimensions
    print("\n[Dim] Top 10 en primera dimensión:")
    tops = explorer.top_by_dimension(0, k=10)
    for w, v in tops:
        print(f"  {_bar(v, 12)}  {v:.4f}  {w}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Explorador de embeddings humanizados")
    parser.add_argument("--demo",        action="store_true", help="Ejecutar demo automático")
    parser.add_argument("--interactive", action="store_true", help="Modo CLI interactivo")
    parser.add_argument("--term",        type=str,            help="Buscar un término y salir")
    args = parser.parse_args()

    # Verify files
    for f in [FAISS_INDEX_FILE, TABLE_EMB_FILE, TABLE_WORDS_FILE, METADATA_FILE]:
        if not Path(f).exists():
            print(f"❌ No encontrado: {f}")
            print("   Ejecuta los pasos anteriores primero.")
            return

    explorer = HumanizedExplorer()

    if args.term:
        results = explorer.search_term(args.term)
        print_results(results, f"Vecinos de «{args.term}»")
        p = explorer.profile(args.term)
        print_profile(p)

    elif args.demo:
        run_demo(explorer)

    else:
        # Default: interactive mode
        interactive_loop(explorer)


if __name__ == "__main__":
    main()
