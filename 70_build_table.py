"""
70_build_table.py
═════════════════
Processes the corpus in batch and builds the humanized embeddings table.

  corpus_final.txt  ──► sentence-transformer ──► SemanticTranslator ──► table [N × 104]

Storage strategy:
  - numpy memmap  (tabla_embeddings.npy)  — fast access without loading everything into RAM
  - JSON          (tabla_words.json)      — list of terms (index ↔ position)
  - numpy         (tabla_stats.npy)       — mean and std per dimension (for normalization)

Interruption-tolerant: saves checkpoints every CHECKPOINT_EVERY batches
and resumes from where it left off if run again.

Usage:
  python 70_build_table.py
  python 70_build_table.py --batch 512 --workers 4
  python 70_build_table.py --resume          # resume from checkpoint
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (same as the GUI)
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATOR_PATH      = "semantic_translator.pth"
METADATA_FILE        = "dataset_metadata.json"

CORPUS_FILE     = Path("corpus/corpus_final.txt")
TABLE_DIR       = Path("tabla")
TABLE_EMB_FILE  = TABLE_DIR / "tabla_embeddings.npy"   # memmap float32 [N, 104] (dims from metadata)
TABLE_WORDS_FILE= TABLE_DIR / "tabla_words.json"
TABLE_STATS_FILE= TABLE_DIR / "tabla_stats.npy"        # [2, 104] → [mean, std]
CHECKPOINT_FILE = TABLE_DIR / "checkpoint.json"

BATCH_SIZE       = 256
NUM_WORKERS      = 2
CHECKPOINT_EVERY = 500    # save checkpoint every N batches
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE           = "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE (identical to the original project)
# ──────────────────────────────────────────────────────────────────────────────
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_models():
    from sentence_transformers import SentenceTransformer

    print("📦 Cargando sentence-transformer…")
    embedder  = SentenceTransformer(EMBEDDING_MODEL_NAME)
    input_dim = embedder.encode("test").shape[0]

    print("📦 Cargando SemanticTranslator…")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    human_dim  = len(meta["dimension_names"])
    dim_names  = meta["dimension_names"]

    translator = SemanticTranslator(input_dim, human_dim)
    translator.load_state_dict(torch.load(TRANSLATOR_PATH, map_location="cpu", weights_only=False))
    translator.eval().to(DEVICE)

    print(f"✅ Modelos listos | input_dim={input_dim} | human_dim={human_dim} | device={DEVICE}")
    return embedder, translator, input_dim, human_dim, dim_names


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"processed": 0, "total": 0}


def save_checkpoint(processed: int, total: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": processed, "total": total}, f)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESS
# ──────────────────────────────────────────────────────────────────────────────
def build_table(batch_size: int, num_workers: int, resume: bool):
    TABLE_DIR.mkdir(exist_ok=True)

    # Load corpus
    print(f"\n📄 Leyendo corpus: {CORPUS_FILE}")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    N = len(words)
    print(f"   {N:,} entradas")

    # Load models
    embedder, translator, input_dim, human_dim, dim_names = load_models()

    # ── Checkpoint / resume ────────────────────────────────────────────────
    ckpt      = load_checkpoint() if resume else {"processed": 0, "total": N}
    start_idx = ckpt["processed"] if resume else 0

    if start_idx > 0:
        print(f"🔄 Reanudando desde índice {start_idx:,} / {N:,}")

    # ── Initialize / open memmap ───────────────────────────────────────────
    mode = "r+" if (TABLE_EMB_FILE.exists() and resume and start_idx > 0) else "w+"
    emb_table = np.memmap(TABLE_EMB_FILE, dtype="float32", mode=mode,
                          shape=(N, human_dim))

    # Words already processed (if resuming)
    if resume and start_idx > 0 and TABLE_WORDS_FILE.exists():
        with open(TABLE_WORDS_FILE, "r", encoding="utf-8") as f:
            saved_words = json.load(f)
    else:
        saved_words = []

    # ── Batch loop ─────────────────────────────────────────────────────────
    total_batches = (N - start_idx + batch_size - 1) // batch_size
    t0 = time.time()
    processed = start_idx

    print(f"\n🚀 Procesando {N - start_idx:,} entradas en lotes de {batch_size}…\n")

    for batch_num, batch_start in enumerate(range(start_idx, N, batch_size)):
        batch_end   = min(batch_start + batch_size, N)
        batch_words = words[batch_start:batch_end]

        # 1. Sentence embeddings
        with torch.no_grad():
            raw_embs = embedder.encode(
                batch_words,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=DEVICE,
            )

            # 2. Translation to humanized space
            human_embs = translator(raw_embs.to(DEVICE)).cpu().numpy()

        # 3. Write to memmap
        emb_table[batch_start:batch_end] = human_embs
        saved_words.extend(batch_words)
        processed = batch_end

        # ── Progress ──────────────────────────────────────────────────────
        elapsed   = time.time() - t0
        speed     = (processed - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (N - processed) / speed if speed > 0 else 0
        pct       = processed * 100 / N

        print(
            f"\r  [{pct:5.1f}%] {processed:>8,}/{N:,}  "
            f"| {speed:6.0f} terms/s  "
            f"| ETA {remaining/60:.1f} min  "
            f"| lote {batch_num+1}/{total_batches}",
            end="", flush=True
        )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (batch_num + 1) % CHECKPOINT_EVERY == 0:
            emb_table.flush()
            with open(TABLE_WORDS_FILE, "w", encoding="utf-8") as f:
                json.dump(saved_words, f, ensure_ascii=False)
            save_checkpoint(processed, N)
            print(f"\n  💾 Checkpoint guardado en {processed:,}")

    # ── Final flush ────────────────────────────────────────────────────────
    emb_table.flush()
    print(f"\n\n✅ Embeddings calculados para {processed:,} entradas")

    # ── Save word list ─────────────────────────────────────────────────────
    with open(TABLE_WORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(saved_words, f, ensure_ascii=False)
    print(f"💾 Palabras guardadas: {TABLE_WORDS_FILE}")

    # ── Compute statistics (mean, std per dimension) ───────────────────────
    print("📊 Calculando estadísticas de la tabla…")
    _compute_stats(emb_table, human_dim, dim_names)

    # ── Clean up checkpoint ────────────────────────────────────────────────
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    size_gb = TABLE_EMB_FILE.stat().st_size / 1e9
    print(f"\n📦 Tabla final:")
    print(f"   Shape  : {N:,} × {human_dim}")
    print(f"   Tamaño : {size_gb:.2f} GB")
    print(f"   Tiempo : {(time.time()-t0)/60:.1f} min")
    print(f"\n➡  Siguiente paso:  python 80_build_faiss_index.py")


def _compute_stats(emb_table, human_dim: int, dim_names: list):
    """Computes mean and std per dimension for later normalization."""
    # Process in chunks so as not to load everything into RAM
    CHUNK = 50_000
    N     = emb_table.shape[0]
    acc   = np.zeros(human_dim, dtype=np.float64)
    acc2  = np.zeros(human_dim, dtype=np.float64)

    for i in range(0, N, CHUNK):
        chunk  = emb_table[i:i+CHUNK].astype(np.float64)
        acc   += chunk.sum(axis=0)
        acc2  += (chunk ** 2).sum(axis=0)

    mean = acc  / N
    std  = np.sqrt(np.maximum(acc2 / N - mean**2, 1e-12))
    stats = np.stack([mean, std]).astype(np.float32)
    np.save(TABLE_STATS_FILE, stats)

    print(f"📊 Estadísticas guardadas: {TABLE_STATS_FILE}")
    print(f"   Dimensión con mayor varianza: "
          f"{dim_names[int(std.argmax())]} (std={std.max():.4f})")
    print(f"   Dimensión con menor varianza: "
          f"{dim_names[int(std.argmin())]} (std={std.min():.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global CORPUS_FILE
    corpus_default = str(CORPUS_FILE)
    parser = argparse.ArgumentParser(description="Construye la tabla de embeddings humanizados")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE,
                        help=f"Tamaño de lote (default: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Workers de dataloader (default: {NUM_WORKERS})")
    parser.add_argument("--resume",  action="store_true",
                        help="Reanudar desde checkpoint anterior")
    parser.add_argument("--corpus",  type=str, default=corpus_default,
                        help="Ruta al corpus de entrada")
    args = parser.parse_args()

    CORPUS_FILE = Path(args.corpus)

    if not CORPUS_FILE.exists():
        print(f"❌ No se encuentra el corpus: {CORPUS_FILE}")
        print("   Ejecuta primero: python 60_download_corpus.py")
        return

    if not Path(TRANSLATOR_PATH).exists():
        print(f"❌ No se encuentra el modelo: {TRANSLATOR_PATH}")
        print("   Asegúrate de tener semantic_translator.pth en el directorio actual")
        return

    build_table(args.batch, args.workers, args.resume)


if __name__ == "__main__":
    main()