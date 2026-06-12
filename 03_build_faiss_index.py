"""
03_build_faiss_index.py
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path

TABLE_DIR        = Path("tabla")
TABLE_EMB_FILE   = TABLE_DIR / "tabla_embeddings.npy"
TABLE_WORDS_FILE = TABLE_DIR / "tabla_words.json"
TABLE_STATS_FILE = TABLE_DIR / "tabla_stats.npy"
FAISS_INDEX_FILE = TABLE_DIR / "faiss_index.bin"
FAISS_META_FILE  = TABLE_DIR / "faiss_meta.json"

HNSW_M            = 32
HNSW_EF_SEARCH    = 64
HNSW_EF_CONSTRUCT = 200
IVF_NLIST  = 4096
IVF_NPROBE = 64
PQ_BITS    = 8


def load_table(normalize):
    print(f"Cargando tabla: {TABLE_EMB_FILE}")
    with open(TABLE_WORDS_FILE, "r", encoding="utf-8") as f:
        words = json.load(f)
    N = len(words)

    file_bytes = TABLE_EMB_FILE.stat().st_size
    human_dim  = file_bytes // (N * 4)
    print(f"   {N:,} vectores x {human_dim} dims  ({file_bytes/1e9:.2f} GB)")

    print("   Cargando en chunks (compatible Windows)...")
    data  = np.empty((N, human_dim), dtype=np.float32)
    CHUNK = 200_000
    with open(TABLE_EMB_FILE, "rb") as f:
        for i in range(0, N, CHUNK):
            end   = min(i + CHUNK, N)
            count = end - i
            raw   = f.read(count * human_dim * 4)
            flat  = np.frombuffer(raw, dtype=np.float32)
            data[i:end] = flat.reshape(count, human_dim)
            pct = end * 100 // N
            print(f"\r   {pct:3d}%  {end:,}/{N:,}", end="", flush=True)
    print()

    if normalize:
        print("   Normalizando L2...")
        import faiss
        faiss.normalize_L2(data)

    stats = np.load(TABLE_STATS_FILE) if TABLE_STATS_FILE.exists() else None
    return data, words, stats, human_dim


def choose_index_type(N, forced):
    if forced:
        return forced
    if N < 100_000:
        return "flat"
    elif N < 1_000_000:
        return "hnsw"
    else:
        return "ivfpq"


def build_flat(data, dim):
    import faiss
    print("Construyendo IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(data)
    return index, "flat"


def build_hnsw(data, dim):
    import faiss
    print(f"Construyendo IndexHNSWFlat (M={HNSW_M})...")
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
    index.hnsw.efSearch        = HNSW_EF_SEARCH
    CHUNK = 100_000
    N     = len(data)
    t0    = time.time()
    for i in range(0, N, CHUNK):
        index.add(data[i:i+CHUNK])
        pct = min(i+CHUNK, N) * 100 // N
        print(f"\r  {pct:3d}%  {min(i+CHUNK,N):,}/{N:,}", end="", flush=True)
    print(f"\n  HNSW listo en {(time.time()-t0):.1f}s")
    return index, "hnsw"


def build_ivfpq(data, dim):
    import faiss
    # pq_m debe dividir dim exactamente
    pq_m = 4
    for m in [dim, dim//2, dim//4, 16, 8, 4]:
        if m > 0 and dim % m == 0:
            pq_m = m
            break
    print(f"Construyendo IndexIVFPQ (nlist={IVF_NLIST}, pq_m={pq_m})...")
    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFPQ(quantizer, dim, IVF_NLIST, pq_m, PQ_BITS,
                                  faiss.METRIC_INNER_PRODUCT)
    index.nprobe = IVF_NPROBE
    N_TRAIN = min(len(data), IVF_NLIST * 40)
    print(f"  Entrenando con {N_TRAIN:,} muestras...")
    idx_train = np.random.choice(len(data), N_TRAIN, replace=False)
    index.train(data[idx_train])
    CHUNK = 200_000
    N     = len(data)
    t0    = time.time()
    for i in range(0, N, CHUNK):
        index.add(data[i:i+CHUNK])
        pct = min(i+CHUNK, N) * 100 // N
        print(f"\r  {pct:3d}%  {min(i+CHUNK,N):,}/{N:,}", end="", flush=True)
    print(f"\n  IVF-PQ listo en {(time.time()-t0):.1f}s")
    return index, "ivfpq"


def benchmark_index(index, data, words, k=10, n_queries=100):
    print(f"\nBenchmark ({n_queries} queries, k={k})...")
    sample_idx = np.random.choice(len(data), n_queries, replace=False)
    queries    = data[sample_idx]
    t0 = time.time()
    D, I = index.search(queries, k)
    qps  = n_queries / (time.time() - t0)
    print(f"   Velocidad: {qps:,.0f} queries/seg")
    print(f"   Muestra para [{words[sample_idx[0]]}]:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"     {rank+1:2d}. [{dist:.4f}]  {words[idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",         choices=["flat", "hnsw", "ivfpq"])
    parser.add_argument("--normalize",    action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    try:
        import faiss
    except ImportError:
        print("FAISS no instalado: pip install faiss-cpu")
        return

    if not TABLE_EMB_FILE.exists():
        print(f"Tabla no encontrada: {TABLE_EMB_FILE}")
        print("Ejecuta primero: python 02_build_table.py")
        return

    data, words, stats, human_dim = load_table(args.normalize)
    N, dim                         = data.shape
    index_type                     = choose_index_type(N, args.type)

    print(f"\nTipo de indice: {index_type.upper()}  (N={N:,}, dim={dim})")
    t0 = time.time()

    if index_type == "flat":
        index, itype = build_flat(data, dim)
    elif index_type == "hnsw":
        index, itype = build_hnsw(data, dim)
    else:
        index, itype = build_ivfpq(data, dim)

    build_time = time.time() - t0
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    index_size_mb = FAISS_INDEX_FILE.stat().st_size / 1e6

    meta = {
        "index_type":     itype,
        "n_entries":      N,
        "dim":            dim,
        "normalized":     args.normalize,
        "build_time_s":   round(build_time, 2),
        "index_size_mb":  round(index_size_mb, 1),
        "hnsw_m":         HNSW_M         if itype == "hnsw"  else None,
        "hnsw_ef_search": HNSW_EF_SEARCH  if itype == "hnsw"  else None,
        "ivf_nlist":      IVF_NLIST      if itype == "ivfpq" else None,
        "ivf_nprobe":     IVF_NPROBE     if itype == "ivfpq" else None,
    }
    with open(FAISS_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nIndice guardado: {FAISS_INDEX_FILE}  ({index_size_mb:.1f} MB)")
    print(f"Metadatos:       {FAISS_META_FILE}")
    print(f"Tiempo:          {build_time:.1f}s")

    if not args.no_benchmark:
        benchmark_index(index, data, words)

    print("\n-> Siguiente paso:  python 04_query_explorer.py")


if __name__ == "__main__":
    main()