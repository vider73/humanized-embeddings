import json
import numpy as np

INPUT_FILE = "humanized_embeddings_dataset.json"
OUTPUT_FILE = "humanized_embeddings_dataset_norm.json"

EPS = 1e-12
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95
OUTPUT_MIN = 0.05
OUTPUT_MAX = 0.95
CORR_THRESHOLD = 0.85


def load_matrix(data):
    terms = list(data.keys())
    dims = list(next(iter(data.values())).keys())

    matrix = np.array([
        [data[term][dim] for dim in dims]
        for term in terms
    ], dtype=np.float64)

    return terms, dims, matrix


def robust_normalize(matrix):
    # 1️⃣ Robust clipping
    low = np.percentile(matrix, LOW_PERCENTILE, axis=0)
    high = np.percentile(matrix, HIGH_PERCENTILE, axis=0)

    clipped = np.clip(matrix, low, high)

    # 2️⃣ Robust scaling (IQR)
    q1 = np.percentile(clipped, 25, axis=0)
    q3 = np.percentile(clipped, 75, axis=0)
    iqr = np.where((q3 - q1) < EPS, 1.0, q3 - q1)

    robust_scaled = (clipped - q1) / iqr

    # 3️⃣ Z-score
    mean = robust_scaled.mean(axis=0)
    std = np.where(robust_scaled.std(axis=0) < EPS, 1.0, robust_scaled.std(axis=0))
    z = (robust_scaled - mean) / std

    # 4️⃣ Soft rescaling to [0.05, 0.95]
    min_vals = z.min(axis=0)
    max_vals = z.max(axis=0)
    range_vals = np.where((max_vals - min_vals) < EPS, 1.0, max_vals - min_vals)

    normalized = (z - min_vals) / range_vals
    normalized = normalized * (OUTPUT_MAX - OUTPUT_MIN) + OUTPUT_MIN

    return normalized


def detect_dead_dimensions(matrix, dims):
    stds = matrix.std(axis=0)
    dead = [dims[i] for i in range(len(dims)) if stds[i] < 1e-3]
    return dead


def detect_dominant_dimensions(matrix, dims):
    variances = matrix.var(axis=0)
    threshold = np.percentile(variances, 95)
    dominant = [dims[i] for i in range(len(dims)) if variances[i] > threshold]
    return dominant


def detect_correlations(matrix, dims):
    corr = np.corrcoef(matrix, rowvar=False)
    correlated_pairs = []

    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            if abs(corr[i, j]) > CORR_THRESHOLD:
                correlated_pairs.append((dims[i], dims[j], corr[i, j]))

    return correlated_pairs


def reconstruct_json(terms, dims, matrix):
    new_data = {}
    for i, term in enumerate(terms):
        new_data[term] = {
            dims[j]: float(matrix[i, j])
            for j in range(len(dims))
        }
    return new_data


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms, dims, matrix = load_matrix(data)

    print("\n--- DIAGNÓSTICO ORIGINAL ---")
    print("Global mean:", matrix.mean())
    print("Global std :", matrix.std())

    dead_dims = detect_dead_dimensions(matrix, dims)
    dominant_dims = detect_dominant_dimensions(matrix, dims)
    correlated = detect_correlations(matrix, dims)

    print("\nDimensiones muertas:", dead_dims)
    print("Dimensiones dominantes:", dominant_dims)
    print("Correlaciones fuertes (> 0.85):")
    for d1, d2, c in correlated:
        print(f"  {d1} <-> {d2} : {c:.3f}")

    normalized_matrix = robust_normalize(matrix)

    print("\n--- DESPUÉS DE NORMALIZAR ---")
    print("Global mean:", normalized_matrix.mean())
    print("Global std :", normalized_matrix.std())
    print("Min:", normalized_matrix.min())
    print("Max:", normalized_matrix.max())

    new_data = reconstruct_json(terms, dims, normalized_matrix)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\nArchivo guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
