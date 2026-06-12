import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# CONFIGURACIÓN
INPUT_FILE = "humanized_embeddings_dataset.json"
DATA_X_FILE = "dataset_X_embeddings.npy" # Entrada (Lo que "lee" la máquina)
DATA_Y_FILE = "dataset_Y_human.npy"      # Salida (Lo que "entiende" el humano)
METADATA_FILE = "dataset_metadata.json"  # Para saber qué fila es qué palabra

# Modelo de Embeddings Ligero y Potente (Estándar de HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Genera vectores de 384 dimensiones

def main():
    print("🚀 Cargando datos y modelo...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ No encuentro el dataset humanizado.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Preparar listas
    concepts = list(data.keys())
    # Ordenamos las dimensiones por ID para asegurar consistencia siempre (d000, d001...)
    first_key = concepts[0]
    dim_names = sorted(data[first_key].keys())
    
    print(f"📦 Conceptos: {len(concepts)} | Dimensiones Objetivo: {len(dim_names)}")
    
    # 2. Generar Embeddings Estándar (X)
    print("🧠 Generando embeddings de origen (Sentence-Transformers)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Encodeamos todas las palabras a la vez (Batch)
    embeddings_X = model.encode(concepts, show_progress_bar=True)
    
    # 3. Generar Matriz de Destino (Y)
    print("🎯 Construyendo matriz de dimensiones humanizadas...")
    human_vectors_Y = []
    
    for concept in tqdm(concepts):
        # Extraemos los valores en el orden correcto de dim_names
        vec = [data[concept][d] for d in dim_names]
        human_vectors_Y.append(vec)
        
    human_vectors_Y = np.array(human_vectors_Y, dtype=np.float32)

    # 4. Guardar todo
    print("💾 Guardando datasets procesados...")
    np.save(DATA_X_FILE, embeddings_X)
    np.save(DATA_Y_FILE, human_vectors_Y)
    
    metadata = {
        "concepts": concepts,
        "dimension_names": dim_names,
        "embedding_model": EMBEDDING_MODEL
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ ¡Datos listos para entrenar!")
    print(f"   X shape: {embeddings_X.shape}")
    print(f"   Y shape: {human_vectors_Y.shape}")

if __name__ == "__main__":
    main()