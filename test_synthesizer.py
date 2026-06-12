import torch
import torch.nn as nn
import numpy as np
import json
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURACIÓN
MODEL_PATH = "reverse_translator.pth"
METADATA_FILE = "dataset_metadata.json"
EMBEDDINGS_DB = "dataset_X_embeddings.npy"
HUMAN_VECTORS_DB = "dataset_Y_human.npy" # Necesario para calcular el promedio

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def find_nearest_words(generated_vector, embeddings_db, concepts, top_k=5):
    sims = cosine_similarity(generated_vector.reshape(1, -1), embeddings_db)
    top_indices = sims[0].argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((concepts[idx], sims[0][idx]))
    return results

def get_dimension_index(user_input, dim_names):
    matches = difflib.get_close_matches(user_input, dim_names, n=1, cutoff=0.3)
    if matches: return dim_names.index(matches[0]), matches[0]
    for i, name in enumerate(dim_names):
        if user_input in name: return i, name
    return -1, None

def main():
    print("🧪 LABORATORIO DE SÍNTESIS SEMÁNTICA (V2 - BASE PROMEDIO)")
    
    # Cargar datos
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    dim_names = meta["dimension_names"]
    concepts = meta["concepts"]
    embeddings_db = np.load(EMBEDDINGS_DB)
    
    # --- CORRECCIÓN CLAVE: CALCULAR VECTOR BASE PROMEDIO ---
    human_db = np.load(HUMAN_VECTORS_DB)
    mean_vector = np.mean(human_db, axis=0)
    print("📊 Vector promedio calculado. Usando como 'Lienzo Neutro'.")
    
    # Inicializar con el promedio, no con ceros
    current_vector = mean_vector.copy()

    # Cargar Modelo
    model = HumanToEmbedding(len(dim_names), embeddings_db.shape[1])
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    except:
        print("❌ Faltan archivos.")
        return

    print(f"✅ Sistema listo. {len(concepts)} conceptos.")
    print("Escribe 'dimensión valor' (ej: 'peligro 0.9'). 'reset' para volver al promedio.")

    while True:
        # Mostrar solo desviaciones significativas del promedio
        diffs = current_vector - mean_vector
        active_changes = [(dim_names[i], val) for i, val in enumerate(current_vector) if abs(diffs[i]) > 0.1]
        
        if active_changes:
            print(f"\nMODIFICACIONES ACTIVAS: {[(d.split('_',1)[1][:10], f'{v:.2f}') for d,v in active_changes]}")
        
        user_input = input("\n>> Sintetizar: ").strip().lower()
        if user_input in ['salir', 'q']: break
        
        if user_input == 'reset':
            current_vector = mean_vector.copy()
            print("✨ Vuelta al neutro.")
            continue
            
        try:
            parts = user_input.split()
            val = float(parts[-1])
            dim_query = " ".join(parts[:-1])
            idx, real_name = get_dimension_index(dim_query, dim_names)
            
            if idx != -1:
                current_vector[idx] = val # Fijamos el valor
                print(f"   Set {real_name} -> {val}")
                
                # Generar
                tensor_in = torch.Tensor(current_vector).unsqueeze(0)
                with torch.no_grad():
                    gen_emb = model(tensor_in).numpy()
                
                # Buscar
                matches = find_nearest_words(gen_emb, embeddings_db, concepts)
                print(f"\n🔮 QUÉ SE PARECE A ESTO:")
                for word, score in matches:
                    print(f"   👉 {word.upper()} ({score:.4f})")
            else:
                print("❌ Dimensión desconocida.")
        except:
            print("⚠️ Formato: 'dimensión valor'")

if __name__ == "__main__":
    main()