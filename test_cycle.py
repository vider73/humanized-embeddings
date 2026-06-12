import torch
import torch.nn as nn
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_PATH = "semantic_translator.pth"
METADATA_FILE = "dataset_metadata.json"
DATA_Y_FILE = "dataset_Y_human.npy" 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEST_WORDS = [
    "cañón",
    "virus",
    "democracia",
    "grito",
    "agujero negro",
    "asesinato",
    "fuego",
    "análisis",
    # ── Nuevas palabras añadidas ──
    "entropía",
    "silencio",
    "espejo",
    "relámpago",
    "olvido",
    "cuchillo",
    "infinito",
    "susurro",
    "colapso",
    "mente",
    "abismo",
    "hueso",
    "neón",
    "traición",
    "singularidad",
    "eco",
    "hielo",
    "paradoja",
    "sombra",
    "algoritmo",
    "nostalgia",
    "supernova",
    "engaño",
    "ceniza",
    "memoria",
    "caos",
    "éter",
    "piel",
    "vórtice",
    "rencor",
    "presidente",
    "olvido",
    "cabra",
    "caballo",
    "castaña",
    "mentira",
    "verdad",
    "AMOR",
    "DIOS",
    "MUERTE",
    "YO",
    "NADA",
]
# Definición de la Red (Debe ser idéntica al training)
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def load_system():
    print("⚙️ Cargando sistema neuronal...")
    
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Falta {METADATA_FILE}")
        return None
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Cargar la "Memoria" (Base de datos de vectores humanos ya calculados)
    try:
        Y_db = np.load(DATA_Y_FILE)
    except:
        print(f"❌ Falta {DATA_Y_FILE}")
        return None

    concepts_db = meta["concepts"]
    dim_names = meta["dimension_names"]
    
    # Cargar Embedder
    print("🧠 Cargando SentenceTransformer...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Configurar dimensiones para cargar el modelo Pytorch
    dummy_vec = embedder.encode("test")
    input_dim = dummy_vec.shape[0]
    output_dim = Y_db.shape[1]
    
    model = SemanticTranslator(input_dim, output_dim)
    try:
        # map_location='cpu' por si lo corres sin la GPU activa ahora mismo
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"❌ Error cargando modelo .pth: {e}")
        return None
    
    return model, embedder, Y_db, concepts_db, dim_names

def analyze_prediction(word, predicted_vector, dim_names, Y_db, concepts_db):
    print(f"\n🔹 ENTRADA: '{word.upper()}'")
    print(f"   {'-'*40}")
    
    # 1. PERFIL SEMÁNTICO (Top 5 Dimensiones activas)
    # Unimos valor con nombre
    pairs = list(zip(dim_names, predicted_vector))
    # Ordenamos de mayor a menor valor
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    print(f"   📊 LO QUE LA IA 'SIENTE' (Top Dimensiones):")
    for name, val in sorted_pairs[:5]:
        # Barra visual
        bar_len = int(val * 20) 
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"      + {name[:25]:<25} : {val:.4f} {bar}")
        
    # 2. BÚSQUEDA INVERSA (Nearest Neighbors)
    # Calculamos similitud coseno con TODA la base de datos
    # Reshape (1, -1) es necesario porque es un solo vector contra muchos
    sims = cosine_similarity(predicted_vector.reshape(1, -1), Y_db)
    
    # Obtener los índices de los 3 más parecidos (orden descendente)
    top_indices = sims[0].argsort()[::-1][:3]
    
    print(f"\n   🔍 BÚSQUEDA EN BASE DE DATOS (Conceptos más cercanos):")
    for idx in top_indices:
        match_word = concepts_db[idx]
        score = sims[0][idx]
        print(f"      👉 '{match_word}' (Similitud: {score:.4f})")
    
    print(f"   {'-'*40}")

def main():
    system = load_system()
    if not system: return
    model, embedder, Y_db, concepts_db, dim_names = system
    
    print("\n🚀 INICIANDO TEST DE CICLO DE VIDA (Traducción Ida y Vuelta)")
    print("============================================================")
    
    # 1. Convertir palabras de texto a Embeddings Standard
    test_embeddings = embedder.encode(TEST_WORDS)
    test_tensor = torch.Tensor(test_embeddings)
    
    # 2. Pasar por la Red Neuronal (Traducción)
    with torch.no_grad():
        humanized_vectors = model(test_tensor).numpy()
    
    # 3. Analizar cada resultado
    for i, word in enumerate(TEST_WORDS):
        analyze_prediction(word, humanized_vectors[i], dim_names, Y_db, concepts_db)

if __name__ == "__main__":
    main()