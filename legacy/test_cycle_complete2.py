import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TRANSLATOR_PATH = "semantic_translator.pth"
REVERSE_PATH = "reverse_translator.pth"
METADATA_FILE = "dataset_metadata.json"
DATA_Y_FILE = "dataset_Y_human.npy" 
GLOBAL_EMB_FILE = "global_embeddings.npy"
GLOBAL_WORDS_FILE = "global_words.json"

# ==========================================
# 🧠 ARQUITECTURA
# ==========================================

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

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 🚀 CARGA Y UTILIDADES
# ==========================================
def draw_bar(val):
    val = max(0.0, min(1.0, val))
    return "█" * int(val * 15) + "░" * (15 - int(val * 15))

def load_system():
    print("⏳ Iniciando Sistema Cognitivo...")
    
    if not os.path.exists(METADATA_FILE): return None
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    if os.path.exists(DATA_Y_FILE): Y_train = np.load(DATA_Y_FILE)
    else: Y_train = None

    if os.path.exists(GLOBAL_EMB_FILE):
        global_emb = np.load(GLOBAL_EMB_FILE)
        with open(GLOBAL_WORDS_FILE, "r", encoding="utf-8") as f:
            global_words = json.load(f)
    else: global_emb, global_words = None, []

    try: embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except: return None

    dummy_vec = embedder.encode("test")
    input_dim = dummy_vec.shape[0] 
    human_dim = len(meta["dimension_names"])

    translator = SemanticTranslator(input_dim, human_dim)
    synthesizer = HumanToEmbedding(human_dim, input_dim)

    try:
        translator.load_state_dict(torch.load(TRANSLATOR_PATH, map_location='cpu'))
        synthesizer.load_state_dict(torch.load(REVERSE_PATH, map_location='cpu'))
        translator.eval()
        synthesizer.eval()
    except: return None

    return {
        "translator": translator,
        "synthesizer": synthesizer,
        "embedder": embedder,
        "Y_train": Y_train,
        "concepts_train": meta["concepts"],
        "global_emb": global_emb,
        "global_words": global_words,
        "dim_names": meta["dimension_names"]
    }

# ==========================================
# 🎛️ INTERFAZ DE AFINACIÓN (TUNING)
# ==========================================
def interactive_tuner(vector, dim_names):
    """
    Interface CLI para modificar manualmente los pesos del vector semántico.
    """
    current_vec = vector.copy() # Trabajamos sobre una copia
    
    while True:
        # Crear lista de tuplas (IndiceReal, Valor, Nombre)
        active_dims = []
        for i, val in enumerate(current_vec):
            active_dims.append((i, val, dim_names[i]))
        
        # Ordenar por valor descendente (lo más activo primero)
        active_dims.sort(key=lambda x: x[1], reverse=True)
        
        # --- RENDERIZADO DEL PANEL ---
        print(f"\n{'='*20} 🎛️ PANEL DE CONTROL {'='*20}")
        print(" TOP DIMENSIONES ACTIVAS:")
        
        for i in range(10): # Mostrar top 10
            idx, val, name = active_dims[i]
            # Limpiar nombre para visualización (quitar prefijos si existen)
            clean_name = name.split('_')[-1] 
            print(f"   [{idx:02d}] {clean_name:<15} : {val:.3f} {draw_bar(val)}")
            
        print("\n Comandos:")
        print("  • [ID] [VALOR]  -> Ej: '63 0.1' (Cambia dimensión 63 a 0.1)")
        print("  • 'ver todo'    -> Ver lista completa")
        print("  • [ENTER]       -> ✅ Confirmar y Procesar")
        print(f"{'='*60}")
        
        # --- CAPTURA DE INPUT ---
        user_cmd = input(" 🔧 Ajuste >> ").strip().lower()
        
        # Caso 1: Enter vacío -> Salir del bucle y devolver vector
        if user_cmd == "":
            return current_vec
        
        # Caso 2: Ver todo
        elif user_cmd == "ver todo":
            print("\n📜 LISTA COMPLETA DE DIMENSIONES:")
            for idx, val, name in active_dims:
                print(f"   [{idx:03d}] {name:<20} : {val:.3f}")
            input("Presiona Enter para volver...")
            
        # Caso 3: Modificar valor
        else:
            try:
                parts = user_cmd.split()
                if len(parts) == 2:
                    target_idx = int(parts[0])
                    target_val = float(parts[1])
                    
                    # Validar rango
                    if 0 <= target_idx < len(current_vec):
                        # Clamp entre 0 y 1 para estabilidad
                        target_val = max(0.0, min(1.0, target_val))
                        current_vec[target_idx] = target_val
                        print(f"   ✅ Actualizado: [{dim_names[target_idx]}] -> {target_val}")
                    else:
                        print(f"   ❌ Error: ID {target_idx} no existe.")
                else:
                    print("   ❌ Formato inválido. Usa: ID VALOR")
            except ValueError:
                print("   ❌ Error: Debes introducir números.")

# ==========================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    sys_data = load_system()
    if not sys_data: 
        print("Error: No se pudieron cargar los modelos o datos.")
        return

    print("\n🎓 SISTEMA OPERATIVO COGNITIVO.")
    print("Modo: Análisis -> Afinación Manual -> Memoria -> Imaginación")
    print("(Escribe 'salir' para terminar)\n")

    while True:
        user_input = input("\n📝 QUERY >> ").strip()
        if user_input.lower() in ["salir", "exit", "q"]: break
        if not user_input: continue

        # 1. Input -> Interpretación Inicial (IA)
        raw_emb = sys_data["embedder"].encode(user_input)
        tensor_emb = torch.Tensor(raw_emb).unsqueeze(0)
        
        with torch.no_grad():
            # Vector original generado por la red
            base_human_vector = sys_data["translator"](tensor_emb).numpy()[0]

        # 2. 🎛️ FASE DE AFINACIÓN (TUNING)
        # Aquí pasamos el vector al usuario para que lo edite
        final_vector = interactive_tuner(base_human_vector, sys_data["dim_names"])

        print(f"\n🧠 PROCESANDO CON VECTOR AJUSTADO...")
        print(f"   {'='*50}")

        # 3. Búsqueda en Memoria (Usando el vector FINAL)
        if sys_data["Y_train"] is not None:
            # Calculamos similitud contra el vector editado
            sims = cosine_similarity(final_vector.reshape(1, -1), sys_data["Y_train"])[0]
            top_indices = sims.argsort()[::-1][:3]
            
            for idx in top_indices:
                mem_word = sys_data["concepts_train"][idx]
                mem_score = sims[idx]
                mem_vector = sys_data["Y_train"][idx]
                
                print(f"   1️⃣  RECUERDO: '{mem_word.upper()}' (Similitud: {mem_score:.2f})")
                
                # 4. Imaginación (Synthesizer)
                # Opcional: ¿Imaginamos desde el recuerdo o desde el vector editado?
                # Lógica original conservada: Recuerdo -> Imaginación
                if sys_data["global_emb"] is not None:
                    tensor_mem = torch.Tensor(mem_vector).unsqueeze(0)
                    with torch.no_grad():
                        synth_emb = sys_data["synthesizer"](tensor_mem).numpy()
                    
                    d_sims = cosine_similarity(synth_emb, sys_data["global_emb"])[0]
                    d_idxs = d_sims.argsort()[::-1][:4] 

                    print(f"       ↳ 🔮 Asocia: ", end="")
                    found_words = []
                    for d_i in d_idxs:
                        w = sys_data["global_words"][d_i]
                        if w.lower() != mem_word.lower():
                            found_words.append(w)
                    
                    print(", ".join(found_words[:3]))
                    print(f"   {'-'*50}")
            print("")

if __name__ == "__main__":
    main()