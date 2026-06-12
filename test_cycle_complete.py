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
# 🚀 UTILS & CARGA
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
# 🎛️ INTERFAZ DE AFINACIÓN (NUEVO)
# ==========================================
def interactive_tuner(vector, dim_names):
    """
    Permite al usuario modificar valores del vector antes de procesarlo.
    """
    current_vec = vector.copy()
    
    while True:
        # Identificar las dimensiones más activas para mostrar
        indexed_dims = [(i, current_vec[i], dim_names[i]) for i in range(len(current_vec))]
        # Ordenar por valor descendente
        indexed_dims.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*20} 🎛️ PANEL DE CONTROL {'='*20}")
        print(" TOP DIMENSIONES ACTIVAS:")
        # Mostrar top 10
        for idx_list, (real_idx, val, name) in enumerate(indexed_dims[:10]):
            print(f"   [{real_idx:02d}] {name.split('_')[-1][:15]:<15} : {val:.3f} {draw_bar(val)}")
        
        print("\n Comandos:")
        print("  • [ID] [VALOR]  -> Ej: '4 0.95' para poner la dimensión 4 al 95%")
        print("  • 'ver todo'    -> Listar todas las dimensiones disponibles")
        print("  • [ENTER]       -> ✅ Confirmar y procesar")
        print(f"{'='*60}")

        cmd = input(" 🔧 Ajuste >> ").strip().lower()

        if cmd == "":
            return current_vec
        
        elif cmd == "ver todo":
            print("\n📜 LISTA COMPLETA:")
            for i, name in enumerate(dim_names):
                val = current_vec[i]
                print(f"   [{i:02d}] {name:<20} : {val:.3f}")
            input("Presiona Enter para volver...")

        else:
            try:
                parts = cmd.split()
                if len(parts) == 2:
                    idx = int(parts[0])
                    val = float(parts[1])
                    
                    if 0 <= idx < len(current_vec):
                        # Clamp valor entre 0 y 1
                        val = max(0.0, min(1.0, val))
                        current_vec[idx] = val
                        print(f"   ✅ Actualizado: [{dim_names[idx]}] -> {val}")
                    else:
                        print("   ❌ ID fuera de rango.")
                else:
                    print("   ❌ Formato incorrecto. Usa: ID VALOR")
            except ValueError:
                print("   ❌ Error de entrada.")

# ==========================================
# 🎯 EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    sys_data = load_system()
    if not sys_data: 
        print("Error cargando el sistema.")
        return

    print("\n🎓 SISTEMA OPERATIVO COGNITIVO v2.0")
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
            # Obtenemos el vector humano base
            human_vector_base = sys_data["translator"](tensor_emb).numpy()[0]

        # 2. 🎛️ Fase de Afinación Manual
        # Pasamos el control al usuario para que modifique el vector
        final_human_vector = interactive_tuner(human_vector_base, sys_data["dim_names"])

        # 3. Procesamiento (Memoria e Imaginación) usando el vector AFINADO
        print(f"\n🧠 PROCESANDO CON VECTOR AJUSTADO...")
        
        if sys_data["Y_train"] is not None:
            # Búsqueda usando el vector modificado
            sims = cosine_similarity(final_human_vector.reshape(1, -1), sys_data["Y_train"])[0]
            top_indices = sims.argsort()[::-1][:3]
            
            print(f"   {'='*50}")

            for idx in top_indices:
                mem_word = sys_data["concepts_train"][idx]
                mem_score = sims[idx]
                mem_vector = sys_data["Y_train"][idx]
                
                print(f"   1️⃣  RECUERDO: '{mem_word.upper()}' (Similitud: {mem_score:.2f})")
                
                # Imaginación
                if sys_data["global_emb"] is not None:
                    # Usamos el vector de memoria para sintetizar (asociación indirecta)
                    # Opcional: Podríamos usar 'final_human_vector' directamente si queremos
                    # que imagine basado en lo que editó el usuario, no en el recuerdo.
                    # Aquí mantengo la lógica original: Recuerdo -> Imaginación.
                    
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