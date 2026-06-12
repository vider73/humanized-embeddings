import torch
import torch.nn as nn
import numpy as np
import json
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import sys

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_REVERSE_PATH = "reverse_translator.pth" # Modelo Generador (Humano -> Embedding)
MODEL_FORWARD_PATH = "semantic_translator.pth" # Modelo Analizador (Embedding -> Humano)
METADATA_FILE = "dataset_metadata.json"
EMBEDDINGS_DB = "dataset_X_embeddings.npy"
HUMAN_VECTORS_DB = "dataset_Y_human.npy"

# ==========================================
# CLASES DE REDES NEURONALES
# ==========================================
class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, output_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# UTILIDADES
# ==========================================
def load_system():
    print("⚙️ Cargando Sistema Semántico Completo...")
    if not os.path.exists(METADATA_FILE): return None
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f: meta = json.load(f)
    
    dim_names = meta["dimension_names"]
    concepts = meta["concepts"]
    embeddings_db = np.load(EMBEDDINGS_DB)
    human_db = np.load(HUMAN_VECTORS_DB)
    
    input_dim_human = len(dim_names)
    input_dim_emb = embeddings_db.shape[1]
    
    # Cargar Modelos
    model_rev = HumanToEmbedding(input_dim_human, input_dim_emb)
    model_fwd = SemanticTranslator(input_dim_emb, input_dim_human)
    
    try:
        model_rev.load_state_dict(torch.load(MODEL_REVERSE_PATH))
        model_fwd.load_state_dict(torch.load(MODEL_FORWARD_PATH))
        model_rev.eval(); model_fwd.eval()
    except:
        print("❌ Error cargando modelos .pth")
        return None
        
    return model_rev, model_fwd, dim_names, concepts, embeddings_db, human_db

def get_concept_vector(word, concepts, human_db):
    if word in concepts:
        idx = concepts.index(word)
        return human_db[idx], True
    return None, False

def print_vector_profile(vector, dim_names, title="PERFIL"):
    pairs = sorted(zip(dim_names, vector), key=lambda x: x[1], reverse=True)
    print(f"\n📊 {title} (Top 5):")
    for name, val in pairs[:5]:
        bar = "█" * int(val*15)
        print(f"   + {name[:25]:<25} : {val:.4f} {bar}")
    print(f"📉 {title} (Bottom 3):")
    for name, val in pairs[-3:]:
        print(f"   - {name[:25]:<25} : {val:.4f}")

# ==========================================
# COMANDOS DE CONSOLA
# ==========================================
def cmd_synthesize(args, current_vector, model_rev, embeddings_db, concepts, dim_names):
    if not args:
        print("   Uso: set [dimensión] [valor]  (ej: set tamaño 0.9)")
        return current_vector

    try:
        val = float(args[-1])
        dim_query = " ".join(args[:-1])
        
        # Buscar dimensión
        matches = difflib.get_close_matches(dim_query, dim_names, n=1, cutoff=0.3)
        if not matches:
            # Intentar substring
            for d in dim_names:
                if dim_query in d: matches = [d]; break
        
        if matches:
            dim_name = matches[0]
            idx = dim_names.index(dim_name)
            current_vector[idx] = val
            print(f"   ✅ {dim_name} -> {val}")
            
            # Generar
            tensor_in = torch.Tensor(current_vector).unsqueeze(0)
            with torch.no_grad(): gen_emb = model_rev(tensor_in).numpy()
            
            # Buscar
            sims = cosine_similarity(gen_emb, embeddings_db)
            top_idxs = sims[0].argsort()[::-1][:5]
            
            print(f"\n🔮 CONCEPTOS MÁS CERCANOS A TU FÓRMULA:")
            for i in top_idxs:
                print(f"   👉 {concepts[i].upper()} ({sims[0][i]:.4f})")
        else:
            print("   ❌ Dimensión no encontrada.")
    except:
        print("   ⚠️ Error de formato.")
    return current_vector

def cmd_inspect(args, concepts, human_db, dim_names):
    word = " ".join(args).lower()
    vec, found = get_concept_vector(word, concepts, human_db)
    if found:
        print_vector_profile(vec, dim_names, title=f"ANÁLISIS DE '{word.upper()}'")
    else:
        print(f"   ❌ '{word}' no está en la base de datos.")

def cmd_compare(args, concepts, human_db, dim_names):
    if len(args) < 2:
        print("   Uso: comparar [palabra1] [palabra2]")
        return
        
    w1, w2 = args[0].lower(), args[1].lower()
    v1, f1 = get_concept_vector(w1, concepts, human_db)
    v2, f2 = get_concept_vector(w2, concepts, human_db)
    
    if f1 and f2:
        diff = v1 - v2
        # Mayor diferencia positiva (w1 > w2)
        pairs = sorted(zip(dim_names, diff), key=lambda x: x[1], reverse=True)
        
        print(f"\n⚖️ COMPARATIVA: {w1.upper()} vs {w2.upper()}")
        print(f"   Donde {w1.upper()} gana:")
        for name, d in pairs[:3]:
            if d > 0.01: print(f"   + {name:<25} (+{d:.3f})")
            
        print(f"   Donde {w2.upper()} gana:")
        for name, d in pairs[-3:]:
            if d < -0.01: print(f"   + {name:<25} (+{abs(d):.3f})")
    else:
        print("   ❌ Una de las palabras no existe.")

def main():
    system = load_system()
    if not system: return
    model_rev, model_fwd, dim_names, concepts, embeddings_db, human_db = system
    
    # Vector de trabajo (Promedio inicial)
    mean_vector = np.mean(human_db, axis=0)
    current_vector = mean_vector.copy()
    
    print("\n🎛️ CONSOLA SEMÁNTICA LISTA")
    print("   Comandos disponibles:")
    print("   - ver [palabra]           : Inspeccionar valores reales")
    print("   - comparar [p1] [p2]      : Diferencias entre conceptos")
    print("   - set [dim] [val]         : Sintetizar concepto (ej: set tamaño 0.9)")
    print("   - reset                   : Reiniciar sintetizador al promedio")
    print("   - salir                   : Terminar")

    while True:
        try:
            user_input = input("\n>> ").strip().lower().split()
            if not user_input: continue
            
            cmd = user_input[0]
            args = user_input[1:]
            
            if cmd in ['salir', 'exit']: break
            elif cmd == 'ver': cmd_inspect(args, concepts, human_db, dim_names)
            elif cmd == 'comparar': cmd_compare(args, concepts, human_db, dim_names)
            elif cmd == 'set': current_vector = cmd_synthesize(args, current_vector, model_rev, embeddings_db, concepts, dim_names)
            elif cmd == 'reset': 
                current_vector = mean_vector.copy()
                print("   ✨ Sintetizador reiniciado.")
            else:
                print("   ❌ Comando desconocido.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

if __name__ == "__main__":
    main()