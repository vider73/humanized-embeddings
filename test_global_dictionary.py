import numpy as np
import json
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURACIÓN
GLOBAL_EMB_FILE = "global_embeddings.npy"
GLOBAL_WORDS_FILE = "global_words.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def cargar_recursos():
    """Carga los archivos y el modelo una sola vez al inicio."""
    print("⏳ Cargando modelo y base de datos... (esto puede tardar unos segundos)")
    
    if not os.path.exists(GLOBAL_EMB_FILE) or not os.path.exists(GLOBAL_WORDS_FILE):
        print(f"❌ ERROR: No se encuentran los archivos {GLOBAL_EMB_FILE} o {GLOBAL_WORDS_FILE}")
        sys.exit(1)

    try:
        # Cargar DB de embeddings
        db_emb = np.load(GLOBAL_EMB_FILE)
        
        # Cargar lista de palabras
        with open(GLOBAL_WORDS_FILE, "r", encoding="utf-8") as f:
            db_words = json.load(f)
            
        # Cargar modelo IA
        model = SentenceTransformer(MODEL_NAME)
        
        print("✅ Recursos cargados correctamente.\n")
        return db_emb, db_words, model
        
    except Exception as e:
        print(f"❌ Error al cargar recursos: {e}")
        sys.exit(1)

def buscar_similares(texto_usuario, db_emb, db_words, model):
    """Calcula similitud y muestra resultados."""
    print(f"🔎 Analizando: '{texto_usuario}'...")
    
    # 1. Generar embedding del input del usuario
    vector_real = model.encode(texto_usuario)
    
    # 2. Calcular similitud coseno con toda la base de datos
    sims = cosine_similarity(vector_real.reshape(1, -1), db_emb)[0]
    
    # 3. Obtener los 10 mejores resultados
    idxs = np.argsort(sims)[::-1][:10]
    
    print(f"\n   👇 RESULTADOS PARA '{texto_usuario}':")
    print("   " + "-"*40)
    for i in idxs:
        # Validación para evitar índices fuera de rango si la lista de palabras es menor
        if i < len(db_words):
            print(f"   - {db_words[i]:<20} | Similitud: {sims[i]:.4f}")
    print("   " + "-"*40 + "\n")

def main():
    # 1. Carga inicial (pesada)
    db_emb, db_words, model = cargar_recursos()

    # 2. Bucle de interacción
    while True:
        try:
            entrada = input("✍️  Escribe una palabra o frase (o 'salir' para terminar): ").strip()
            
            if entrada.lower() in ['salir', 'exit', 'quit']:
                print("👋 Cerrando programa.")
                break
            
            if not entrada:
                continue

            buscar_similares(entrada, db_emb, db_words, model)

        except KeyboardInterrupt:
            print("\n👋 Programa interrumpido por el usuario.")
            break
        except Exception as e:
            print(f"⚠️ Ocurrió un error en la búsqueda: {e}")

if __name__ == "__main__":
    main()