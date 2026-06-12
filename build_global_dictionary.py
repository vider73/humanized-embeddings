import requests
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Lista de frecuencia de palabras (subtítulos 2018)
URL_FREQ = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"

OUTPUT_EMBEDDINGS = "global_embeddings.npy"
OUTPUT_WORDS = "global_words.json"

# CAMBIO CLAVE: Modelo Multilingüe (Entiende significados, no solo letras)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cantidad de palabras (Top 20k cubre el 95% del lenguaje humano normal)
TOP_N_WORDS = 20000 

def download_and_clean():
    print("📡 Descargando lista de frecuencia...")
    try:
        response = requests.get(URL_FREQ, timeout=15)
        response.encoding = 'utf-8'
        lines = response.text.splitlines()
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return []

    print(f"   ✅ Descargadas {len(lines)} líneas.")
    
    clean_words = []
    seen = set()
    
    # Palabras que no aportan significado por sí solas (Stopwords)
    stopwords = {
        "que", "de", "no", "a", "la", "el", "es", "y", "en", "lo", "un", "por", 
        "qué", "me", "una", "te", "los", "se", "con", "para", "mi", "está", 
        "si", "bien", "pero", "yo", "eso", "las", "sí", "su", "tu", "aquí", 
        "del", "al", "como", "le", "más", "esto", "ya", "todo", "esta", "vamos"
    }

    print("🧹 Limpiando vocabulario...")
    for line in lines:
        parts = line.split()
        if not parts: continue
        
        word = parts[0].strip().lower()
        
        # Filtros:
        # 1. Solo letras (nada de '2018', '???')
        # 2. Más de 2 letras (quita 'y', 'o', 'tu')
        # 3. No es una stopword
        if word.isalpha() and len(word) > 2 and word not in seen and word not in stopwords:
            clean_words.append(word)
            seen.add(word)
        
        if len(clean_words) >= TOP_N_WORDS:
            break
            
    return clean_words

def main():
    print(f"🚀 INICIANDO CONSTRUCCIÓN CON MODELO: {MODEL_NAME}")
    print("   (Este modelo distingue 'coche' de 'noche' perfectamente)")
    
    # 1. Obtener palabras
    words = download_and_clean()
    if not words: return

    print(f"📝 Vocabulario: {len(words)} palabras.")

    # 2. Cargar Modelo
    print(f"🧠 Cargando modelo (puede tardar la primera vez)...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Vectorizar
    print("⚡ Calculando Embeddings...")
    embeddings = model.encode(words, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # 4. Guardar
    print(f"💾 Guardando {OUTPUT_EMBEDDINGS}...")
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    with open(OUTPUT_WORDS, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)

    print("\n🎉 DICCIONARIO RECONSTRUIDO.")
    print("⚠️ IMPORTANTE: Ahora debes actualizar 'MODEL_NAME' en tus otros scripts")
    print(f"   Usa: MODEL_NAME = '{MODEL_NAME}'")
    print("   Y re-entrenar tu red neuronal para que aprenda este nuevo 'idioma' vectorial.")

if __name__ == "__main__":
    main()