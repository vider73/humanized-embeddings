import torch
from transformers import pipeline
import pandas as pd
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN ---
MODEL_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_FILE = "dataset_humanizer_v3.json"
MODEL_EMB = "all-MiniLM-L6-v2"

BLOQUES_DIMENSIONES = [
    ["magnitud_espacial", "masa_densidad", "temperatura", "estado_materia", "velocidad_cinetica", "volatilidad"],
    ["longevidad_temporal", "luminosidad", "rugosidad", "humedad", "opacidad", "sonoridad"],
    ["vitalidad", "humanidad", "sensibilidad", "concrecion", "artificialidad", "complejidad"],
    ["orden_entropia", "moralidad", "peligrosidad", "sacralidad", "sociabilidad", "intencionalidad"],
    ["belleza_estetica", "valor_economico", "genero", "emocion_valencia", "energia_psiquica", "realidad_ontologica"]
]

print("⏳ Cargando modelos en la RTX 4090...")
pipe = pipeline("text-generation", model=MODEL_LLM, 
                model_kwargs={"dtype": torch.bfloat16, "low_cpu_mem_usage": True}, 
                device_map="auto")
model_emb = SentenceTransformer(MODEL_EMB, device='cuda')

def llamar_a_llama(prompt):
    messages = [{"role": "system", "content": "Eres un experto en análisis semántico y físico de alta precisión. Responde solo en JSON."},
                {"role": "user", "content": prompt}]
    # Bajamos la temperatura para mayor consistencia numérica
    outputs = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.1)
    return outputs[0]["generated_text"][-1]["content"]

def imprimir_reporte_palabra(palabra, scores, categoria):
    """Muestra los pesos recolectados de forma elegante en consola."""
    print(f"\n" + "═"*60)
    print(f" 💎 CONCEPTO: {palabra.upper()} | Categoría: {categoria}")
    print("═"*60)
    
    # Imprimir por bloques para que sea legible
    for i, bloque in enumerate(BLOQUES_DIMENSIONES):
        linea = []
        for dim in bloque:
            val = scores.get(dim, 0.0)
            linea.append(f"{dim[:12]}: {val:.3f}")
        print(f" B{i+1} | " + " | ".join(linea))
    print("─"*60)

def etiquetar_concepto_por_bloques(palabra, subcat):
    resultados_palabra = {}
    
    for bloque in BLOQUES_DIMENSIONES:
        prompt = f"""
        Analiza '{palabra}' ({subcat}) con precisión de 3 decimales.
        Asigna valores (0.000 a 1.000) para: {bloque}
        REGLA: Evita el 0.000 o 0.500 a menos que sea exacto. Usa todo el espectro (ej. 0.142, 0.887).
        Responde solo JSON.
        """
        res = llamar_a_llama(prompt)
        try:
            start, end = res.find('{'), res.rfind('}') + 1
            data = json.loads(res[start:end])
            for dim in bloque:
                if dim in data:
                    resultados_palabra[dim] = round(float(data[dim]), 3)
        except:
            continue
    
    return resultados_palabra if len(resultados_palabra) >= 28 else None # Tolerancia de 2 nulos

# --- CATEGORÍAS ---
CATEGORIAS = {
    "Materia": ["Elementos Químicos", "Minerales", "Astrofísica"],
    "Vida": ["Botánica", "Animales", "Anatomía"],
    "Tecnología": ["Herramientas", "Digital", "Motores"],
    "Abstracto": ["Teorías", "Ideologías", "Emociones"]
}

dataset = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

vistas = {d['palabra'].lower() for d in dataset}

try:
    for cat, subs in CATEGORIAS.items():
        for sub in subs:
            # Generación de palabras
            res_gen = llamar_a_llama(f"Lista 15 palabras únicas de {sub}. Solo palabras separadas por comas.")
            palabras = [x.strip() for x in res_gen.split(',')]
            
            for p in palabras:
                if not p or p.lower() in vistas: continue
                
                scores = etiquetar_concepto_por_bloques(p, sub)
                
                if scores:
                    emb = model_emb.encode(p).tolist()
                    dataset.append({
                        "palabra": p, "categoria": cat, "subcategoria": sub, 
                        "embedding_v_x": emb, **scores
                    })
                    vistas.add(p.lower())
                    
                    # IMPRIMIR REPORTE EN TIEMPO REAL
                    imprimir_reporte_palabra(p, scores, sub)
                    
                    if len(dataset) % 5 == 0:
                        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            json.dump(dataset, f, indent=2, ensure_ascii=False)
                        print(f"💾 Checkpoint: {len(dataset)} palabras guardadas.")

except KeyboardInterrupt:
    print("\n🛑 Detenido.")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)