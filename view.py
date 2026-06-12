import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_PATH = "dataset_humanizer.json"
COL_METADATA = ['palabra', 'categoria', 'subcategoria', 'embedding_v_x']

# 1. CARGA DE LLAMA 3 (Matriz de Vocabulario)
print(f"⏳ Cargando matriz de embeddings de Llama 3 (128k tokens)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda" # Aprovechamos tu 4090
)

# Extraemos y normalizamos la matriz de pesos (vocabulario)
with torch.no_grad():
    # [128256, 4096]
    llm_weights = model_llm.get_input_embeddings().weight.data.to(torch.float32)
    llm_weights_norm = F.normalize(llm_weights, p=2, dim=1)

def cargar_datos_locales():
    if not os.path.exists(DATASET_PATH): return None, None
    with open(DATASET_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data).sort_values(by='palabra').reset_index(drop=True)
    cols_dim = df.columns.difference(COL_METADATA)
    df[cols_dim] = df[cols_dim].apply(pd.to_numeric, errors='coerce').fillna(0.5)
    mat_hum = df[cols_dim].values
    return df, mat_hum

def buscar_vecinos_reales_ia(palabra, top_n=5):
    token_ids = tokenizer.encode(palabra, add_special_tokens=False)
    if not token_ids: return []
    
    # Vector de la palabra (promedio de sus tokens)
    word_vec = llm_weights[token_ids].mean(dim=0, keepdim=True)
    word_vec_norm = F.normalize(word_vec, p=2, dim=1)
    
    # Similitud contra TODO el vocabulario de Llama 3
    sims = torch.mm(word_vec_norm, llm_weights_norm.t()).squeeze()
    top_values, top_indices = torch.top_k(sims, k=top_n + 10)
    
    resultados = []
    vistos = {palabra.lower()}
    for val, idx in zip(top_values, top_indices):
        token_txt = tokenizer.decode([idx.item()]).strip().lower()
        if len(token_txt) > 2 and token_txt not in vistos:
            resultados.append((token_txt, val.item()))
            vistos.add(token_txt)
        if len(resultados) >= top_n: break
    return resultados

def render_bar(valor, ancho=20):
    bloques = int(max(0, min(valor, 1)) * ancho)
    return f"[{'█' * bloques}{'░' * (ancho - bloques)}] {valor:.2f}"

# --- INTERFAZ ---
current_idx = None
df, mat_hum = cargar_datos_locales()

print("\n🚀 NAVEGADOR 'LLAMA EYE' v7.0")
print("Busca en el vocabulario real de 128k tokens.")

while True:
    if df is None:
        print("⚠️ Dataset no encontrado."); break
        
    prompt = f"\n🔎 [{len(df)} local] Palabra (o n/p/exit): "
    cmd = input(prompt).strip().lower()

    if cmd == 'exit': break
    
    # Navegación
    if cmd == 'n': current_idx = (current_idx + 1) % len(df)
    elif cmd == 'p': current_idx = (current_idx - 1) % len(df)
    else:
        match = df.index[df['palabra'].str.lower() == cmd]
        if len(match) > 0: current_idx = match[0]
        else: print("❌ No está en el diccionario local."); continue

    palabra_act = df.iloc[current_idx]['palabra']
    
    print("\n" + "═"*75)
    print(f" 🔍 ANÁLISIS PROFUNDO: {palabra_act.upper()}")
    print("═"*75)

    # 1. VECINOS IA (Vocabulario Real de Llama 3)
    vecinos_ia = buscar_vecinos_reales_ia(palabra_act)
    print(f"\n🤖 VECINOS EN EL CEREBRO DE LLAMA 3 (128k Vocab):")
    for txt, score in vecinos_ia:
        print(f"   • {txt:<18} | Similitud: {score*100:5.1f}%")

    # 2. VECINOS CONCEPTUALES (Tu Diccionario)
    from sklearn.metrics.pairwise import cosine_similarity
    sims_hum = cosine_similarity(mat_hum[current_idx].reshape(1,-1), mat_hum)[0]
    idx_hum = sims_hum.argsort()[-6:-1][::-1]
    
    print(f"\n🧠 VECINOS POR TUS DIALES (Humano):")
    for i in idx_hum:
        print(f"   • {df.iloc[i]['palabra']:<18} | Afinidad: {sims_hum[i]*100:5.1f}%")

    # 3. PERFIL DE DIALES
    print(f"\n📊 TOP DIMENSIONES HUMANAS:")
    cols_dim = df.columns.difference(COL_METADATA)
    vals = df.iloc[current_idx][cols_dim].values
    top_idx = np.abs(vals - 0.5).argsort()[-8:][::-1]
    for i in top_idx:
        d = cols_dim[i]
        print(f"   {d[:20]:<25} | {render_bar(df.iloc[current_idx][d])}")