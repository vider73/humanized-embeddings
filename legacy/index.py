import json
import torch
import re
import os
import time
import sys
import random
from typing import List, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import init, Fore, Style, Back

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
INPUT_FILE = "master_dimensions_prompts.json"
OUTPUT_FILE = "humanized_embeddings_dataset.json"

# Ajustes de Generación
WORDS_PER_BATCH = 5  # Pocas palabras por lote para mantener la pantalla moviéndose rápido
BATCH_SIZE = 1 

# Inicializar colores
init(autoreset=True)

# ==========================================
# 1. CARGA DEL MODELO (OPTIMIZADA)
# ==========================================
def load_model():
    print(f"{Fore.CYAN}🚀 Iniciando carga del modelo en RTX 4090...{Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="sdpa" 
        )
        print(f"{Fore.GREEN}✅ Sistema en línea. Motor neuronal listo.{Style.RESET_ALL}")
        return tokenizer, model
    except Exception as e:
        print(f"{Fore.RED}❌ Fallo crítico en núcleo: {e}{Style.RESET_ALL}")
        sys.exit(1)

# ==========================================
# 2. GENERADOR CREATIVO (IMAGINACIÓN)
# ==========================================
def generate_concepts(model, tokenizer, dimension_name, anchors):
    """
    Pide a la IA que imagine objetos relacionados con una dimensión.
    """
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Eres un motor de asociación de ideas. Genera sustantivos tangibles y abstractos.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Dame una lista de {WORDS_PER_BATCH} palabras (sustantivos) que tengan mucha relación con el concepto: '{dimension_name}' "
        f"(en el espectro de '{anchors['min']}' a '{anchors['max']}').\n"
        "Se original. No repitas lo obvio.\n"
        "FORMATO: Solo las palabras separadas por comas.\n"
        "Respuesta:<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.9, # Alta temperatura para creatividad
            do_sample=True,
            top_p=0.95
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_text = response.split("assistant")[-1].strip()
    
    # Limpieza
    if ":" in raw_text: raw_text = raw_text.split(":")[-1]
    potential_words = re.split(r',|\n|•|- ', raw_text)
    
    clean_words = []
    for w in potential_words:
        w = w.strip().lower().replace('.', '').replace('"', '')
        if len(w) > 2 and len(w) < 30:
            clean_words.append(w)
            
    return clean_words

# ==========================================
# 3. ANALIZADOR SEMÁNTICO (JUICIO)
# ==========================================
def get_scalar_value(model, tokenizer, prompt_template, concept):
    final_prompt = prompt_template.replace("{concepto}", concept)
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=8, 
            temperature=0.1, 
            do_sample=True 
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"(\d+\.\d+|\d+)", response)
    
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except: return 0.5
    return 0.5

# ==========================================
# UTILIDADES VISUALES
# ==========================================
def draw_bar(val):
    """Crea una barra de progreso estilo cyberpunk"""
    bars = int(val * 20) # Escala a 20 caracteres
    empty = 20 - bars
    
    # Colores basados en intensidad
    if val > 0.8: color = Fore.RED + Style.BRIGHT
    elif val > 0.6: color = Fore.MAGENTA
    elif val > 0.4: color = Fore.YELLOW
    elif val > 0.2: color = Fore.CYAN
    else: color = Fore.BLUE + Style.DIM

    full_bar = "█" * bars
    empty_bar = "░" * empty
    return f"{color}{full_bar}{empty_bar}{Style.RESET_ALL} {color}{val:.3f}{Style.RESET_ALL}"

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    tokenizer, model = load_model()
    
    # Cargar definiciones
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dimensions_db = json.load(f)

    # Cargar o crear base de datos
    full_dataset = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                full_dataset = json.load(f)
        except: pass

    known_concepts = set(full_dataset.keys())
    
    print(f"\n{Back.WHITE}{Fore.BLACK} SISTEMA DE GENERACIÓN PERPETUA INICIADO {Style.RESET_ALL}")
    print(f"{Fore.CYAN}Conceptos en memoria: {len(known_concepts)}{Style.RESET_ALL}\n")

    try:
        while True:
            # 1. Seleccionar una dimensión aleatoria para inspirarse
            target_dim = random.choice(dimensions_db)
            
            # Generar candidatos
            candidates = generate_concepts(model, tokenizer, target_dim['name'], target_dim['scale'])
            
            # Filtrar
            new_words = [w for w in candidates if w not in known_concepts]
            
            if not new_words:
                sys.stdout.write(f"{Fore.BLACK}{Back.BLACK}.{Style.RESET_ALL}") # Latido invisible si no hay nada
                sys.stdout.flush()
                continue

            # PROCESAR NUEVAS PALABRAS
            for word in new_words:
                print(f"\n{Fore.GREEN}╔════════════════════════════════════════════════════╗{Style.RESET_ALL}")
                print(f"{Fore.GREEN}║ NUEVO CONCEPTO DETECTADO: {Fore.WHITE}{word.upper().center(28)}{Fore.GREEN} ║{Style.RESET_ALL}")
                print(f"{Fore.GREEN}╚════════════════════════════════════════════════════╝{Style.RESET_ALL}")
                
                vector_data = {}
                
                # Calcular vector completo
                # Recorremos todas las dimensiones para esta palabra
                for idx, dim in enumerate(dimensions_db):
                    val = get_scalar_value(model, tokenizer, dim['llama3_prompt'], word)
                    vector_data[dim['id']] = val
                    
                    # VISUALIZACIÓN EN TIEMPO REAL
                    dim_label = f"{dim['name'][:15]:<15}"
                    print(f"  {Fore.WHITE}├─ {dim_label} {draw_bar(val)}")

                # Guardar
                full_dataset[word] = vector_data
                known_concepts.add(word)
                
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(full_dataset, f, indent=2, ensure_ascii=False)
                
                print(f"{Fore.GREEN}  └──> [💾 GUARDADO] Base de datos: {len(full_dataset)} items{Style.RESET_ALL}")
                
                # Breve pausa para admirar el resultado
                time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n{Back.RED}{Fore.WHITE} 🛑 SISTEMA DETENIDO POR EL USUARIO {Style.RESET_ALL}")
        print(f"Total recolectado: {len(full_dataset)} conceptos.")

if __name__ == "__main__":
    main()