import json
import torch
import re
import os
import time
import sys
import random
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import init, Fore, Style, Back

# ==========================================
# 🔇 SILENCIAR ADVERTENCIAS MOLESTAS
# ==========================================
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
INPUT_FILE = "master_dimensions_prompts.json"
OUTPUT_FILE = "humanized_embeddings_dataset.json"

# Ajustes
WORDS_TO_DREAM = 10  # Cuantas palabras inventa por ronda
MIN_WORD_LENGTH = 3

init(autoreset=True)

# ==========================================
# 1. CARGA DEL MODELO (RTX 4090)
# ==========================================
def load_model():
    print(f"\n{Back.BLACK}{Fore.CYAN} 💾 CARGANDO NÚCLEO LLAMA-3 EN VRAM... {Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="sdpa"
        )
        print(f"{Fore.GREEN}✅ SISTEMA ONLINE. {Fore.BLACK}{Back.GREEN} RTX 4090 LISTA {Style.RESET_ALL}")
        return tokenizer, model
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR: {e}{Style.RESET_ALL}")
        sys.exit(1)

# ==========================================
# 2. SOÑAR CONCEPTOS (Creatividad)
# ==========================================
def dream_concepts(model, tokenizer, dimension_name):
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Genera una lista de {WORDS_TO_DREAM} términos (sustantivos comunes y nombres propios de objetos, lugares o entidades) en español relacionados con relacionados con: '{dimension_name}'.\n"
        "Solo palabras, separadas por comas. Sin explicaciones. ex: Enemigo, Napoleón, Carlos III\n"
        "Respuesta:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    words = [w.strip().lower().replace('.', '') for w in re.split(r',|\n|-', text) if len(w) > MIN_WORD_LENGTH]
    return words

# ==========================================
# 3. ANALIZAR (Juicio)
# ==========================================
def analyze(model, tokenizer, prompt_base, word):
    # Forzamos formato JSON estricto o numero directo
    final_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_base}\n"
        f"Palabra a evaluar: '{word}'\n"
        "Responde SOLO con un número decimal entre 0.0 y 1.0. Nada más.\n"
        "Respuesta:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
    res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"(\d+\.\d+|\d+)", res)
    if match:
        return float(match.group(1))
    return 0.5

# ==========================================
# VISUALES
# ==========================================
def draw_bar(label, val):
    bars = int(val * 20)
    space = 20 - bars
    
    if val > 0.8: col = Fore.RED + Style.BRIGHT
    elif val > 0.6: col = Fore.YELLOW + Style.BRIGHT
    elif val > 0.4: col = Fore.CYAN
    else: col = Fore.BLUE + Style.DIM
    
    bar = f"{col}{'█' * bars}{'░' * space}{Style.RESET_ALL}"
    return f"{Fore.WHITE}{label:<25} {bar} {col}{val:.3f}{Style.RESET_ALL}"

# ==========================================
# MAIN
# ==========================================
def main():
    tokenizer, model = load_model()
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dims_db = json.load(f)
        
    full_dataset = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try: full_dataset = json.load(f)
            except: pass
            
    known = set(full_dataset.keys())
    print(f"{Fore.MAGENTA}Base de datos actual: {len(known)} conceptos.{Style.RESET_ALL}\n")

    try:
        while True:
            # FASE 1: SOÑAR
            target_dim = random.choice(dims_db)
            print(f"{Fore.BLACK}{Back.WHITE} 💤 SOÑANDO CONCEPTOS SOBRE: {target_dim['name'].upper()} {Style.RESET_ALL}")
            
            candidates = dream_concepts(model, tokenizer, target_dim['name'])
            new_words = [w for w in candidates if w not in known]
            
            if not new_words:
                print("...nada nuevo...")
                continue
                
            # FASE 2: ANALIZAR
            for word in new_words:
                print(f"\n{Fore.GREEN}╔══════════════════════════════════════════╗{Style.RESET_ALL}")
                print(f"{Fore.GREEN}║ PROCESANDO: {Fore.WHITE}{word.upper().center(26)}{Fore.GREEN} ║{Style.RESET_ALL}")
                print(f"{Fore.GREEN}╚══════════════════════════════════════════╝{Style.RESET_ALL}")
                
                vector = {}
                for dim in dims_db:
                    # Usamos el campo llama3_prompt o prompt según tengas en tu JSON
                    prompt = dim.get('llama3_prompt', dim.get('prompt', ''))
                    
                    val = analyze(model, tokenizer, prompt, word)
                    vector[dim['id']] = val
                    
                    # VISUALIZACIÓN EN TIEMPO REAL
                    print(f"  {draw_bar(dim['name'], val)}")
                
                full_dataset[word] = vector
                known.add(word)
                
                # Guardado seguro
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(full_dataset, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\n👋 Guardado y cerrado.")

if __name__ == "__main__":
    main()