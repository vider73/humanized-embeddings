"""
SEMANTIC ATTENTION INJECTION v3 — Llama3 via Ollama
=====================================================
Adapta el LogitsProcessor semántico para atacar modelos Llama3
servidos localmente via Ollama.

El problema con Llama3 vs bloom:
- Llama3 es un modelo de instrucciones (chat), no solo completado
- Ollama no expone logits directamente via API REST
- Necesitamos usar llama-cpp-python o cargar el modelo directamente

DOS MODOS disponibles:
  MODE = "llamacpp"   → llama-cpp-python (acceso a logits, sesgo real)
  MODE = "ollama"     → Ollama API REST  (sesgo via prompt engineering)

Requiere para llamacpp:
    pip install llama-cpp-python sentence-transformers scikit-learn

Requiere para ollama:
    pip install ollama sentence-transformers scikit-learn
    + Ollama corriendo en localhost:11434
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATOR_PATH      = "semantic_translator.pth"
REVERSE_PATH         = "reverse_translator.pth"
METADATA_FILE        = "dataset_metadata.json"
DATA_Y_FILE          = "dataset_Y_human.npy"
GLOBAL_EMB_FILE      = "global_embeddings.npy"
GLOBAL_WORDS_FILE    = "global_words.json"

# Elige el modo:
# "llamacpp" → sesgo real sobre logits (necesita el .gguf del modelo)
# "ollama"   → sesgo via system prompt enriquecido (más fácil, menos preciso)
MODE = "llamacpp"

# Configuración llamacpp
GGUF_PATH   = "models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  # Ajusta tu ruta
N_CTX       = 2048
N_GPU_LAYERS = -1   # -1 = todas las capas en GPU

# Configuración Ollama
OLLAMA_MODEL  = "llama3.2:3b"
OLLAMA_HOST   = "http://localhost:11434"

# Parámetros compartidos
ALPHA          = 5.0
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.001
TOP_P          = 0.92

# Cuántas dimensiones top incluir en el system prompt (modo ollama)
TOP_DIMS_IN_PROMPT = 5

# ==========================================
# 🧠 ARQUITECTURAS
# ==========================================
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),       nn.ReLU(),                       nn.Dropout(0.1),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  nn.BatchNorm1d(256),  nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(256, 512),        nn.BatchNorm1d(512),  nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, 1024),       nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 🎯 MODO A: LLAMACPP — Sesgo real sobre logits
# ==========================================
class SemanticLogitsProcessorLlamaCpp:
    """
    Intercepta los logits de llama-cpp-python en cada paso
    y aplica el boost semántico directamente.
    
    llama-cpp-python acepta logits_processor como lista de callables
    con firma: (input_ids: List[int], logits: List[float]) -> List[float]
    """
    def __init__(self, boost: np.ndarray, alpha: float):
        self.boost = boost   # [vocab_size] numpy array
        self.alpha = alpha

    def __call__(self, input_ids, logits):
        # logits es una lista de floats de tamaño vocab_size
        logits_arr = np.array(logits, dtype=np.float32)
        logits_arr += self.boost * self.alpha
        return logits_arr.tolist()


class VocabProjectorNumpy:
    """
    Versión numpy del proyector para llamacpp (sin torch en el callback).
    Proyección lineal simple: human_dim → vocab_size
    """
    def __init__(self, human_dim: int, vocab_size: int):
        # Inicialización pequeña pero con señal
        np.random.seed(42)
        self.W1 = np.random.normal(0, 0.01, (human_dim, 512)).astype(np.float32)
        self.b1 = np.zeros(512, dtype=np.float32)
        self.W2 = np.random.normal(0, 0.01, (512, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def project(self, semantic_vec: np.ndarray) -> np.ndarray:
        h = np.tanh(semantic_vec @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def run_llamacpp(sys_data, semantic_vec_np, user_input, config):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python no instalado. Ejecuta: pip install llama-cpp-python")
        return

    print(f"⏳ Cargando {GGUF_PATH}...")
    llm = Llama(
        model_path    = GGUF_PATH,
        n_ctx         = N_CTX,
        n_gpu_layers  = N_GPU_LAYERS,
        verbose       = False
    )
    vocab_size = llm.n_vocab()
    human_dim  = len(semantic_vec_np)

    projector = VocabProjectorNumpy(human_dim, vocab_size)
    boost     = projector.project(semantic_vec_np)

    # Formato de chat Llama3
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Continue the user's thought coherently."},
        {"role": "user",   "content": user_input}
    ]

    # Generación NEUTRAL
    print(f"\n{'─'*55}\n⚪ NEUTRAL (sin sesgo):\n{'─'*55}")
    out_n = llm.create_chat_completion(
        messages    = messages,
        max_tokens  = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        top_p       = TOP_P,
    )
    print(f"  {out_n['choices'][0]['message']['content']}")

    # Generación SESGADA
    semantic_processor = SemanticLogitsProcessorLlamaCpp(boost, config["alpha"])

    print(f"\n{'─'*55}\n🔴 DIRIGIDA (α={config['alpha']}):\n{'─'*55}")
    out_b = llm.create_chat_completion(
        messages          = messages,
        max_tokens        = MAX_NEW_TOKENS,
        temperature       = TEMPERATURE,
        top_p             = TOP_P,
        logits_processor  = [semantic_processor],
    )
    print(f"  {out_b['choices'][0]['message']['content']}")


# ==========================================
# 🎯 MODO B: OLLAMA — Sesgo via system prompt
# ==========================================
def build_semantic_system_prompt(semantic_vec_np, dim_names, top_n=TOP_DIMS_IN_PROMPT):
    """
    Construye un system prompt que inyecta el perfil semántico
    como instrucciones de registro y tono para el modelo.
    
    Es sesgo indirecto — no toca logits pero es efectivo con
    modelos de instrucciones como Llama3 que siguen bien el system prompt.
    """
    ranked = sorted(enumerate(semantic_vec_np), key=lambda x: x[1], reverse=True)
    top_dims = ranked[:top_n]

    # Mapa de dimensiones a instrucciones de registro
    DIM_TO_INSTRUCTION = {
        "racionalidad":        "Use precise, logical, mathematical language.",
        "asco":                "Express strong aversion, disgust, or rejection.",
        "calma":               "Use calm, peaceful, measured language.",
        "necesidad":           "Convey urgency, necessity, vital importance.",
        "curiosidad":          "Show intellectual curiosity and wonder.",
        "tristeza":            "Use melancholic, sorrowful tone.",
        "alegria":             "Use joyful, enthusiastic, positive language.",
        "misterio":            "Use mysterious, enigmatic, suggestive language.",
        "violencia":           "Describe with intensity and raw force.",
        "ternura":             "Use gentle, warm, affectionate language.",
        "caos":                "Embrace disorder, unpredictability, entropy.",
        "divinidad":           "Use sacred, transcendent, spiritual language.",
        "conocimiento":        "Demonstrate expertise and depth of knowledge.",
        "memoria":             "Reference past, history, recollection.",
        "fragilidad":          "Emphasize vulnerability, transience, delicacy.",
        "control":             "Use precise, controlled, structured language.",
        "claridad":            "Be clear, direct, unambiguous.",
        "verdad":              "Emphasize facts, truth, objectivity.",
        "belleza":             "Use aesthetic, elegant, beautiful language.",
        "dolor":               "Convey suffering, pain, anguish.",
        "esperanza":           "Project optimism, possibility, future.",
        "soledad":             "Convey isolation, loneliness, silence.",
        "poder":               "Use authoritative, powerful, commanding language.",
        "durabilidad":         "Emphasize permanence, lasting effects, endurance.",
        "salud":               "Reference wellbeing, vitality, balance.",
        "artificialidad":      "Highlight artifice, construction, design.",
        "probabilidad":        "Use probabilistic, statistical, uncertain language.",
        "intencionalidad":     "Show deliberate purpose and intent.",
    }

    instructions = []
    for idx, val in top_dims:
        name = dim_names[idx].lower()
        # Buscar coincidencia parcial en el mapa
        matched = next(
            (instr for key, instr in DIM_TO_INSTRUCTION.items() if key in name),
            None
        )
        if matched:
            weight = "strongly" if val > 0.8 else "moderately"
            instructions.append(f"- {weight.capitalize()} {matched.lower()}")

    if not instructions:
        instructions = ["- Respond naturally and coherently."]

    system = (
        "You are a helpful assistant with a specific cognitive-emotional profile. "
        "Your response style must reflect the following active semantic dimensions:\n"
        + "\n".join(instructions)
        + "\n\nMaintain this profile consistently throughout your response. "
        "Be concise but expressive."
    )
    return system


def run_ollama(sys_data, semantic_vec_np, user_input, config):
    try:
        import ollama
    except ImportError:
        print("❌ ollama no instalado. Ejecuta: pip install ollama")
        return

    dim_names = sys_data["dim_names"]

    # System prompt NEUTRAL (genérico)
    system_neutral = "You are a helpful assistant. Continue the user's thought coherently and naturally."

    # System prompt SESGADO (con perfil semántico)
    system_biased = build_semantic_system_prompt(semantic_vec_np, dim_names)

    print(f"\n🧬 SYSTEM PROMPT SEMÁNTICO GENERADO:")
    print(f"{'─'*55}")
    print(system_biased)

    # Generación NEUTRAL
    print(f"\n{'─'*55}\n⚪ NEUTRAL (sin sesgo):\n{'─'*55}")
    resp_n = ollama.chat(
        model   = OLLAMA_MODEL,
        messages = [
            {"role": "system",  "content": system_neutral},
            {"role": "user",    "content": user_input}
        ],
        options = {
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "num_predict": MAX_NEW_TOKENS,
        }
    )
    print(f"  {resp_n['message']['content']}")

    # Generación SESGADA
    print(f"\n{'─'*55}\n🔴 DIRIGIDA (perfil semántico activo):\n{'─'*55}")
    resp_b = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [
            {"role": "system",  "content": system_biased},
            {"role": "user",    "content": user_input}
        ],
        options = {
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "num_predict": MAX_NEW_TOKENS,
        }
    )
    print(f"  {resp_b['message']['content']}")


# ==========================================
# 📦 CARGA DEL SISTEMA COGNITIVO
# ==========================================
def load_cognitive_system():
    print("⏳ Cargando Sistema Cognitivo Semántico...")
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"No se encuentra {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    Y_train      = np.load(DATA_Y_FILE)     if os.path.exists(DATA_Y_FILE)     else None
    global_emb   = np.load(GLOBAL_EMB_FILE) if os.path.exists(GLOBAL_EMB_FILE) else None
    global_words = json.load(open(GLOBAL_WORDS_FILE, encoding="utf-8")) \
                   if os.path.exists(GLOBAL_WORDS_FILE) else []

    embedder  = SentenceTransformer(EMBEDDING_MODEL_NAME)
    input_dim = embedder.encode("test").shape[0]
    human_dim = len(meta["dimension_names"])

    translator  = SemanticTranslator(input_dim, human_dim)
    synthesizer = HumanToEmbedding(human_dim, input_dim)
    translator.load_state_dict(torch.load(TRANSLATOR_PATH,  map_location="cpu"))
    synthesizer.load_state_dict(torch.load(REVERSE_PATH,    map_location="cpu"))
    translator.eval(); synthesizer.eval()

    print(f"   ✅ Traductor cargado — {human_dim} dimensiones semánticas")
    return {
        "translator": translator, "synthesizer": synthesizer,
        "embedder": embedder, "Y_train": Y_train,
        "concepts_train": meta["concepts"],
        "global_emb": global_emb, "global_words": global_words,
        "dim_names": meta["dimension_names"],
        "human_dim": human_dim,
    }


# ==========================================
# 🖥️ UTILIDADES
# ==========================================
def draw_bar(val, width=15):
    val = max(0.0, min(1.0, val))
    return "█" * int(val * width) + "░" * (width - int(val * width))

def print_profile(vector, dim_names, top_n=5):
    ranked = sorted(enumerate(vector), key=lambda x: x[1], reverse=True)
    print(f"\n🧠 TOP-{top_n} DIMENSIONES ACTIVAS:")
    for rank, (idx, val) in enumerate(ranked[:top_n], 1):
        name = dim_names[idx].replace("d0", "").replace("_", " ").strip()
        print(f"   {rank}. [{idx:03d}] {name:<28} {val:.3f} {draw_bar(val)}")

def print_memories(sys_data, semantic_vector):
    if sys_data["Y_train"] is None:
        return
    sims = cosine_similarity(semantic_vector.reshape(1, -1), sys_data["Y_train"])[0]
    top  = sims.argsort()[::-1][:3]
    print(f"\n📚 RECUERDOS:")
    for idx in top:
        word  = sys_data["concepts_train"][idx]
        score = sims[idx]
        assoc = []
        if sys_data["global_emb"] is not None:
            mem_t = torch.Tensor(sys_data["Y_train"][idx]).unsqueeze(0)
            with torch.no_grad():
                synth = sys_data["synthesizer"](mem_t).numpy()
            d_s = cosine_similarity(synth, sys_data["global_emb"])[0]
            d_i = d_s.argsort()[::-1][:5]
            assoc = [sys_data["global_words"][i] for i in d_i
                     if sys_data["global_words"][i].lower() != word.lower()][:3]
        print(f"   • {word.upper():<20} ({score:.3f})  🔮 {', '.join(assoc)}")


# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    sys_data  = load_cognitive_system()
    config    = {"alpha": ALPHA}

    print("\n" + "="*60)
    print(f"  SISTEMA COGNITIVO v3 — Llama3")
    print(f"  Modo: {MODE.upper()}  |  Modelo: {OLLAMA_MODEL if MODE == 'ollama' else GGUF_PATH}")
    print("  Escribe 'salir' para terminar")
    print("="*60)

    while True:
        user_input = input("\n📝 QUERY >> ").strip()
        if user_input.lower() in ["salir", "exit", "q"]:
            break
        if not user_input:
            continue

        # Traducción semántica
        raw_emb    = sys_data["embedder"].encode(user_input)
        tensor_emb = torch.Tensor(raw_emb).unsqueeze(0)
        with torch.no_grad():
            sem_vec_np = sys_data["translator"](tensor_emb).numpy()[0]

        print(f"\n📝 QUERY: «{user_input}»")
        print("─" * 55)
        print_profile(sem_vec_np, sys_data["dim_names"])
        print_memories(sys_data, sem_vec_np)

        # Ejecutar según modo
        if MODE == "llamacpp":
            run_llamacpp(sys_data, sem_vec_np, user_input, config)
        else:
            run_ollama(sys_data, sem_vec_np, user_input, config)

        # Ajuste de alpha (solo relevante en llamacpp)
        if MODE == "llamacpp":
            print(f"\n💡 Alpha: {config['alpha']} | Nuevo valor o Enter:")
            a = input("   α >> ").strip()
            if a:
                try:
                    config["alpha"] = float(a)
                    print(f"   ✅ Alpha → {config['alpha']}")
                except ValueError:
                    pass

        print()


if __name__ == "__main__":
    main()
