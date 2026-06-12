"""
config.py — Parametros del motor de steering semantico.
Un unico sitio para tocar rutas, capa objetivo y fuerza de inyeccion.
"""
from pathlib import Path

# --- Raices ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../embtoconcept
VEC_DIR = Path(__file__).resolve().parent / "vectors"  # salida de control vectors
VEC_DIR.mkdir(exist_ok=True)

# --- Artefactos del humanizador (ya existentes) ----------------------------
TRANSLATOR_PATH = ROOT / "semantic_translator.pth"     # MiniLM(384) -> 104 dims
METADATA_FILE   = ROOT / "dataset_metadata.json"       # concepts + dimension_names
DATA_Y_FILE     = ROOT / "dataset_Y_human.npy"         # (N, 104) valores por concepto

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- LLM objetivo (HuggingFace, corre en tu 4090) --------------------------
# meta-llama esta "gated": acepta la licencia en HF y haz `huggingface-cli login`,
# o usa un mirror abierto: "NousResearch/Meta-Llama-3.1-8B-Instruct"
LLM_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOAD_4BIT  = True          # 4-bit cabe sobrado en 24GB; pon False para fp16
# hidden_size y nº de capas se leen del modelo en runtime (no hardcodear).
# Estos valores son solo de referencia para Llama-3.1-8B:
HIDDEN_DIM = 4096
N_LAYERS   = 32

# Modelo diminuto y ABIERTO para el smoke test de los hooks (rapido, hasta en CPU).
SMOKE_LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # arquitectura Llama

# --- Donde y cuanto inyectar ----------------------------------------------
# Capas para las que se DERIVAN control vectors (derive_vectors.py).
# Deriva una banda amplia una vez; luego eliges en cuales inyectar.
TARGET_LAYERS = tuple(range(12, 20))   # capas 12..19 (las que ya derivaste)

# Capas en las que REALMENTE se inyecta (subconjunto de las derivadas).
# No requiere re-derivar. VALIDADO 2026-06-11: UNA capa con clamp aguanta
# alpha ~0.35 donde tres capas componian distorsion y rompian a ~0.25.
INJECT_LAYERS = (15,)
# ALPHA ahora es una FRACCION de la norma del residual en cada posicion:
#   0.1  -> empujar ~10% de la magnitud local en la direccion del perfil.
# Asi el empuje NO depende de la frase ni de cuantas dims se activen.
# Arranca en ~0.1 y sube; sobre ~0.6 suele empezar a delirar.
ALPHA = 0.12

# Modo de inyeccion:
#  "add"   = clasico: h += alpha*|h|*v en cada posicion (empuje ciego, fijo).
#  "clamp" = controlador P sobre la proyeccion: h += KP*(alpha*|h| - h.v)*v.
#            Empuja solo lo que FALTA para que h.v alcance el setpoint
#            (alpha*|h|). Donde el modelo ya obedece, no anade — ahi es donde
#            el modo add rompia la gramatica. Con KP=1 fija la proyeccion
#            exacta (estilo Golden Gate Claude); KP<1 suaviza.
STEER_MODE = "clamp"   # validado: techo de alpha ~2x respecto a "add"
KP = 1.0

# --- Derivacion de control vectors ----------------------------------------
# "concepts" = diferencia de medias de conceptos alto/bajo de la tabla (v1/v2,
#              confundido: arrastra correlaciones globales de la tabla).
# "caa"      = Contrastive Activation Addition: contraste PURO entre los dos
#              polos (min/max) de cada dimension. El contraste no comparte nada
#              salvo el eje buscado -> direcciones nitidas y fuertes.
DERIVE_MODE  = "caa"
K_CONTRAST   = 60          # (modo concepts) nº de conceptos por polo
POOLING      = "mean"      # "mean" sobre tokens, o "last"
# (modo concepts) plantillas para envolver el concepto.
TEMPLATES = (
    "{c}",
    "Esto trata sobre {c}.",
    "La idea de {c}.",
)
# (modo caa) moldes que encajan la DESCRIPCION del polo. Se usan como FALLBACK
# si no hay estimulos generados. Moldes cortos => direccion lexica (rompe el
# idioma antes de evocar). Por eso preferimos estimulos ricos (ver abajo).
CAA_TEMPLATES = (
    "{p}",
    "Esto es {p}.",
    "Pienso en {p}.",
    "Imagina {p}.",
    "Hablo de {p}.",
    "Todo aqui evoca {p}.",
    "Una escena de {p}.",
    "La idea central es {p}.",
)
# (modo caa, recomendado) frases ricas y variadas por polo. Si el fichero
# existe, se usan en vez de los moldes -> direccion SEMANTICA, no lexica.
# Generalo con:  python -m steering.generate_stimuli   (luego editable a mano).
K_STIM       = 24          # frases por polo
STIMULI_FILE = VEC_DIR / "caa_stimuli.json"

# --- BLANQUEO (el arreglo del colapso de rango) ---------------------------
# Las activaciones de Llama estan dominadas por unas pocas coordenadas
# "outlier" (massive activations) que aplastan la diferencia de medias a un
# unico eje. Z-score por coordenada (dividir por la std de referencia) las
# neutraliza y deja respirar la senal semantica de cada dimension.
WHITEN = True
N_REF  = 800               # conceptos de referencia para estimar la std por capa

# Fichero donde se guarda la matriz de control vectors derivada.
# shape final: (n_target_layers, 104, HIDDEN_DIM)
_MODE = "_caa" if DERIVE_MODE == "caa" else ""
_SUF  = "_white" if WHITEN else ""
CONTROL_VECTORS_FILE = VEC_DIR / f"control_vectors{_MODE}{_SUF}.npy"
CONTROL_META_FILE    = VEC_DIR / f"control_meta{_MODE}{_SUF}.json"
