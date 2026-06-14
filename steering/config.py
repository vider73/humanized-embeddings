"""
config.py — Parameters of the semantic steering engine.
A single place to tweak paths, target layer and injection strength.

MULTI-MODEL OVERRIDES (for the periodic-table experiments): every steering
tool reads this module, so overriding via environment variables makes the
whole toolchain model-agnostic without touching any tool:
  EMB_LLM    = HF model name           (default: Llama-3.1-8B-Instruct)
  EMB_TAG    = artifact suffix         (vectors/reports become *_<tag>.*)
  EMB_LAYERS = "a-b" derive band       (default: 12-19)
  EMB_INJECT = injection layer         (default: middle of the band / 15)
  EMB_4BIT   = "1"/"0"                 (default: 1)
With no env set, everything behaves exactly as before.
"""
import os
from pathlib import Path

# --- Roots -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../embtoconcept
VEC_DIR = Path(__file__).resolve().parent / "vectors"  # control vectors output
VEC_DIR.mkdir(exist_ok=True)

# --- Humanizer artifacts (already existing) ---------------------------------
TRANSLATOR_PATH = ROOT / "semantic_translator.pth"     # MiniLM(384) -> 104 dims
METADATA_FILE   = ROOT / "dataset_metadata.json"       # concepts + dimension_names
DATA_Y_FILE     = ROOT / "dataset_Y_human.npy"         # (N, 104) values per concept

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Target LLM (HuggingFace, runs on your 4090) ---------------------------
# meta-llama is "gated": accept the license on HF and do `huggingface-cli login`,
# or use an open mirror: "NousResearch/Meta-Llama-3.1-8B-Instruct"
LLM_NAME   = os.environ.get("EMB_LLM", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LOAD_4BIT  = os.environ.get("EMB_4BIT", "1") == "1"   # 4-bit fits in 24GB; "0" for fp16
# hidden_size and number of layers are read from the model at runtime (don't hardcode).
# These values are for reference only, for Llama-3.1-8B:
HIDDEN_DIM = 4096
N_LAYERS   = 32

# Tiny, OPEN model for the hooks smoke test (fast, even on CPU).
SMOKE_LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Llama architecture

# --- Where and how much to inject -------------------------------------------
# Layers for which control vectors are DERIVED (derive_vectors.py).
# Derive a wide band once; then choose which ones to inject into.
if os.environ.get("EMB_LAYERS"):                       # e.g. "5-9"
    _a, _b = os.environ["EMB_LAYERS"].split("-")
    TARGET_LAYERS = tuple(range(int(_a), int(_b) + 1))
else:
    TARGET_LAYERS = tuple(range(12, 20))   # layers 12..19 (the ones you already derived)

# Layers where injection ACTUALLY happens (subset of the derived ones).
# No re-derivation needed. VALIDATED 2026-06-11: ONE layer with clamp holds up to
# alpha ~0.35 where three layers compounded distortion and broke at ~0.25.
if os.environ.get("EMB_INJECT"):
    INJECT_LAYERS = (int(os.environ["EMB_INJECT"]),)
elif os.environ.get("EMB_LAYERS"):
    INJECT_LAYERS = (TARGET_LAYERS[len(TARGET_LAYERS) // 2],)   # middle of the band
else:
    INJECT_LAYERS = (15,)
# ALPHA is now a FRACTION of the residual norm at each position:
#   0.1  -> push ~10% of the local magnitude in the profile's direction.
# This way the push does NOT depend on the sentence nor on how many dims activate.
# Start around ~0.1 and go up; above ~0.6 it usually starts raving.
ALPHA = 0.12

# Injection mode:
#  "add"   = classic: h += alpha*|h|*v at each position (blind, fixed push).
#  "clamp" = P controller on the projection: h += KP*(alpha*|h| - h.v)*v.
#            Pushes only what is MISSING for h.v to reach the setpoint
#            (alpha*|h|). Where the model already complies, it adds nothing — that's
#            where add mode was breaking the grammar. With KP=1 it pins the exact
#            projection (Golden Gate Claude style); KP<1 softens it.
STEER_MODE = "clamp"   # validated: alpha ceiling ~2x compared to "add"
KP = 1.0

# --- Control vector derivation ----------------------------------------------
# "concepts" = difference of means of high/low concepts from the table (v1/v2,
#              confounded: drags in the table's global correlations).
# "caa"      = Contrastive Activation Addition: PURE contrast between the two
#              poles (min/max) of each dimension. The contrast shares nothing
#              except the sought axis -> sharp, strong directions.
DERIVE_MODE  = "caa"
K_CONTRAST   = 60          # (concepts mode) number of concepts per pole
POOLING      = "mean"      # "mean" over tokens, or "last"
# (concepts mode) templates to wrap the concept in.
TEMPLATES = (
    "{c}",
    "Esto trata sobre {c}.",
    "La idea de {c}.",
)
# (caa mode) molds that the pole's DESCRIPTION slots into. Used as FALLBACK
# when there are no generated stimuli. Short molds => lexical direction (breaks
# the language before evoking). That's why we prefer rich stimuli (see below).
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
# (caa mode, recommended) rich, varied sentences per pole. If the file
# exists, they are used instead of the molds -> SEMANTIC direction, not lexical.
# Generate it with:  python -m steering.generate_stimuli   (then hand-editable).
K_STIM       = 24          # sentences per pole
# EMB_STIMULI overrides the stimuli file (null_control.py uses a shuffled copy
# to test whether the method manufactures dials). Default = the real file.
STIMULI_FILE = (Path(os.environ["EMB_STIMULI"]) if os.environ.get("EMB_STIMULI")
                else VEC_DIR / "caa_stimuli.json")

# --- WHITENING (the rank-collapse fix) --------------------------------------
# Llama's activations are dominated by a few "outlier" coordinates
# (massive activations) that crush the difference of means down to a
# single axis. Per-coordinate z-score (dividing by the reference std)
# neutralizes them and lets each dimension's semantic signal breathe.
WHITEN = True
N_REF  = 800               # reference concepts to estimate the per-layer std

# File where the derived control vector matrix is stored.
# final shape: (n_target_layers, 104, HIDDEN_DIM)
_MODE = "_caa" if DERIVE_MODE == "caa" else ""
_SUF  = "_white" if WHITEN else ""
_TAG  = f"_{os.environ['EMB_TAG']}" if os.environ.get("EMB_TAG") else ""
CONTROL_VECTORS_FILE = VEC_DIR / f"control_vectors{_MODE}{_SUF}{_TAG}.npy"
CONTROL_META_FILE    = VEC_DIR / f"control_meta{_MODE}{_SUF}{_TAG}.json"
