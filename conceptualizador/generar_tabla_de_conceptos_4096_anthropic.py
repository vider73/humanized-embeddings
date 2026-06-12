import json
import os
import time
import re
from openai import OpenAI
from typing import List, Dict, Set

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FILE = "d_semantic_universe.json"
REJECTED_FILE = "d_rejected_duplicates.json"
TOTAL_TARGET_DIMENSIONS = 300  # Empieza con 300, escala después
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 0.5    # Segundos entre llamadas normales
ORTHOGONALITY_SAMPLE_SIZE = 30 # Cuántas dims existentes revisar para duplicados

API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL = "gemma3:12b"

# ==========================================
# TAXONOMY MEJORADA
# Incluye objetos, seres vivos, lugares
# ==========================================
TAXONOMY = {
    "Fundamental Physics": [
        "Quantum Mechanics", "Thermodynamics", "Astrophysics",
        "Optics", "Gravity", "Electromagnetism"
    ],
    "Matter and Chemistry": [
        "States of Matter", "Reactions", "Materials", "Toxicity"
    ],
    "Biology and Life": [
        "Physiology", "Genetics", "Evolution",
        "Ecology", "Health", "Instincts"
    ],
    "Sensory Perception": [
        "Visual", "Auditory", "Tactile",
        "Olfactory and Taste", "Temperature and Pain"
    ],
    "Physical Objects": [
        "Tools and Weapons", "Food and Drink", "Furniture and Structures",
        "Vehicles", "Clothing and Containers"
    ],
    "Living Entities": [
        "Wild Animals", "Domestic Animals", "Plants and Fungi",
        "Human Roles", "Microorganisms"
    ],
    "Space and Place": [
        "Natural Landscapes", "Urban Environments",
        "Geographic Scale", "Interior Spaces", "Cosmic Scale"
    ],
    "Identity and Proper Nouns": [
        "Historical Persons", "Living Persons",
        "Place Names", "Brands and Products", "Mythological Entities"
    ],
    "Psychology and Mind": [
        "Basic Emotions", "Complex Emotions", "Cognition",
        "Memory", "Personality", "Mental Disorders"
    ],
    "Sociology and Culture": [
        "Hierarchy and Power", "Politics", "Economy",
        "Religion", "Tradition", "Crime", "Fame"
    ],
    "Ethics and Morality": [
        "Virtues", "Vices", "Justice", "Responsibility"
    ],
    "Aesthetics and Art": [
        "Beauty", "Artistic Styles", "Music", "Narrative"
    ],
    "Technology and Artificiality": [
        "Computing", "Mechanics", "Digitalization", "Industrialization"
    ],
    "Metaphysics and Abstract": [
        "Time", "Existence", "Logic", "Truth", "Causality"
    ],
    "Action and Dynamics": [
        "Movement and Speed", "Creation and Destruction",
        "Communication", "Control and Agency"
    ]
}

# ==========================================
# SUBCATEGORÍAS QUE USAN ESCALA LOGARÍTMICA
# Solo física pura, no biología
# ==========================================
LOGARITHMIC_SUBCATEGORIES = {
    "Quantum Mechanics", "Thermodynamics", "Astrophysics",
    "Optics", "Gravity", "Electromagnetism",
    "States of Matter", "Reactions", "Cosmic Scale",
    "Geographic Scale"
}

# ==========================================
# ESCALAS DE REFERENCIA
# ==========================================
LOGARITHMIC_ANCHORS = """
    [0.0] -> Subatomic Level / Planck Scale
    [0.2] -> Molecular Level / Nanoscopic
    [0.4] -> Human Level / Everyday Object
    [0.6] -> Geographic Level / Planetary
    [0.8] -> Stellar Level / Solar System
    [1.0] -> Galactic Level / Universal
"""

def get_scale_instruction(subcategory: str, category: str) -> str:
    if subcategory in LOGARITHMIC_SUBCATEGORIES:
        return f"""
    SCALE TYPE: LOGARITHMIC (orders of magnitude).
    Use this anchor table in the llama3_prompt:
    {LOGARITHMIC_ANCHORS}
    """
    elif category in ("Physical Objects", "Living Entities", "Space and Place", "Identity and Proper Nouns"):
        return """
    SCALE TYPE: COMPARATIVE within category.
    The anchors MUST use concrete examples of objects, animals, or places.
    Example for size: 0.0=ant, 0.5=dog, 1.0=elephant
    Example for domestication: 0.0=wild wolf, 0.5=cat, 1.0=lapdog
    The LLM must be able to score a specific animal, object or place, not just abstract concepts.
    """
    else:
        return """
    SCALE TYPE: LINEAR qualitative.
    Clearly define 0.0 (minimum), 0.5 (midpoint) and 1.0 (maximum) with concrete examples.
    """

# ==========================================
# CLIENTE LLM
# ==========================================
def _raw_query(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    """Llamada base al LLM. Devuelve el JSON parseado sin normalizar."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    system_enforced = system_prompt + "\n\nCRITICAL: Return VALID JSON ONLY. No markdown. No comments. No extra text."

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_enforced},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = response.choices[0].message.content.strip()

            if "```" in content:
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
                content = content.strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON inválido (intento {attempt+1}/3): {e}")
        except Exception as e:
            print(f"  ⚠️ Error de conexión (intento {attempt+1}/3): {e}")
            time.sleep(2)

    return None


def query_llm_dimensions(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    """Para generación de dimensiones. Espera {"dimensions": [...]}"""
    parsed = _raw_query(system_prompt, user_prompt, temperature)

    if parsed is None:
        print("  ❌ Fallaron los 3 intentos. Saltando batch.")
        return {"dimensions": []}

    if isinstance(parsed, list):
        return {"dimensions": parsed}
    elif "dimensions" in parsed:
        return parsed
    else:
        print(f"  ⚠️ JSON válido pero sin clave 'dimensions': {list(parsed.keys())}")
        return {"dimensions": []}


def query_llm_validation(system_prompt: str, user_prompt: str) -> dict:
    """Para validación de ortogonalidad. Espera {"redundant": bool, "similar_to": str}"""
    parsed = _raw_query(system_prompt, user_prompt, temperature=0.1)

    if parsed is None:
        # Si falla la validación, asumimos no redundante para no bloquear
        return {"redundant": False, "similar_to": None}

    # Puede venir directo o anidado
    if "redundant" in parsed:
        return parsed
    # Algunos modelos anidan la respuesta
    for v in parsed.values():
        if isinstance(v, dict) and "redundant" in v:
            return v

    # Si no encontramos la estructura esperada, no bloqueamos
    return {"redundant": False, "similar_to": None}


# ==========================================
# VALIDACIÓN DE ORTOGONALIDAD
# ==========================================
def is_redundant(new_dim_name: str, existing_dims: List[dict]) -> tuple[bool, str]:
    if not existing_dims:
        return False, ""

    sample = existing_dims[-ORTHOGONALITY_SAMPLE_SIZE:]
    sample_names = [d["name"] for d in sample]

    system_prompt = """You are a semantic ontology validator. 
    Detect redundant or highly similar dimensions.
    Return ONLY valid JSON."""

    user_prompt = f"""
    New dimension proposed: "{new_dim_name}"
    Existing dimensions (sample): {json.dumps(sample_names)}
    
    Is the new dimension semantically redundant or very similar to any existing one?
    Consider redundant if they would produce nearly identical scores for the same concept.
    
    Return: {{"redundant": true or false, "similar_to": "name of similar dimension or null"}}
    """

    result = query_llm_validation(system_prompt, user_prompt)
    return bool(result.get("redundant", False)), str(result.get("similar_to", "") or "")


# ==========================================
# GENERACIÓN DE BATCH
# ==========================================
def generate_batch(
    category: str,
    subcategory: str,
    start_id: int,
    batch_size: int,
    existing_dims: List[dict],
    existing_names: Set[str]
) -> List[dict]:

    scale_instruction = get_scale_instruction(subcategory, category)

    system_prompt = f"""
    You are an Architect of Semantic Ontologies designing measurement dimensions for an AI embedding system.
    Category: '{category}' > Subcategory: '{subcategory}'
    
    Generate exactly {batch_size} dimension objects. Each object must have:
    
    1. "name": Short, precise dimension name (e.g., "Domestication Level", "Predatory Nature")
    2. "scale": {{"min": 0.0, "max": 1.0, "min_label": "description", "max_label": "description"}}
    3. "llama3_prompt": The EXACT prompt for Llama 3 to score a concept on this dimension.
    
    {scale_instruction}
    
    FORMAT FOR llama3_prompt:
    - Start with: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are an instrument measuring [dimension]..."
    - Include: "REFERENCE SCALE - [Name]: [anchors with concrete examples]"
    - Include: "Return ONLY a float between 0.0 and 1.0. No explanation."
    - End with: "Score for '{{concept}}':<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
    
    IMPORTANT:
    - Dimensions must be ORTHOGONAL (non-redundant with each other)
    - Each dimension must be scoreable for ANY concept: abstract, concrete object, animal, place or person
    - Avoid these already-existing names: {[d['name'] for d in existing_dims[-20:]]}
    
    Return JSON: {{"dimensions": [{{...}}, ...]}}
    """

    user_prompt = f"Generate {batch_size} unique, orthogonal, and precise dimensions for '{subcategory}'. Be creative and specific."

    response = query_llm_dimensions(system_prompt, user_prompt)
    raw_dims = response.get("dimensions", [])

    validated = []
    rejected = []

    for dim in raw_dims:
        name = dim.get("name", "").strip()
        if not name:
            continue

        name_lower = name.lower()

        # Filtro 1: duplicado exacto por nombre
        if name_lower in existing_names:
            print(f"    ⚠️  Duplicado exacto: '{name}' — descartado")
            rejected.append({"name": name, "reason": "exact_duplicate"})
            continue

        # Filtro 2: redundancia semántica vía LLM
        redundant, similar_to = is_redundant(name, existing_dims + validated)
        if redundant:
            print(f"    ⚠️  Redundante con '{similar_to}': '{name}' — descartado")
            rejected.append({"name": name, "reason": f"redundant_with_{similar_to}"})
            continue

        # Construir objeto final
        dim_id = f"d{str(start_id + len(validated)).zfill(4)}_{name_lower.replace(' ', '_')}"
        formatted = {
            "id": dim_id,
            "category_index": start_id + len(validated),
            "category_group": category,
            "subcategory": subcategory,
            "name": name,
            "scale": dim.get("scale", {"min": 0.0, "max": 1.0}),
            "llama3_prompt": dim.get("llama3_prompt", "")
        }
        validated.append(formatted)
        existing_names.add(name_lower)
        print(f"    ✅ '{name}'")

    return validated, rejected


# ==========================================
# GUARDADO INCREMENTAL
# ==========================================
def save_incremental(all_dims: List[dict], rejected: List[dict]):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_dims, f, indent=2, ensure_ascii=False)
    with open(REJECTED_FILE, 'w', encoding='utf-8') as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)


# ==========================================
# MAIN
# ==========================================
def main():
    all_dims: List[dict] = []
    all_rejected: List[dict] = []
    existing_names: Set[str] = set()
    global_id = 0

    # Retomar desde archivo si existe
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            all_dims = json.load(f)
        global_id = len(all_dims)
        existing_names = {d["name"].lower() for d in all_dims}
        print(f"▶ Retomando desde {global_id} dimensiones existentes.")

    total_subcats = sum(len(v) for v in TAXONOMY.values())
    dims_per_subcat = max(1, TOTAL_TARGET_DIMENSIONS // total_subcats)

    print(f"\n🎯 Objetivo: {TOTAL_TARGET_DIMENSIONS} dimensiones")
    print(f"📂 Subcategorías: {total_subcats}")
    print(f"📐 Dimensiones por subcategoría: ~{dims_per_subcat}")
    print(f"🤖 Modelo: {MODEL}\n")

    for category, subcategories in TAXONOMY.items():
        for subcat in subcategories:

            # Contar cuántas ya tenemos de esta subcategoría
            already_have = sum(1 for d in all_dims if d["subcategory"] == subcat)
            dims_needed = dims_per_subcat - already_have

            if dims_needed <= 0:
                print(f"⏭  {category} > {subcat}: ya completo ({already_have} dims)")
                continue

            print(f"\n🔧 {category} > {subcat} (necesito {dims_needed} más)")

            while dims_needed > 0:
                batch_size = min(BATCH_SIZE, dims_needed)

                new_dims, rejected = generate_batch(
                    category=category,
                    subcategory=subcat,
                    start_id=global_id,
                    batch_size=batch_size,
                    existing_dims=all_dims,
                    existing_names=existing_names
                )

                if new_dims:
                    all_dims.extend(new_dims)
                    all_rejected.extend(rejected)
                    global_id += len(new_dims)
                    dims_needed -= len(new_dims)
                    save_incremental(all_dims, all_rejected)
                    print(f"  📊 Total acumulado: {len(all_dims)} dimensiones")
                else:
                    print("  ❌ Batch vacío tras validación, reintentando subcategoría...")
                    break

                time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\n✅ Proceso completado.")
    print(f"   Dimensiones generadas: {len(all_dims)}")
    print(f"   Dimensiones rechazadas: {len(all_rejected)}")
    print(f"   Archivo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
