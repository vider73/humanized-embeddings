import json
import os
import time
import re
from openai import OpenAI
from typing import List, Dict

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FILE = "d4096_semantic_universe_llama.json"
TOTAL_TARGET_DIMENSIONS = 4096
BATCH_SIZE = 10  # Generate 10 dimensions per call to maintain high quality

# API Client Simulation (Replace this with your real call to OpenAI/Ollama)
# If using Ollama locally, change the query_llm function parameters if needed.
API_KEY = "ollama"  # Not strictly needed for local Ollama, but the library requires a value
BASE_URL = "http://localhost:11434/v1" # Default Ollama port
MODEL = "gemma3:12b" # Or the exact name you have in 'ollama list'

# ==========================================
# MASTER TAXONOMY (Seed for diversity)
# ==========================================
# We distribute the 4096 dimensions across areas to ensure total coverage.
TAXONOMY = {
    "Fundamental Physics": ["Quantum Mechanics", "Thermodynamics", "Astrophysics", "Optics", "Gravity", "Electromagnetism"],
    "Matter and Chemistry": ["States of Matter", "Periodic Table", "Reactions", "Materials", "Toxicity"],
    "Biology and Life": ["Physiology", "Genetics", "Evolution", "Ecology", "Health", "Instincts"],
    "Sensory Perception": ["Visual (Color/Light)", "Auditory (Sound)", "Tactile (Texture)", "Olfactory/Taste", "Proprioception"],
    "Psychology and Mind": ["Basic Emotions", "Complex Emotions", "Cognition", "Memory", "Personality", "Disorders"],
    "Sociology and Culture": ["Hierarchy", "Politics", "Economy", "Religion", "Tradition", "Crime", "Fame"],
    "Ethics and Morality": ["Virtues", "Vices", "Moral Dilemmas", "Justice", "Responsibility"],
    "Aesthetics and Art": ["Beauty", "Artistic Styles", "Music", "Narrative", "Symbolism"],
    "Technology and Artificiality": ["Computing", "Mechanics", "Digitalization", "Futurism", "Industrialization"],
    "Metaphysics and Abstract": ["Time (Philosophical)", "Existence", "Logic", "Causality", "Truth"],
    "Action and Verbs": ["Movement", "Interaction", "Creation", "Destruction", "Communication"]
}

# ==========================================
# MASTER PROMPTS (LOGARITHMIC vs LINEAR)
# ==========================================

def get_logarithmic_instruction():
    return """
    IMPORTANT: This is a PHYSICAL dimension that covers massive scales.
    You must instruct the use of a LOGARITHMIC SCALE (Orders of Magnitude).
    
    Generate the field 'llama3_prompt' explicitly including this anchor table in the prompt text:
    [0.0] -> Subatomic Level / Planck
    [0.2] -> Molecular Level / Nanoscopic
    [0.4] -> Human Level / Everyday Object
    [0.6] -> Geographic Level / Planetary
    [0.8] -> Stellar Level / Solar System
    [1.0] -> Galactic Level / Universal
    """

def get_linear_instruction():
    return """
    This is a QUALITATIVE or LINEAR dimension.
    Generate the field 'llama3_prompt' clearly defining the absolute extremes (0.0 and 1.0) and a reference midpoint (0.5).
    """

# ==========================================
# GENERATION LOGIC
# ==========================================

def query_llm(system_prompt, user_prompt):
    """
    Implementation for Llama 3 (70B) via Ollama/Localhost.
    Includes Markdown cleaning and format error handling.
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    # Llama 3 needs to be reminded to be strict with JSON in the system prompt
    system_prompt_enforced = system_prompt + "\n\nCRITICAL: RESPONSE MUST BE VALID JSON ONLY. NO MARKDOWN. NO COMMENTS."

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt_enforced},
                    {"role": "user", "content": user_prompt}
                ],
                # Some local backends support json_object, others don't.
                # If you get an error, remove the 'response_format' line.
                response_format={"type": "json_object"}, 
                temperature=0.7,
            )
            
            content = response.choices[0].message.content.strip()
            
            # --- MARKDOWN CLEANING (Essential for Llama 3) ---
            # If the model returns ```json ... ```, remove it.
            if "```" in content:
                content = re.sub(r"```json\s*", "", content) # Remove start
                content = re.sub(r"```\s*$", "", content)    # Remove end
                content = content.strip()
            
            # Attempt to parse
            parsed_json = json.loads(content)
            
            # Normalization: Ensure we return a dictionary with "dimensions"
            if isinstance(parsed_json, list):
                return {"dimensions": parsed_json}
            elif "dimensions" in parsed_json:
                return parsed_json
            else:
                # If it returns a dict but without the 'dimensions' key, try to infer
                # or return error to retry
                print(f"  ⚠️ Valid JSON but incorrect structure. Retrying...")
                continue
                
        except json.JSONDecodeError:
            print(f"  ⚠️ JSON Syntax Error (Model hallucinated text). Retrying...")
        except Exception as e:
            print(f"  ⚠️ Local Connection Error: {e}. Is Ollama running?")
            time.sleep(2) # Short wait
            
    print(f"  ❌ Failed after {max_retries} attempts. Skipping batch.")
    return {"dimensions": []}

def generate_batch(category, subcategory, start_id_index):
    # Check against the English keys defined in TAXONOMY
    is_physical = category in ["Fundamental Physics", "Matter and Chemistry", "Biology and Life"]
    
    scale_instruction = get_logarithmic_instruction() if is_physical else get_linear_instruction()
    
    system_prompt = f"""
    You are an Architect of Semantic Ontologies. Your task is to define precise measurement dimensions for an AI.
    You are working in the category: '{category}' > Subcategory: '{subcategory}'.
    
    You must generate a JSON with {BATCH_SIZE} objects. Each object must have:
    1. 'name': Name of the dimension (e.g., "Opacity", "Empathy").
    2. 'scale': Object with 'min' and 'max' describing the extremes.
    3. 'llama3_prompt': The EXACT prompt that will be given to Llama 3 to measure this.
    
    {scale_instruction}
    
    FORMAT OF 'llama3_prompt':
    Must start with: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are an instrument..."
    Must include: "REFERENCE SCALE - [Name]:" followed by the list of anchors.
    Must end with: "Score for '{{concept}}':<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
    """

    user_prompt = f"Generate {BATCH_SIZE} unique and orthogonal dimensions for '{subcategory}'. Do not repeat generic concepts."

    try:
        response_data = query_llm(system_prompt, user_prompt)
        dimensions = response_data.get("dimensions", [])
        
        formatted_dimensions = []
        current_id = start_id_index
        
        for dim in dimensions:
            dim_id = f"d{str(current_id).zfill(4)}_{dim['name'].lower().replace(' ', '_')}"
            
            # Final object structure
            formatted_obj = {
                "id": dim_id,
                "category_index": current_id,
                "category_group": category,
                "subcategory": subcategory,
                "name": dim['name'],
                "scale": dim['scale'],
                "llama3_prompt": dim['llama3_prompt']
            }
            formatted_dimensions.append(formatted_obj)
            current_id += 1
            
        return formatted_dimensions

    except Exception as e:
        print(f"Error generating batch: {e}")
        return []

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    all_dimensions = []
    global_id_counter = 0
    
    # Calculate how many dimensions per subcategory we need
    # Total flattened categories:
    total_subcats = sum(len(v) for v in TAXONOMY.values())
    dims_per_subcat = TOTAL_TARGET_DIMENSIONS // total_subcats
    
    print(f"Target: {TOTAL_TARGET_DIMENSIONS} dimensions.")
    print(f"Total Subcategories: {total_subcats}")
    print(f"Dimensions per subcategory: ~{dims_per_subcat}")
    
    for category, subcategories in TAXONOMY.items():
        for subcat in subcategories:
            print(f"\nProcessing: {category} -> {subcat}")
            
            # Generate in small loops to avoid saturating LLM context
            dims_needed = dims_per_subcat
            while dims_needed > 0:
                current_batch_size = min(BATCH_SIZE, dims_needed)
                
                # Call the AI here
                new_dims = generate_batch(category, subcat, global_id_counter)
                
                if new_dims:
                    all_dimensions.extend(new_dims)
                    global_id_counter += len(new_dims)
                    dims_needed -= len(new_dims)
                    print(f"  + Generated {len(new_dims)} dimensions. Accumulated total: {len(all_dimensions)}")
                else:
                    print("  ! Generation failed, retrying or skipping...")
                    break
                
                # Incremental save for safety
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_dimensions, f, indent=2, ensure_ascii=False)

    print(f"\nProcess finished! File saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()