import json
import os
import time
from openai import OpenAI
from typing import List, Dict

# ==========================================
# CONFIGURACIÓN
# ==========================================
OUTPUT_FILE = "d4096_semantic_universe.json"
TOTAL_TARGET_DIMENSIONS = 4096
BATCH_SIZE = 10  # Generar 10 dimensiones por llamada para mantener la calidad alta

# Simulación de cliente API (Reemplaza esto con tu llamada real a OpenAI/Ollama)
# Si usas Ollama localmente, cambia la función query_llm.
API_KEY = "tu-api-key-aqui" 
MODEL = "gpt-4-turbo" # O "llama3:70b" si usas local

# ==========================================
# TAXONOMÍA MAESTRA (Semilla para la diversidad)
# ==========================================
# Distribuimos las 4096 dimensiones en áreas para asegurar cobertura total.
TAXONOMY = {
    "Física Fundamental": ["Mecánica Cuántica", "Termodinámica", "Astrofísica", "Óptica", "Gravedad", "Electromagnetismo"],
    "Materia y Química": ["Estados de la materia", "Tabla Periódica", "Reacciones", "Materiales", "Toxicidad"],
    "Biología y Vida": ["Fisiología", "Genética", "Evolución", "Ecología", "Salud", "Instintos"],
    "Percepción Sensorial": ["Visual (Color/Luz)", "Auditiva (Sonido)", "Táctil (Textura)", "Olfato/Gusto", "Propiocepción"],
    "Psicología y Mente": ["Emociones Básicas", "Emociones Complejas", "Cognición", "Memoria", "Personalidad", "Trastornos"],
    "Sociología y Cultura": ["Jerarquía", "Política", "Economía", "Religión", "Tradición", "Crimen", "Fama"],
    "Ética y Moral": ["Virtudes", "Vicios", "Dilemas Morales", "Justicia", "Responsabilidad"],
    "Estética y Arte": ["Belleza", "Estilos Artísticos", "Música", "Narrativa", "Simbolismo"],
    "Tecnología y Artificialidad": ["Computación", "Mecánica", "Digitalización", "Futurismo", "Industrialización"],
    "Metafísica y Abstracto": ["Tiempo (Filosófico)", "Existencia", "Lógica", "Causalidad", "Verdad"],
    "Acción y Verbos": ["Movimiento", "Interacción", "Creación", "Destrucción", "Comunicación"]
}

# ==========================================
# PROMPTS MAESTROS (LOGARÍTMICO vs LINEAL)
# ==========================================

def get_logarithmic_instruction():
    return """
    IMPORTANTE: Esta es una dimensión FÍSICA que abarca escalas masivas.
    Debes instruir el uso de una ESCALA LOGARÍTMICA (Órdenes de Magnitud).
    
    Genera el campo 'llama3_prompt' incluyendo explícitamente esta tabla de anclajes en el texto del prompt:
    [0.0] -> Nivel Subatómico / Planck
    [0.2] -> Nivel Molecular / Nanoscópico
    [0.4] -> Nivel Humano / Objeto cotidiano
    [0.6] -> Nivel Geográfico / Planetario
    [0.8] -> Nivel Estelar / Sistema Solar
    [1.0] -> Nivel Galáctico / Universal
    """

def get_linear_instruction():
    return """
    Esta es una dimensión CUALITATIVA o LINEAL.
    Genera el campo 'llama3_prompt' definiendo claramente los extremos absolutos (0.0 y 1.0) y un punto medio de referencia (0.5).
    """

# ==========================================
# LÓGICA DE GENERACIÓN
# ==========================================

def query_llm(system_prompt, user_prompt):
    """
    Implementación real para OpenAI (v1.x) con soporte para JSON Mode
    y reintentos automáticos en caso de fallo de red.
    """
    client = OpenAI(api_key=API_KEY)
    
    # Intentar hasta 3 veces si hay error de conexión
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,  # Recomendado: "gpt-4-turbo" o "gpt-4o"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # 'json_object' fuerza al modelo a devolver JSON válido, vital para este script
                response_format={"type": "json_object"}, 
                temperature=0.7, # Creatividad suficiente para no repetir dimensiones
            )
            
            content = response.choices[0].message.content
            
            # Verificación extra: a veces devuelve el JSON dentro de una clave extraña
            parsed_json = json.loads(content)
            
            # Aseguramos que tenga la clave "dimensions", si no, devolvemos el json tal cual
            if "dimensions" not in parsed_json and isinstance(parsed_json, list):
                 return {"dimensions": parsed_json}
            
            return parsed_json

        except Exception as e:
            wait_time = (attempt + 1) * 2  # Espera 2s, 4s, 6s...
            print(f"  ⚠️ Error API (Intento {attempt+1}/{max_retries}): {e}. Reintentando en {wait_time}s...")
            time.sleep(wait_time)
            
    # Si falla 3 veces, devolvemos lista vacía para no romper el bucle principal
    print(f"  ❌ Error fatal en este lote tras {max_retries} intentos. Saltando...")
    return {"dimensions": []}

def generate_batch(category, subcategory, start_id_index):
    is_physical = category in ["Física Fundamental", "Materia y Química", "Biología y Vida"]
    
    scale_instruction = get_logarithmic_instruction() if is_physical else get_linear_instruction()
    
    system_prompt = f"""
    Eres un Arquitecto de Ontologías Semánticas. Tu tarea es definir dimensiones de medición precisas para una IA.
    Estás trabajando en la categoría: '{category}' > Subcategoría: '{subcategory}'.
    
    Debes generar un JSON con {BATCH_SIZE} objetos. Cada objeto debe tener:
    1. 'name': Nombre de la dimensión (ej: "Opacidad", "Empatía").
    2. 'scale': Objeto con 'min' y 'max' describiendo los extremos.
    3. 'llama3_prompt': El prompt EXACTO que se le dará a Llama 3 para medir esto.
    
    {scale_instruction}
    
    FORMATO DEL 'llama3_prompt':
    Debe comenzar con: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nEres un instrumento..."
    Debe incluir: "ESCALA DE REFERENCIA - [Nombre]:" seguido de la lista de anclajes.
    Debe terminar con: "Puntuación para '{{concepto}}':<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
    """

    user_prompt = f"Genera {BATCH_SIZE} dimensiones únicas y ortogonales para '{subcategory}'. No repitas conceptos genéricos."

    try:
        response_data = query_llm(system_prompt, user_prompt)
        dimensions = response_data.get("dimensions", [])
        
        formatted_dimensions = []
        current_id = start_id_index
        
        for dim in dimensions:
            dim_id = f"d{str(current_id).zfill(4)}_{dim['name'].lower().replace(' ', '_')}"
            
            # Estructura final del objeto
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
        print(f"Error generando batch: {e}")
        return []

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

def main():
    all_dimensions = []
    global_id_counter = 0
    
    # Calcular cuántas dimensiones por subcategoría necesitamos
    # Total categorías aplanadas:
    total_subcats = sum(len(v) for v in TAXONOMY.values())
    dims_per_subcat = TOTAL_TARGET_DIMENSIONS // total_subcats
    
    print(f"Objetivo: {TOTAL_TARGET_DIMENSIONS} dimensiones.")
    print(f"Subcategorías totales: {total_subcats}")
    print(f"Dimensiones por subcategoría: ~{dims_per_subcat}")
    
    for category, subcategories in TAXONOMY.items():
        for subcat in subcategories:
            print(f"\nProcesando: {category} -> {subcat}")
            
            # Generamos en bucles pequeños para no saturar el contexto del LLM
            dims_needed = dims_per_subcat
            while dims_needed > 0:
                current_batch_size = min(BATCH_SIZE, dims_needed)
                
                # Aquí llamamos a la IA
                new_dims = generate_batch(category, subcat, global_id_counter)
                
                if new_dims:
                    all_dimensions.extend(new_dims)
                    global_id_counter += len(new_dims)
                    dims_needed -= len(new_dims)
                    print(f"  + Generadas {len(new_dims)} dimensiones. Total acumulado: {len(all_dimensions)}")
                else:
                    print("  ! Fallo en generación, reintentando o saltando...")
                    break
                
                # Guardado incremental por seguridad
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_dimensions, f, indent=2, ensure_ascii=False)

    print(f"\n¡Proceso finalizado! Archivo guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()