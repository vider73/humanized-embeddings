import numpy as np
import json
import os

# CONFIGURACIÓN
METADATA_FILE = "dataset_metadata.json"
HUMAN_VECTORS_DB = "dataset_Y_human.npy"

def main():
    print("🩺 CHEQUEO DE SIGNOS VITALES DEL DATASET")
    
    if not os.path.exists(HUMAN_VECTORS_DB):
        print("❌ Faltan archivos.")
        return

    # Cargar datos
    Y = np.load(HUMAN_VECTORS_DB) # Matriz (Palabras x Dimensiones)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    dim_names = meta["dimension_names"]
    concepts = meta["concepts"]
    
    print(f"📊 Analizando {Y.shape[0]} palabras y {Y.shape[1]} dimensiones...\n")

    # 1. BUSCAR DIMENSIONES MUERTAS (Sin varianza)
    print("💀 DIMENSIONES 'ZOMBIE' (Casi todo es 0 o constante):")
    zombies = []
    
    for i, dim in enumerate(dim_names):
        col = Y[:, i]
        variance = np.var(col)
        max_val = np.max(col)
        min_val = np.min(col)
        mean_val = np.mean(col)
        
        # Si la varianza es extremadamente baja, la dimensión está muerta
        if variance < 0.001:
            zombies.append((dim, variance, max_val))
            # print(f"   - {dim[:30]:<30} | Var: {variance:.6f} | Max: {max_val:.6f}")

    # Ordenar por las más muertas
    zombies.sort(key=lambda x: x[1])
    for z in zombies[:10]:
         print(f"   ❌ {z[0][:30]:<30} | Max Real: {z[2]:.6f} (Nadie supera esto)")

    print(f"\n   Total Dimensiones Muertas (<0.001 var): {len(zombies)} de {len(dim_names)}")

    # 2. VER EL RANKING REAL DE UNA DIMENSIÓN
    # Vamos a ver qué palabras son las "Reyes" de la Peligrosidad según tu dataset actual
    target_dim = "peligrosidad_social" # Busca parcial
    
    found_idx = -1
    for i, name in enumerate(dim_names):
        if target_dim in name:
            found_idx = i
            print(f"\n🔎 INSPECCIONANDO: '{name.upper()}'")
            break
            
    if found_idx != -1:
        # Obtener valores de esa columna
        col = Y[:, found_idx]
        
        # Ordenar índices de mayor a menor
        top_indices = np.argsort(col)[::-1]
        
        print("   🏆 TOP 10 MÁXIMOS VALORES (¿Tienen sentido?):")
        for idx in top_indices[:10]:
            print(f"      {col[idx]:.4f} -> {concepts[idx]}")
            
        print("\n   📉 TOP 5 MÍNIMOS VALORES:")
        for idx in top_indices[-5:]:
            print(f"      {col[idx]:.4f} -> {concepts[idx]}")
            
        print(f"\n   Estadísticas: Promedio={np.mean(col):.4f} | Mediana={np.median(col):.4f}")
        
        if np.max(col) < 0.1:
            print("\n🚨 ALERTA CRÍTICA: El valor máximo es bajísimo.")
            print("   La red neuronal piensa que 'Peligro 0.9' es imposible porque nunca vio nada mayor a 0.1.")
            print("   SOLUCIÓN: Necesitas re-entrenar normalizando los datos.")

if __name__ == "__main__":
    main()