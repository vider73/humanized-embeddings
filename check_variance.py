import numpy as np
import json
import os

# CONFIGURATION
METADATA_FILE = "dataset_metadata.json"
HUMAN_VECTORS_DB = "dataset_Y_human.npy"

def main():
    print("🩺 CHEQUEO DE SIGNOS VITALES DEL DATASET")
    
    if not os.path.exists(HUMAN_VECTORS_DB):
        print("❌ Faltan archivos.")
        return

    # Load data
    Y = np.load(HUMAN_VECTORS_DB) # Matrix (Words x Dimensions)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    dim_names = meta["dimension_names"]
    concepts = meta["concepts"]
    
    print(f"📊 Analizando {Y.shape[0]} palabras y {Y.shape[1]} dimensiones...\n")

    # 1. LOOK FOR DEAD DIMENSIONS (No variance)
    print("💀 DIMENSIONES 'ZOMBIE' (Casi todo es 0 o constante):")
    zombies = []
    
    for i, dim in enumerate(dim_names):
        col = Y[:, i]
        variance = np.var(col)
        max_val = np.max(col)
        min_val = np.min(col)
        mean_val = np.mean(col)
        
        # If the variance is extremely low, the dimension is dead
        if variance < 0.001:
            zombies.append((dim, variance, max_val))
            # print(f"   - {dim[:30]:<30} | Var: {variance:.6f} | Max: {max_val:.6f}")

    # Sort by the most dead
    zombies.sort(key=lambda x: x[1])
    for z in zombies[:10]:
         print(f"   ❌ {z[0][:30]:<30} | Max Real: {z[2]:.6f} (Nadie supera esto)")

    print(f"\n   Total Dimensiones Muertas (<0.001 var): {len(zombies)} de {len(dim_names)}")

    # 2. SEE THE REAL RANKING OF A DIMENSION
    # Let's see which words are the "Kings" of Dangerousness according to your current dataset
    target_dim = "peligrosidad_social" # Partial search
    
    found_idx = -1
    for i, name in enumerate(dim_names):
        if target_dim in name:
            found_idx = i
            print(f"\n🔎 INSPECCIONANDO: '{name.upper()}'")
            break
            
    if found_idx != -1:
        # Get the values of that column
        col = Y[:, found_idx]

        # Sort indices from highest to lowest
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