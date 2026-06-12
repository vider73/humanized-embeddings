import json
import os
import difflib # For smart suggestions

INPUT_FILE = "humanized_embeddings_dataset.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ No encuentro el archivo humanized_embeddings_dataset.json")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_words = list(data.keys())
    print(f"📊 Dataset cargado: {len(all_words)} palabras.")
    print("💡 Escribe una palabra para ver su 'Verdad Ground Truth' (según Llama 3).")
    
    while True:
        query = input("\n🔍 Palabra: ").lower().strip()
        if query in ['salir', 'exit', 'q']: break
        
        # Look for the exact word
        if query in data:
            match = query
        else:
            # Look for a similar one
            matches = difflib.get_close_matches(query, all_words, n=3, cutoff=0.6)
            if matches:
                print(f"⚠️ No encontré '{query}'. Quizás quisiste decir: {matches}")
                match = matches[0] # We assume the first one
                op = input(f"¿Analizar '{match}'? (s/n): ")
                if op.lower() != 's': continue
            else:
                print("❌ Palabra no encontrada en el entrenamiento.")
                continue

        print(f"\n🧠 ANÁLISIS DE '{match.upper()}' (Según Llama 3):")
        print("-" * 40)
        
        vec = data[match]
        
        # Sort dimensions by value
        sorted_items = sorted(vec.items(), key=lambda x: x[1], reverse=True)
        
        print("🔥 TOP 10 DIMENSIONES MÁS ACTIVAS:")
        for k, v in sorted_items[:10]:
            bar_len = int(v * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {k:<30} : {v:.6f} {bar}")

        print("\n❄️ TOP 5 DIMENSIONES MENOS ACTIVAS (Cerca de 0):")
        for k, v in sorted_items[-5:]:
            print(f"  {k:<30} : {v:.6f}")
            
        # Physical anomaly detection
        phys_check = vec.get("d004_temperatura", 0)
        size_check = vec.get("d000_tamaño_físico", 0)
        
        if phys_check < 0.01 and size_check < 0.01:
            print("\n⚠️ ALERTA: Esta palabra tiene valores físicos EXTREMADAMENTE BAJOS.")
            print("   Esto sugiere que Llama 3 la comparó con el 'Universo' y colapsó a 0.")
            print("   (La Red Neuronal tendrá problemas aprendiendo esto).")

if __name__ == "__main__":
    main()