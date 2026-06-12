import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_PATH = "semantic_translator.pth"
METADATA_FILE = "dataset_metadata.json"
DATA_X_FILE = "dataset_X_embeddings.npy"
DATA_Y_FILE = "dataset_Y_human.npy"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Debe coincidir con el usado en training

# Definición de la Red (Debe ser idéntica a la usada en training)
class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def load_data():
    if not os.path.exists(METADATA_FILE):
        print("❌ Error: No encuentro metadatos.")
        return None, None, None, None
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    X = np.load(DATA_X_FILE)
    Y = np.load(DATA_Y_FILE)
    
    return X, Y, meta["concepts"], meta["dimension_names"]

def analyze_dimension_performance(Y_true, Y_pred, dim_names):
    """
    Calcula el error promedio por dimensión para ver cuáles son aprendibles y cuáles no.
    """
    errors = np.abs(Y_true - Y_pred) # Matriz de errores absolutos
    mean_errors = np.mean(errors, axis=0) # Promedio por columna (dimensión)
    
    # Crear DataFrame para ordenar
    df = pd.DataFrame({
        "Dimension": dim_names,
        "MAE": mean_errors
    }).sort_values(by="MAE")
    
    print("\n🏆 TOP 5 DIMENSIONES MEJOR APRENDIDAS (Menor Error):")
    print(df.head(5).to_string(index=False))
    
    print("\n💀 TOP 5 DIMENSIONES PEORES (Mayor Error - Posible Ruido/Alucinación):")
    print(df.tail(5).to_string(index=False))
    
    return df

def test_manual_words(model, embedder, dim_names):
    """
    Permite probar palabras nuevas que NO estaban en el dataset original.
    """
    print("\n🧪 --- TEST DE GENERALIZACIÓN (Palabras Nuevas) ---")
    test_words = ["amor", "pistola", "microbio", "galaxia", "mentira"]
    
    embeddings = embedder.encode(test_words)
    tensor_x = torch.Tensor(embeddings)
    
    model.eval()
    with torch.no_grad():
        predictions = model(tensor_x).numpy()
    
    # Mostrar resultados clave para cada palabra
    # Seleccionamos algunas dimensiones interesantes para mostrar
    target_dims = ["tamaño_físico", "peligrosidad", "positividad", "tangibilidad", "temperatura"]
    
    # Encontrar índices de esas dimensiones
    indices = []
    for target in target_dims:
        # Buscamos coincidencias parciales (ej. "d000_tamaño_físico")
        for i, name in enumerate(dim_names):
            if target in name.lower():
                indices.append((i, name))
                break
    
    for i, word in enumerate(test_words):
        print(f"\n🔹 Palabra: {word.upper()}")
        for idx, dim_name in indices:
            val = predictions[i][idx]
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {dim_name[:25]:<25} : {val:.4f} {bar}")

def main():
    print("🔍 Iniciando Auditoría del Traductor Semántico...")
    
    # 1. Cargar Datos y Modelo
    X, Y, concepts, dim_names = load_data()
    if X is None: return

    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    model = SemanticTranslator(input_dim, output_dim)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("✅ Modelo cargado correctamente.")
    except:
        print(f"❌ No encuentro '{MODEL_PATH}'. Entrena primero.")
        return

    # 2. Generar Predicciones Masivas (Dataset Completo)
    tensor_X = torch.Tensor(X)
    with torch.no_grad():
        Y_pred = model(tensor_X).numpy()

    # 3. Métricas Globales
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    print(f"\n📊 ERROR GLOBAL DEL MODELO:")
    print(f"   MAE (Error Absoluto Medio): {mae:.4f} (en escala 0-1)")
    print(f"   MSE (Error Cuadrático Medio): {mse:.4f}")
    
    if mae < 0.1:
        print("   ✅ Resultado: EXCELENTE. La red ha capturado la estructura.")
    elif mae < 0.2:
        print("   ⚠️ Resultado: ACEPTABLE. Hay ruido, pero funciona.")
    else:
        print("   ❌ Resultado: POBRE. La red no está convergiendo (datos insuficientes o ruidosos).")

    # 4. Análisis por Dimensión
    df_metrics = analyze_dimension_performance(Y, Y_pred, dim_names)
    
    # 5. Análisis Visual de un Caso Concreto
    # Buscamos una palabra específica si existe, si no, una aleatoria
    target_word = "acero inoxidable"
    if target_word in concepts:
        idx = concepts.index(target_word)
        print(f"\n🔬 ANÁLISIS DETALLADO: '{target_word}'")
        
        real = Y[idx]
        pred = Y_pred[idx]
        
        diff = np.abs(real - pred)
        
        # Mostramos las dimensiones con mayor error para esta palabra
        worst_indices = np.argsort(diff)[-5:][::-1] # Top 5 errores
        
        print("   Dimensiones con MAYOR DISCREPANCIA (Real vs Predicho):")
        for i in worst_indices:
            d_name = dim_names[i]
            print(f"   ❌ {d_name}: Real={real[i]:.4f} | Pred={pred[i]:.4f} | Diff={diff[i]:.4f}")
    
    # 6. Test de Generalización (Palabras Nuevas)
    print("\n🔄 Cargando SentenceTransformer para pruebas en vivo...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_manual_words(model, embedder, dim_names)

if __name__ == "__main__":
    main()