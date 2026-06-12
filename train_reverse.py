import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
EPOCHS = 1500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "reverse_translator.pth" # El modelo "Generador"

# Archivos generados previamente (Invertimos el orden lógico)
# AHORA: INPUT = Human Vectors, TARGET = Embeddings
INPUT_DATA = "dataset_Y_human.npy" 
TARGET_DATA = "dataset_X_embeddings.npy"

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Arquitectura "Expansiva" (De pocas dimensiones a muchas)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, output_dim),
            # IMPORTANTE: No usamos Sigmoid aquí.
            # Los embeddings viven en espacio vectorial abierto (-1 a 1 aprox)
            nn.Tanh() # Tanh ayuda a mantener los valores entre -1 y 1
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("🔄 Iniciando Entrenamiento Inverso (Concepto -> Embedding)...")
    
    if not os.path.exists(INPUT_DATA):
        print("❌ Faltan los archivos .npy. Ejecuta generate_training_data.py primero.")
        return

    # Cargar datos
    # Note: X_human es nuestra entrada ahora
    X_human = np.load(INPUT_DATA)      # (N, 104)
    Y_embeddings = np.load(TARGET_DATA) # (N, 384)

    # Convertir a Tensores
    tensor_x = torch.Tensor(X_human)
    tensor_y = torch.Tensor(Y_embeddings)

    # Dataset
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configurar Red
    input_dim = X_human.shape[1]
    output_dim = Y_embeddings.shape[1]
    
    model = HumanToEmbedding(input_dim, output_dim).cuda()
    
    # CosineEmbeddingLoss es ideal, pero MSELoss es más estable para empezar
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🧠 Red Configurada: {input_dim} entradas (Tus dims) -> {output_dim} salidas (Embedding)")

    # Loop de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f"   Epoch {epoch}/{EPOCHS} | Loss: {running_loss / len(dataloader):.6f}")

    # Guardar
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Modelo Inverso guardado en: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()