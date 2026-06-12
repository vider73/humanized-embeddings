import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# CONFIGURATION
# ==========================================
EPOCHS = 1500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "reverse_translator.pth" # The "Generator" model

# Files generated previously (We invert the logical order)
# NOW: INPUT = Human Vectors, TARGET = Embeddings
INPUT_DATA = "dataset_Y_human.npy" 
TARGET_DATA = "dataset_X_embeddings.npy"

class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # "Expansive" architecture (From few dimensions to many)
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
            # IMPORTANT: We do not use Sigmoid here.
            # Embeddings live in an open vector space (-1 to 1 approx.)
            nn.Tanh() # Tanh helps keep the values between -1 and 1
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("🔄 Iniciando Entrenamiento Inverso (Concepto -> Embedding)...")
    
    if not os.path.exists(INPUT_DATA):
        print("❌ Faltan los archivos .npy. Ejecuta 40_build_training_data.py primero.")
        return

    # Load data
    # Note: X_human is our input now
    X_human = np.load(INPUT_DATA)      # (N, 104)
    Y_embeddings = np.load(TARGET_DATA) # (N, 384)

    # Convert to Tensors
    tensor_x = torch.Tensor(X_human)
    tensor_y = torch.Tensor(Y_embeddings)

    # Dataset
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configure Network
    input_dim = X_human.shape[1]
    output_dim = Y_embeddings.shape[1]
    
    model = HumanToEmbedding(input_dim, output_dim).cuda()
    
    # CosineEmbeddingLoss is ideal, but MSELoss is more stable to start with
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🧠 Red Configurada: {input_dim} entradas (Tus dims) -> {output_dim} salidas (Embedding)")

    # Training Loop
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

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Modelo Inverso guardado en: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()