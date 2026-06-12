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
BATCH_SIZE = 64
LEARNING_RATE = 0.0005 
SAVE_PATH = "reverse_translator.pth" # Will overwrite the old model

# Data files (Inputs and Targets inverted with respect to the normal translator)
HUMAN_DATA = "dataset_Y_human.npy"      # INPUT (What enters the network)
EMBEDDING_DATA = "dataset_X_embeddings.npy" # TARGET (What the network must imagine)

# ==========================================
# IMPROVED ARCHITECTURE (V2 - DEEP)
# ==========================================
class HumanToEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # "Deep Decoder" architecture
        # Goal: Decompress abstract concepts (few dims) into dense vectors (many dims)
        self.net = nn.Sequential(
            # Layer 1: Initial expansion
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2), # LeakyReLU avoids neuron death in generative networks
            nn.Dropout(0.1),

            # Layer 2: Intermediate processing
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            # Layer 3: "High Resolution" (Oversampling)
            # We expand to 1024 to capture subtle nuances before compressing
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            # Output Layer: Final compression into the embedding space
            nn.Linear(1024, output_dim),
            nn.Tanh() # Output between -1 and 1 (Crucial for cosine-normalized embeddings)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# TRAINING
# ==========================================
def main():
    print("🔥 INICIANDO ENTRENAMIENTO DEL SINTETIZADOR (V2 DEEP)...")
    
    # 1. Load Data
    if not os.path.exists(HUMAN_DATA) or not os.path.exists(EMBEDDING_DATA):
        print("❌ Error: Faltan los archivos .npy (Ejecuta primero 40_build_training_data.py)")
        return

    X_human = np.load(HUMAN_DATA)       # (N, 104) ~ Inputs
    Y_embed = np.load(EMBEDDING_DATA)   # (N, 384) ~ Targets

    # Convert to Tensors
    tensor_x = torch.FloatTensor(X_human)
    tensor_y = torch.FloatTensor(Y_embed)

    # Split 80/20
    split_idx = int(len(X_human) * 0.8)
    
    train_ds = TensorDataset(tensor_x[:split_idx], tensor_y[:split_idx])
    test_ds = TensorDataset(tensor_x[split_idx:], tensor_y[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 2. Initialize Network
    input_dim = X_human.shape[1]
    output_dim = Y_embed.shape[1]

    # Detect device (GPU if possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ⚙️  Dispositivo: {device}")
    
    model = HumanToEmbedding(input_dim, output_dim).to(device)
    
    # ⚠️ CRITICAL CHANGE: We use CosineEmbeddingLoss
    # This optimizes the ANGLE between vectors, not the exact Euclidean distance.
    # It is much better for semantic spaces (meaning).
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f"   🧠 Arquitectura: {input_dim} (Humano) -> [256 -> 512 -> 1024] -> {output_dim} (Embedding)")
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            prediction = model(batch_x)
            
            # Target for CosineLoss: 1 means "we want them to be equal"
            target_ones = torch.ones(batch_x.shape[0]).to(device)
            
            loss = criterion(prediction, batch_y, target_ones)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction = model(batch_x)
                target_ones = torch.ones(batch_x.shape[0]).to(device)
                loss = criterion(prediction, batch_y, target_ones)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        if (epoch + 1) % 100 == 0:
            print(f"   Época {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    # 4. Save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Modelo V2 guardado en: {SAVE_PATH}")
    
    # 5. Quick sanity test
    print("\n--- TEST DE CORDURA (Similitud Coseno) ---")
    model.eval()
    
    # We take an example from the test set
    idx = 0
    sample_in = tensor_x[split_idx + idx].unsqueeze(0).to(device)
    sample_target = tensor_y[split_idx + idx].cpu().numpy()
    
    with torch.no_grad():
        sample_pred = model(sample_in).cpu().numpy()[0]
    
    # Compute cosine similarity manually
    dot = np.dot(sample_pred, sample_target)
    norm_a = np.linalg.norm(sample_pred)
    norm_b = np.linalg.norm(sample_target)
    cos_sim = dot / (norm_a * norm_b)
    
    print(f"   Vector Target (Norm): {norm_b:.4f}")
    print(f"   Vector Predicho (Norm): {norm_a:.4f}")
    print(f"   👉 Similitud Coseno: {cos_sim:.4f} (Ideal: 1.0)")

if __name__ == "__main__":
    main()