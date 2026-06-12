import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# CONFIGURACIÓN
EPOCHS = 1500 # Ajustar según cantidad de datos (más datos = menos épocas necesarias)
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class SemanticTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Arquitectura "Embudo Invertido"
        # Queremos expandir la representación para encontrar correlaciones ocultas
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1), # Evita memorizar datos exactos
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, output_dim),
            nn.Sigmoid() # ¡Crucial! Fuerza la salida entre 0 y 1 (tus dimensiones)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # 1. Cargar Datos
    try:
        X = np.load("dataset_X_embeddings.npy")
        Y = np.load("dataset_Y_human.npy")
    except:
        print("❌ Ejecuta primero generate_training_data.py")
        return

    # Convertir a Tensores
    tensor_x = torch.Tensor(X) # (N, 384)
    tensor_y = torch.Tensor(Y) # (N, 104)

    # Split (80% Train, 20% Test) para ver si generaliza de verdad
    split_idx = int(len(X) * 0.8)
    train_ds = TensorDataset(tensor_x[:split_idx], tensor_y[:split_idx])
    test_ds = TensorDataset(tensor_x[split_idx:], tensor_y[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 2. Inicializar Red
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = SemanticTranslator(input_dim, output_dim).cuda() # ¡A la 4090!
    
    criterion = nn.MSELoss() # Error cuadrático medio (ideal para regresión)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🔥 Entrenando Traductor: {input_dim} inputs -> {output_dim} outputs")
    
    # 3. Bucle de Entrenamiento
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        loss_history.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 4. Guardar Modelo
    torch.save(model.state_dict(), "semantic_translator.pth")
    print("✅ Modelo entrenado y guardado.")

    # 5. Visualizar un ejemplo del Test Set
    print("\n--- TEST DE REALIDAD ---")
    model.eval()
    
    # Tomamos un ejemplo aleatorio del test set
    test_idx = 0 
    real_input = tensor_x[split_idx + test_idx].unsqueeze(0).cuda()
    real_target = tensor_y[split_idx + test_idx].cpu().numpy()
    
    with torch.no_grad():
        prediction = model(real_input).cpu().numpy()[0]
    
    print("Comparativa (Primeras 5 dimensiones):")
    print(f"Real:     {real_target[:5]}")
    print(f"Predicho: {prediction[:5]}")
    
    diff = np.abs(real_target - prediction)
    print(f"Error Medio Absoluto en este ejemplo: {np.mean(diff):.6f}")

if __name__ == "__main__":
    main()