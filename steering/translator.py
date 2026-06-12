"""
translator.py — Tu humanizador, reutilizado tal cual.

texto --MiniLM--> 384d --SemanticTranslator--> perfil de 104 dims (0..1)

Es el MISMO modelo que ya entrenaste (semantic_translator.pth). Aqui solo
lo envolvemos para producir el "perfil" que luego pilotara al LLM.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from . import config


class SemanticTranslator(nn.Module):
    """Identica a train_translator.py — no se reentrena, solo se carga."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),       nn.ReLU(),                       nn.Dropout(0.1),
            nn.Linear(256, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Humanizer:
    """Texto -> perfil de 104 dims. Carga MiniLM + tu traductor una sola vez."""

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
        self.dim_names = meta["dimension_names"]
        self.human_dim = len(self.dim_names)
        self.device = device

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
        in_dim = self.embedder.get_sentence_embedding_dimension()

        self.translator = SemanticTranslator(in_dim, self.human_dim).to(device)
        state = torch.load(config.TRANSLATOR_PATH, map_location=device)
        self.translator.load_state_dict(state)
        self.translator.eval()

    @torch.no_grad()
    def profile(self, text: str) -> np.ndarray:
        """Devuelve el perfil de 104 dims (numpy float32, rango ~0..1)."""
        emb = self.embedder.encode([text], convert_to_numpy=True)
        t = torch.tensor(emb, dtype=torch.float32, device=self.device)
        # BatchNorm necesita >1 muestra en train; en eval usa stats guardadas, ok con 1.
        out = self.translator(t).cpu().numpy()[0]
        return out.astype(np.float32)

    def top_dims(self, prof: np.ndarray, n: int = 6):
        order = np.argsort(prof)[::-1][:n]
        return [(int(i), self.dim_names[i], float(prof[i])) for i in order]
