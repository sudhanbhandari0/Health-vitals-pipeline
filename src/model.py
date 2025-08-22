import torch
from torch import nn
from pathlib import Path
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd

REQUIRED_VITALS = ["hr", "sbp", "dbp", "spo2"]  # 4 inputs

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, bottleneck: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

@dataclass
class TrainingConfig:
    epoches: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42

def _set_seed(seed: int):
    torch.manual_seed(seed)

def train_one_epoch(model:nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, x:torch.Tensor) -> float:
    model.train()
    total_loss = 0.0
    N = x.size(0)
    batch_size = 32

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = x[start:end]

        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * (end-start)

    return total_loss/ N

def load_scaled_train_matrix(train_csv: str | Path, scaler_path: str | Path) -> np.ndarray:

    from data import load_train_only

    train_df = load_train_only(train_csv)
    X = train_df[REQUIRED_VITALS].to_numpy(dtype=np.float32)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    return X_scaled



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_csv = project_root / "data" / "raw_vitals_normal.csv"
    scaler_path = project_root / "models" / "scaler.joblib"

    X_scaled = load_scaled_train_matrix(train_csv, scaler_path)
    print("Scaled training matrix shape:", X_scaled.shape)   # expect (N, 4)
    print("First row (scaled):", X_scaled[0])





# if __name__ == "__main__":
#     cfg = TrainingConfig(epoches=5, batch_size=32, lr=1e-3, seed=42)  # Use "epoches"
#     _set_seed(cfg.seed)

#     model = Autoencoder(input_dim=len(REQUIRED_VITALS), hidden_dim=8, bottleneck=2)
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
#     criterion = nn.MSELoss()

#     x = torch.randn(256, len(REQUIRED_VITALS))

#     for epoch in range(1, cfg.epoches + 1):  # Use cfg.epoches
#         avg_loss = train_one_epoch(model, optimizer, criterion, x)
#         print(f"epoch {epoch}/{cfg.epoches}  loss={avg_loss:.6f}")

#     with torch.no_grad():
#         y = model(x[:5])
#     print("input shape :", x[:5].shape)
#     print("output shape:", y.shape)